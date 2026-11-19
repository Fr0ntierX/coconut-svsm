// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2026 Fr0ntierX
//
// Watchdog for partition vCPUs (uPolaris CMVM model B, threat model section
// 9.3, F0X-75).
//
// VMPL0 has no timer on hosts without alternate injection, and a vCPU that
// loops on nested page faults (a write to a page its VMPL lost access to)
// never hands control to VMPL0. So VMPL0 only learns that a vCPU is stuck if
// another vCPU calls in and VMPL0 has a progress signal it can check from
// there. That is this module:
//
// * Each watched vCPU has a 64-bit heartbeat counter in guest memory, which
//   the trusted CPL0 kernel bumps from its timer tick on that vCPU. CPL3
//   partitions cannot write it, and a kernel that stops making progress stops
//   bumping it. VMPL0 only reads it.
// * One vCPU (the watchdog vCPU) runs no partition and calls `watchdog_tick`
//   on a fixed period. The first vCPU to tick becomes the watchdog vCPU, and
//   ticks from any other vCPU are refused.
// * On each tick VMPL0 reads every armed slot's heartbeat. A changed value is
//   recorded with the current TSC. A value unchanged for longer than the
//   slot's deadline makes VMPL0 retire that vCPU with `retire_guest_vcpu`,
//   which also revokes the slot's partition pages. The kernel cannot veto it.
//
// A retire that fails is retried on later ticks when a retry can help, up to
// a per-slot bound, and then tried once more with `RetireOptions::force` as
// the last resort (see `RetryPolicy`). Force goes on to the VMSA RMPADJUST
// even when the host refuses AP_DESTROY. On a vCPU the host still runs, that
// stops the whole VM at the target's next VMRUN (measured on the lab part, 19
// Nov 2026). A watchdog that reaches force on a running target trades the VM
// for the reclaim. That is why force comes last, after the host has refused
// every plain attempt, and why a slot can be armed without it.
//
// The kernel must keep a watched vCPU out of its own IPI and NMI targets for
// as long as it is watched, because a stuck vCPU cannot run an offline
// handshake, and an NMI to a retired vCPU stops the VM.
//
// Limits: without ticks nothing is checked (host denial of service, or a
// failure of the trusted kernel, which is in the TCB). VMPL0 cannot tell a
// vCPU that stopped from one the host did not schedule for longer than the
// deadline, so the deadline must sit well above host scheduling jitter.

#![cfg_attr(
    not(feature = "parttest"),
    expect(
        dead_code,
        reason = "parttest is the only caller until the CMVM monitor lands"
    )
)]

extern crate alloc;

use crate::address::{Address, PhysAddr};
use crate::cpu::msr::rdtsc;
use crate::cpu::percpu::{PERCPU_VMSAS, this_cpu};
use crate::error::SvsmError;
use crate::locking::SpinLock;
use crate::mm::{PerCPUPageMappingGuard, valid_phys_region};
use crate::protocols::core::{
    RETIRE_DEFAULT_INUSE_DEADLINE, RetireError, RetireOptions, RetireStats, first_invalid_page,
    retire_guest_vcpu,
};
use crate::types::PAGE_SIZE;
use crate::utils::MemoryRegion;
use alloc::vec::Vec;

/// Slots in the watch table. The tick reports slots as bitmasks, and the
/// parttest tick call packs two of them into one register, so this must stay
/// at or below 16.
pub(crate) const WATCHDOG_SLOTS: usize = 16;
/// Partition pages one slot can carry for the revoke on reclaim.
pub(crate) const WATCHDOG_MAX_PAGES: usize = 64;
/// Plain retries after a failed reclaim, before the forced one, when the
/// slot was armed with the default policy.
pub(crate) const WATCHDOG_DEFAULT_RETRIES: u16 = 3;
/// How long one watchdog retire keeps retrying an RMPADJUST that returns
/// FAIL_INUSE, in TSC ticks (about 0.3 ms on the lab part). Shorter than
/// the helper's default because a Busy retire is retried on the next tick
/// anyway, and the retire holds up every later retire in its tick.
const WATCHDOG_INUSE_DEADLINE: u64 = RETIRE_DEFAULT_INUSE_DEADLINE / 10;

const _: () = assert!(WATCHDOG_SLOTS <= 16);

/// What a slot does when a reclaim fails with an error a retry can help
/// (`ApDestroy`, `Busy`, `NoMapping`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RetryPolicy {
    /// Plain retries after the first failed attempt, one per tick.
    pub retries: u16,
    /// Also stop the plain retries once this many TSC ticks have passed
    /// since the reclaim was decided. 0 means no time bound.
    pub window: u64,
    /// After the plain retries, try once more with `RetireOptions::force`.
    /// Without it the slot ends `Failed`. Force on a vCPU the host still
    /// runs stops the VM, see the module comment.
    pub force: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            retries: WATCHDOG_DEFAULT_RETRIES,
            window: 0,
            force: true,
        }
    }
}

/// What one failed reclaim attempt means for its slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfterFailure {
    /// Try again on the next tick, with force if set.
    Retry { force: bool },
    /// The vCPU is retired but the clean-up after it failed. Retrying the
    /// retire would only be refused, so the slot is finished.
    Gone,
    /// Give up. A retry cannot help, or the bound and the forced attempt are
    /// used up.
    GiveUp,
}

/// Decide what follows a failed attempt. `forced` says whether that attempt
/// used force, `retries_done` how many attempts came before it, and
/// `elapsed` the TSC ticks since the reclaim was decided.
fn after_failure(
    e: RetireError,
    forced: bool,
    retries_done: u32,
    elapsed: u64,
    policy: &RetryPolicy,
) -> AfterFailure {
    match e {
        // After the VMSA RMPADJUST, so the vCPU can never run again.
        RetireError::PageRevoke { .. } | RetireError::VmsaRelease(_) => AfterFailure::Gone,
        // The host refused, the RMPADJUST stayed busy, or a mapping failed.
        // None of these changed the RMP, so the next attempt starts clean.
        RetireError::ApDestroy(_) | RetireError::Busy(_) | RetireError::NoMapping => {
            if forced {
                AfterFailure::GiveUp
            } else if retries_done < u32::from(policy.retries)
                && (policy.window == 0 || elapsed < policy.window)
            {
                AfterFailure::Retry { force: false }
            } else if policy.force {
                AfterFailure::Retry { force: true }
            } else {
                AfterFailure::GiveUp
            }
        }
        // The VMSA is no longer a guest VMSA (another call deleted or
        // retired it), the page list became invalid, or the RMPADJUST
        // returned a status that will not change. Force cannot help either.
        RetireError::InvalidTarget | RetireError::InvalidPage(_) | RetireError::VmsaAdjust(_) => {
            AfterFailure::GiveUp
        }
    }
}

/// Where a slot is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlotState {
    Free,
    Armed,
    /// The deadline passed and a retire is running outside the table lock.
    Reclaiming,
    /// A retire failed with this error and the next tick tries again. The
    /// heartbeat is no longer checked: once the deadline has passed the
    /// reclaim goes ahead whatever the vCPU does next.
    Retrying(RetireError),
    Reclaimed,
    /// The vCPU is retired, but revoking its pages or handing back its VMSA
    /// page failed with this error. Finished, never retried.
    ReclaimedPartly(RetireError),
    /// The reclaim failed with this error and is not retried, either because
    /// a retry cannot help or because the policy's attempts are used up. See
    /// `RetireError` for what it left behind.
    Failed(RetireError),
}

impl SlotState {
    /// Armed, or past its deadline and not finished. A VMSA in a live slot
    /// cannot be armed again.
    fn live(self) -> bool {
        matches!(self, Self::Armed | Self::Reclaiming | Self::Retrying(_))
    }
}

/// Why an arm or a tick was refused.
#[derive(Debug, Clone, Copy)]
pub(crate) enum WatchdogError {
    /// Not a guest-created in-use VMSA, or it belongs to the watchdog vCPU.
    BadTarget,
    /// The heartbeat is not an 8-byte aligned guest address, or its page is a
    /// VMSA or calling area.
    BadHeartbeat,
    /// The partition page at this index is not a 4K guest page, or it is a
    /// VMSA or calling area.
    BadPage(usize),
    /// Too many partition pages.
    TooManyPages,
    /// The VMSA is already watched by a slot that is armed or reclaiming.
    AlreadyWatched,
    /// No free slot.
    Full,
    /// No such slot, or it is not in a state the call accepts.
    BadSlot,
    /// The tick came from a vCPU other than the watchdog vCPU.
    NotWatchdogCpu,
    /// The heartbeat could not be read when arming.
    Read(SvsmError),
}

/// One watched vCPU.
#[derive(Debug, Clone)]
struct Slot {
    state: SlotState,
    vmsa_pa: PhysAddr,
    heartbeat_pa: PhysAddr,
    cpu_index: usize,
    deadline: u64,
    pages: Vec<PhysAddr>,
    last_value: u64,
    last_change: u64,
    armed_at: u64,
    changes_seen: u64,
    max_gap: u64,
    max_still: u64,
    read_errors: u64,
    reclaim_decided: u64,
    stale_at_reclaim: u64,
    retire_started: u64,
    retire: RetireStats,
    policy: RetryPolicy,
    /// Retire attempts so far. Retries are this minus one.
    attempts: u32,
    /// The next attempt uses force.
    force_next: bool,
    /// An attempt used force.
    forced: bool,
    /// Error of the first failed attempt.
    first_error: Option<RetireError>,
}

impl Slot {
    const fn free() -> Self {
        Self {
            state: SlotState::Free,
            vmsa_pa: PhysAddr::null(),
            heartbeat_pa: PhysAddr::null(),
            cpu_index: 0,
            deadline: 0,
            pages: Vec::new(),
            last_value: 0,
            last_change: 0,
            armed_at: 0,
            changes_seen: 0,
            max_gap: 0,
            max_still: 0,
            read_errors: 0,
            reclaim_decided: 0,
            stale_at_reclaim: 0,
            retire_started: 0,
            retire: RetireStats {
                ap_destroy_ticks: 0,
                lock_ticks: 0,
                vmsa_ticks: 0,
                total_ticks: 0,
                inuse_retries: 0,
                pages_revoked: 0,
                ap_destroy_refused: false,
            },
            policy: RetryPolicy {
                retries: WATCHDOG_DEFAULT_RETRIES,
                window: 0,
                force: true,
            },
            attempts: 0,
            force_next: false,
            forced: false,
            first_error: None,
        }
    }
}

struct Watchdog {
    slots: [Slot; WATCHDOG_SLOTS],
    tick_cpu: Option<usize>,
    ticks: u64,
    last_tick: u64,
}

static WATCHDOG: SpinLock<Watchdog> = SpinLock::new(Watchdog {
    slots: [const { Slot::free() }; WATCHDOG_SLOTS],
    tick_cpu: None,
    ticks: 0,
    last_tick: 0,
});

/// A snapshot of one slot, for status calls.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SlotStatus {
    pub state: SlotState,
    pub vmsa_pa: PhysAddr,
    pub cpu_index: usize,
    pub deadline: u64,
    pub last_value: u64,
    /// TSC of the tick that first saw `last_value`.
    pub last_change: u64,
    pub armed_at: u64,
    /// Ticks that saw the heartbeat move.
    pub changes_seen: u64,
    /// Longest gap between two ticks that saw the heartbeat move, in TSC
    /// ticks. This includes late ticks, so it can exceed the deadline
    /// without a reclaim.
    pub max_gap: u64,
    /// Longest time the heartbeat had been still at a tick that saw no
    /// change, without passing the deadline. How close a vCPU came to a
    /// reclaim: a false positive needs this to pass the deadline.
    pub max_still: u64,
    pub read_errors: u64,
    /// TSC at which the tick decided to reclaim, just before the retire.
    pub reclaim_decided: u64,
    /// How long the heartbeat had been still when the tick decided.
    pub stale_at_reclaim: u64,
    /// TSC at which this slot's last retire attempt started. Later than
    /// `reclaim_decided` when an earlier slot was retired in the same tick,
    /// or when the reclaim needed retries.
    pub retire_started: u64,
    /// Stats of the last retire attempt.
    pub retire: RetireStats,
    pub policy: RetryPolicy,
    /// Retire attempts after the first, the forced one included.
    pub retries: u32,
    /// The next attempt uses force (only while `Retrying`).
    pub force_next: bool,
    /// An attempt used force. `retire.ap_destroy_refused` then says whether
    /// the host refused AP_DESTROY on it, so force decided the outcome.
    pub forced: bool,
    /// Error of the first failed attempt, if one failed.
    pub first_error: Option<RetireError>,
    pub watchdog_ticks: u64,
    pub last_tick: u64,
}

/// What one tick did.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TickReport {
    /// Armed slots whose heartbeat was checked.
    pub checked: u32,
    /// Bit n set: slot n's vCPU was retired in this tick (`Reclaimed` or
    /// `ReclaimedPartly`).
    pub reclaimed: u64,
    /// Bit n set: slot n finished with an error in this tick (`Failed`, or
    /// `ReclaimedPartly`, which also sets its `reclaimed` bit).
    pub failed: u64,
    /// Bit n set: slot n's retire failed in this tick and is retried on the
    /// next.
    pub retrying: u64,
    /// VMPL0 time for the whole tick, retires included.
    pub ticks: u64,
}

fn read_heartbeat(pa: PhysAddr) -> Result<u64, SvsmError> {
    let guard = fault_inject!(
        HeartbeatRead,
        _ => Err(SvsmError::Fault),
        PerCPUPageMappingGuard::create_4k(pa.page_align())
    )?;
    guard.guest_ptr::<u64>(pa.page_offset())?.read()
}

fn heartbeat_ok(pa: PhysAddr) -> bool {
    let page = pa.page_align();
    pa.is_aligned(8)
        && MemoryRegion::checked_new(page, PAGE_SIZE)
            .is_some_and(|r| valid_phys_region(&r) && !PERCPU_VMSAS.overlaps(&r))
}

/// Watch the vCPU running the guest-created VMSA at `vmsa_pa`. Its heartbeat
/// is the u64 at `heartbeat_pa`, and it is retired once that value has not
/// changed for `deadline` TSC ticks. `pages` are revoked from every guest
/// VMPL when it is retired. `policy` says how a failed retire is retried.
/// Returns the slot index.
pub(crate) fn watchdog_arm(
    vmsa_pa: PhysAddr,
    heartbeat_pa: PhysAddr,
    deadline: u64,
    pages: Vec<PhysAddr>,
    policy: RetryPolicy,
) -> Result<usize, WatchdogError> {
    if pages.len() > WATCHDOG_MAX_PAGES {
        return Err(WatchdogError::TooManyPages);
    }
    if deadline == 0 {
        return Err(WatchdogError::BadSlot);
    }
    let cpu_index = match PERCPU_VMSAS.in_use_owner(vmsa_pa) {
        Some((index, true)) if vmsa_pa.is_page_aligned() => index,
        _ => return Err(WatchdogError::BadTarget),
    };
    if !heartbeat_ok(heartbeat_pa) {
        return Err(WatchdogError::BadHeartbeat);
    }
    if let Some(i) = first_invalid_page(&pages) {
        return Err(WatchdogError::BadPage(i));
    }
    let value = read_heartbeat(heartbeat_pa).map_err(WatchdogError::Read)?;

    let mut wd = WATCHDOG.lock();
    if wd.tick_cpu == Some(cpu_index) {
        return Err(WatchdogError::BadTarget);
    }
    if wd
        .slots
        .iter()
        .any(|s| s.state.live() && s.vmsa_pa == vmsa_pa)
    {
        return Err(WatchdogError::AlreadyWatched);
    }
    let free = fault_inject!(
        WatchdogFull,
        _ => None,
        wd.slots.iter().position(|s| s.state == SlotState::Free)
    );
    let index = free.ok_or(WatchdogError::Full)?;
    let now = rdtsc();
    wd.slots[index] = Slot {
        state: SlotState::Armed,
        vmsa_pa,
        heartbeat_pa,
        cpu_index,
        deadline,
        pages,
        last_value: value,
        last_change: now,
        armed_at: now,
        policy,
        ..Slot::free()
    };
    Ok(index)
}

/// Stop watching a slot and free it. An armed slot is simply dropped, and a
/// finished one (reclaimed or failed) is cleared. A retrying slot is dropped
/// too, which gives up the reclaim and leaves the vCPU as the last attempt
/// left it. A slot whose retire is running is refused.
pub(crate) fn watchdog_disarm(index: usize) -> Result<(), WatchdogError> {
    let mut wd = WATCHDOG.lock();
    let slot = wd.slots.get_mut(index).ok_or(WatchdogError::BadSlot)?;
    match slot.state {
        SlotState::Free | SlotState::Reclaiming => Err(WatchdogError::BadSlot),
        _ => {
            *slot = Slot::free();
            Ok(())
        }
    }
}

pub(crate) fn watchdog_status(index: usize) -> Result<SlotStatus, WatchdogError> {
    let wd = WATCHDOG.lock();
    let s = wd.slots.get(index).ok_or(WatchdogError::BadSlot)?;
    Ok(SlotStatus {
        state: s.state,
        vmsa_pa: s.vmsa_pa,
        cpu_index: s.cpu_index,
        deadline: s.deadline,
        last_value: s.last_value,
        last_change: s.last_change,
        armed_at: s.armed_at,
        changes_seen: s.changes_seen,
        max_gap: s.max_gap,
        max_still: s.max_still,
        read_errors: s.read_errors,
        reclaim_decided: s.reclaim_decided,
        stale_at_reclaim: s.stale_at_reclaim,
        retire_started: s.retire_started,
        retire: s.retire,
        policy: s.policy,
        retries: s.attempts.saturating_sub(1),
        force_next: s.force_next,
        forced: s.forced,
        first_error: s.first_error,
        watchdog_ticks: wd.ticks,
        last_tick: wd.last_tick,
    })
}

/// One reclaim attempt the tick runs outside the table lock.
struct Attempt {
    index: usize,
    vmsa_pa: PhysAddr,
    pages: Vec<PhysAddr>,
    force: bool,
    /// Order within the tick: 0 first attempts, 1 plain retries, 2 forced.
    pass: u8,
}

/// What one attempt did, for the log after the last retire.
struct Done {
    index: usize,
    vmsa_pa: PhysAddr,
    res: Result<(), RetireError>,
    stats: RetireStats,
    state: SlotState,
    retries: u32,
    forced: bool,
    force_next: bool,
}

/// One watchdog period, called from the watchdog vCPU. Checks every armed
/// slot, retires the vCPUs past their deadline, and retries the reclaims
/// that failed on an earlier tick. Retires run outside the table lock, so
/// arm, disarm and status calls on other vCPUs never wait on the host's part
/// of a retire.
///
/// First attempts run before retries, and forced retries run last, so a slot
/// whose host keeps refusing never delays a fresh reclaim, and a forced
/// retire that may stop the VM comes after every other reclaim in the tick.
/// One slot's failure never stops the others being checked or retired.
pub(crate) fn watchdog_tick() -> Result<TickReport, WatchdogError> {
    let t0 = rdtsc();
    let me = this_cpu().get_cpu_index();
    let mut report = TickReport::default();
    let mut attempts: Vec<Attempt> = Vec::new();

    {
        let mut wd = WATCHDOG.lock();
        match wd.tick_cpu {
            None => wd.tick_cpu = Some(me),
            Some(cpu) if cpu != me => return Err(WatchdogError::NotWatchdogCpu),
            Some(_) => {}
        }
        wd.ticks += 1;
        wd.last_tick = t0;

        for (i, slot) in wd.slots.iter_mut().enumerate() {
            let pass = match slot.state {
                SlotState::Armed => {
                    report.checked += 1;
                    let now = rdtsc();
                    match read_heartbeat(slot.heartbeat_pa) {
                        Ok(v) if v != slot.last_value => {
                            slot.max_gap = slot.max_gap.max(now - slot.last_change);
                            slot.last_value = v;
                            slot.last_change = now;
                            slot.changes_seen += 1;
                            continue;
                        }
                        Ok(_) => {}
                        // A heartbeat VMPL0 cannot read counts as no progress.
                        Err(_) => slot.read_errors += 1,
                    }
                    let stale = now.saturating_sub(slot.last_change);
                    if stale <= slot.deadline {
                        slot.max_still = slot.max_still.max(stale);
                        continue;
                    }
                    slot.stale_at_reclaim = stale;
                    slot.reclaim_decided = now;
                    0
                }
                SlotState::Retrying(_) if slot.force_next => 2,
                SlotState::Retrying(_) => 1,
                _ => continue,
            };
            if attempts.try_reserve(1).is_err() {
                // The slot keeps its state and is tried again next tick.
                continue;
            }
            // The retire takes the page list for the attempt. A failed
            // attempt that will be retried puts it back.
            attempts.push(Attempt {
                index: i,
                vmsa_pa: slot.vmsa_pa,
                pages: core::mem::take(&mut slot.pages),
                force: slot.force_next,
                pass,
            });
            slot.state = SlotState::Reclaiming;
            slot.attempts += 1;
        }
    }

    // Retire before logging anything. A log line goes out over the serial
    // console and would hold up the next retire by milliseconds.
    let mut done: Vec<Done> = Vec::new();
    for pass in 0..3 {
        for a in attempts.iter_mut().filter(|a| a.pass == pass) {
            let opts = RetireOptions {
                inuse_deadline: WATCHDOG_INUSE_DEADLINE,
                force: a.force,
                log: false,
            };
            let started = rdtsc();
            let res = retire_guest_vcpu(a.vmsa_pa, &a.pages, &opts);
            let now = rdtsc();
            let bit = 1u64 << a.index;
            let mut wd = WATCHDOG.lock();
            let slot = &mut wd.slots[a.index];
            slot.retire_started = started;
            slot.forced |= a.force;
            let (res, stats) = match res {
                Ok(stats) => {
                    slot.state = SlotState::Reclaimed;
                    report.reclaimed |= bit;
                    (Ok(()), stats)
                }
                Err((e, stats)) => {
                    slot.first_error.get_or_insert(e);
                    let elapsed = now.saturating_sub(slot.reclaim_decided);
                    match after_failure(e, a.force, slot.attempts - 1, elapsed, &slot.policy) {
                        AfterFailure::Retry { force } => {
                            slot.state = SlotState::Retrying(e);
                            slot.force_next = force;
                            slot.pages = core::mem::take(&mut a.pages);
                            report.retrying |= bit;
                        }
                        AfterFailure::Gone => {
                            slot.state = SlotState::ReclaimedPartly(e);
                            slot.force_next = false;
                            report.reclaimed |= bit;
                            report.failed |= bit;
                        }
                        AfterFailure::GiveUp => {
                            slot.state = SlotState::Failed(e);
                            slot.force_next = false;
                            report.failed |= bit;
                        }
                    }
                    (Err(e), stats)
                }
            };
            slot.retire = stats;
            let entry = Done {
                index: a.index,
                vmsa_pa: a.vmsa_pa,
                res,
                stats,
                state: slot.state,
                retries: slot.attempts - 1,
                forced: a.force,
                force_next: slot.force_next,
            };
            drop(wd);
            if done.try_reserve(1).is_ok() {
                done.push(entry);
            }
        }
    }
    report.ticks = rdtsc() - t0;

    for d in done {
        let i = d.index;
        let vmsa_pa = d.vmsa_pa;
        let s = d.stats;
        let how = match (d.forced, s.ap_destroy_refused) {
            (true, true) => ", forced past a refused AP_DESTROY",
            (true, false) => ", forced attempt",
            _ => "",
        };
        match (d.res, d.state) {
            (Ok(()), _) => log::info!(
                "watchdog: slot {i} VMSA {vmsa_pa:#x} retired after {} retries{how}: ap_destroy {} lock {} vmsa {} total {} ticks, inuse {}, pages {}",
                d.retries,
                s.ap_destroy_ticks,
                s.lock_ticks,
                s.vmsa_ticks,
                s.total_ticks,
                s.inuse_retries,
                s.pages_revoked,
            ),
            (Err(e), SlotState::Retrying(_)) => log::warn!(
                "watchdog: slot {i} VMSA {vmsa_pa:#x} retire failed: {e:?} (retry {}{how}), next tick{}",
                d.retries,
                if d.force_next { " with force" } else { "" }
            ),
            (Err(e), SlotState::ReclaimedPartly(_)) => log::warn!(
                "watchdog: slot {i} VMSA {vmsa_pa:#x} retired after {} retries{how}, clean-up failed: {e:?}, pages {}",
                d.retries,
                s.pages_revoked,
            ),
            (Err(e), _) => log::warn!(
                "watchdog: slot {i} VMSA {vmsa_pa:#x} retire failed: {e:?} after {} retries{how}, giving up",
                d.retries,
            ),
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: RetryPolicy = RetryPolicy {
        retries: 2,
        window: 0,
        force: true,
    };

    #[test]
    fn retryable_errors_retry_then_force_then_give_up() {
        for e in [
            RetireError::ApDestroy(2),
            RetireError::Busy(7),
            RetireError::NoMapping,
        ] {
            // First attempt failed, then the two plain retries.
            assert_eq!(
                after_failure(e, false, 0, 0, &P),
                AfterFailure::Retry { force: false }
            );
            assert_eq!(
                after_failure(e, false, 1, 0, &P),
                AfterFailure::Retry { force: false }
            );
            // Bound reached: one forced attempt.
            assert_eq!(
                after_failure(e, false, 2, 0, &P),
                AfterFailure::Retry { force: true }
            );
            // The forced attempt failed too.
            assert_eq!(after_failure(e, true, 3, 0, &P), AfterFailure::GiveUp);
        }
    }

    #[test]
    fn no_force_gives_up_at_the_bound() {
        let p = RetryPolicy { force: false, ..P };
        let e = RetireError::ApDestroy(2);
        assert_eq!(
            after_failure(e, false, 1, 0, &p),
            AfterFailure::Retry { force: false }
        );
        assert_eq!(after_failure(e, false, 2, 0, &p), AfterFailure::GiveUp);
        let p0 = RetryPolicy { retries: 0, ..p };
        assert_eq!(after_failure(e, false, 0, 0, &p0), AfterFailure::GiveUp);
    }

    #[test]
    fn window_ends_plain_retries_early() {
        let p = RetryPolicy {
            retries: 1000,
            window: 100,
            force: true,
        };
        let e = RetireError::Busy(1);
        assert_eq!(
            after_failure(e, false, 5, 99, &p),
            AfterFailure::Retry { force: false }
        );
        assert_eq!(
            after_failure(e, false, 5, 100, &p),
            AfterFailure::Retry { force: true }
        );
    }

    #[test]
    fn errors_after_the_vmsa_rmpadjust_finish_the_slot() {
        for e in [
            RetireError::PageRevoke {
                index: 0,
                status: 6,
            },
            RetireError::VmsaRelease(2),
        ] {
            assert_eq!(after_failure(e, false, 0, 0, &P), AfterFailure::Gone);
            assert_eq!(after_failure(e, true, 3, 0, &P), AfterFailure::Gone);
        }
    }

    #[test]
    fn errors_a_retry_cannot_fix_give_up() {
        for e in [
            RetireError::InvalidTarget,
            RetireError::InvalidPage(3),
            RetireError::VmsaAdjust(2),
        ] {
            assert_eq!(after_failure(e, false, 0, 0, &P), AfterFailure::GiveUp);
        }
    }

    #[test]
    fn default_policy_retries_then_forces() {
        let p = RetryPolicy::default();
        assert_eq!(p.retries, WATCHDOG_DEFAULT_RETRIES);
        assert!(p.force);
        assert_eq!(p.window, 0);
    }

    #[test]
    fn live_states() {
        assert!(SlotState::Armed.live());
        assert!(SlotState::Reclaiming.live());
        assert!(SlotState::Retrying(RetireError::NoMapping).live());
        assert!(!SlotState::Reclaimed.live());
        assert!(!SlotState::Failed(RetireError::NoMapping).live());
        assert!(!SlotState::ReclaimedPartly(RetireError::VmsaRelease(0)).live());
        assert!(!SlotState::Free.live());
    }
}
