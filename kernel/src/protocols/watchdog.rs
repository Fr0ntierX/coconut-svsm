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
    RetireError, RetireOptions, RetireStats, first_invalid_page, retire_guest_vcpu,
};
use crate::types::PAGE_SIZE;
use crate::utils::MemoryRegion;
use alloc::vec::Vec;

/// Slots in the watch table. The tick reports reclaims as a bitmask, so this
/// must stay at or below 64.
pub(crate) const WATCHDOG_SLOTS: usize = 16;
/// Partition pages one slot can carry for the revoke on reclaim.
pub(crate) const WATCHDOG_MAX_PAGES: usize = 64;

const _: () = assert!(WATCHDOG_SLOTS <= 64);

/// Where a slot is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlotState {
    Free,
    Armed,
    /// The deadline passed and a retire is running outside the table lock.
    Reclaiming,
    Reclaimed,
    /// The deadline passed but the retire failed. See `RetireError` for what
    /// it left behind.
    Failed(RetireError),
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
    /// The VMSA is already watched by an armed slot.
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
    /// TSC at which this slot's retire started. Later than
    /// `reclaim_decided` when an earlier slot was retired in the same tick.
    pub retire_started: u64,
    pub retire: RetireStats,
    pub watchdog_ticks: u64,
    pub last_tick: u64,
}

/// What one tick did.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TickReport {
    /// Armed slots whose heartbeat was checked.
    pub checked: u32,
    /// Bit n set: slot n passed its deadline in this tick and was retired.
    pub reclaimed: u64,
    /// Bit n set: slot n passed its deadline but the retire failed.
    pub failed: u64,
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
/// VMPL when it is retired. Returns the slot index.
pub(crate) fn watchdog_arm(
    vmsa_pa: PhysAddr,
    heartbeat_pa: PhysAddr,
    deadline: u64,
    pages: Vec<PhysAddr>,
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
        .any(|s| s.state == SlotState::Armed && s.vmsa_pa == vmsa_pa)
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
        ..Slot::free()
    };
    Ok(index)
}

/// Stop watching a slot and free it. An armed slot is simply dropped, and a
/// finished one (reclaimed or failed) is cleared. A slot whose retire is
/// running is refused.
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
        watchdog_ticks: wd.ticks,
        last_tick: wd.last_tick,
    })
}

/// One watchdog period, called from the watchdog vCPU. Checks every armed
/// slot and retires the vCPUs past their deadline. Retires run outside the
/// table lock, so arm, disarm and status calls on other vCPUs never wait on
/// the host's part of a retire.
pub(crate) fn watchdog_tick() -> Result<TickReport, WatchdogError> {
    let t0 = rdtsc();
    let me = this_cpu().get_cpu_index();
    let mut report = TickReport::default();
    let mut expired: Vec<(usize, PhysAddr, Vec<PhysAddr>)> = Vec::new();

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
            if slot.state != SlotState::Armed {
                continue;
            }
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
            } else {
                slot.state = SlotState::Reclaiming;
                slot.stale_at_reclaim = stale;
                slot.reclaim_decided = now;
                // The retire takes the page list. Status reports how many
                // pages it revoked.
                let pages = core::mem::take(&mut slot.pages);
                if expired.try_reserve(1).is_err() {
                    // Keep it armed and try again next tick.
                    slot.pages = pages;
                    slot.state = SlotState::Armed;
                    continue;
                }
                expired.push((i, slot.vmsa_pa, pages));
            }
        }
    }

    // Retire every expired vCPU before logging anything. A log line goes out
    // over the serial console and would hold up the next retire by
    // milliseconds.
    let opts = RetireOptions {
        log: false,
        ..RetireOptions::default()
    };
    let mut done: Vec<(usize, PhysAddr, Result<(), RetireError>, RetireStats)> = Vec::new();
    for (i, vmsa_pa, pages) in expired {
        let started = rdtsc();
        let res = retire_guest_vcpu(vmsa_pa, &pages, &opts);
        let mut wd = WATCHDOG.lock();
        let slot = &mut wd.slots[i];
        slot.retire_started = started;
        let (res, stats) = match res {
            Ok(stats) => {
                slot.state = SlotState::Reclaimed;
                report.reclaimed |= 1 << i;
                (Ok(()), stats)
            }
            Err((e, stats)) => {
                slot.state = SlotState::Failed(e);
                report.failed |= 1 << i;
                (Err(e), stats)
            }
        };
        slot.retire = stats;
        if done.try_reserve(1).is_ok() {
            done.push((i, vmsa_pa, res, stats));
        }
    }
    report.ticks = rdtsc() - t0;

    for (i, vmsa_pa, res, s) in done {
        match res {
            Ok(()) => log::info!(
                "watchdog: slot {i} VMSA {vmsa_pa:#x} retired: ap_destroy {} lock {} vmsa {} total {} ticks, inuse {}, pages {}{}",
                s.ap_destroy_ticks,
                s.lock_ticks,
                s.vmsa_ticks,
                s.total_ticks,
                s.inuse_retries,
                s.pages_revoked,
                if s.ap_destroy_refused {
                    ", AP_DESTROY refused"
                } else {
                    ""
                }
            ),
            Err(e) => log::warn!("watchdog: slot {i} VMSA {vmsa_pa:#x} retire failed: {e:?}"),
        }
    }
    Ok(report)
}
