// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2026 Fr0ntierX
//
// Partition-isolation test protocol (F0X-78, idea B).
//
// B isolates partitions by separate page tables inside one VMPL, and rests on
// one invariant: the SVSM owns a partition's page tables and the partition
// cannot write its own page-table pages. That is RMP write-protection enforced
// against the partition's VMPL, and it is the half of the invariant that cannot
// be tested from inside the guest.
//
// The guest side is already settled on silicon: a partition at VMPL2 cannot
// RMPADJUST VMPL2, VMPL1 or VMPL0 (all FAIL_PERMISSION), so it cannot re-grant
// itself access the SVSM revoked. What was missing is proof that the revocation
// is actually enforced when the partition then touches the page. This protocol
// supplies the VMPL0 half so a guest prober can complete the test:
//
//   1. guest allocates a page it owns and writes to it              (succeeds)
//   2. guest calls REVOKE_WRITE with that GPA
//   3. guest writes again                                           (must fault)
//   4. guest calls GRANT_WRITE to restore, so the page is reusable
//
// Read is deliberately left in place by REVOKE_WRITE rather than dropping all
// access. Page tables have to stay readable for the hardware walker at the
// partition's VMPL, so read-only is the configuration B actually runs, and
// no-access would test a stronger property than the design needs.
//
// This is a test surface that can hand a guest a way to make its own pages
// unwritable, so it is behind the `parttest` cargo feature and must never be
// enabled in a shipping build. It is not registered in the protocol dispatch
// unless that feature is on.

extern crate alloc;

use crate::address::{Address, PhysAddr};
use crate::cpu::msr::rdtsc;
use crate::error::SvsmError;
use crate::greq::pld_key::SnpDerivedKeyRequest;
use crate::greq::services::get_derived_key;
use crate::mm::{PerCPUPageMappingGuard, valid_phys_region};
use crate::protocols::RequestParams;
use crate::protocols::core::{
    PVALIDATE_LOCK, RETIRE_DEFAULT_INUSE_DEADLINE, RetireError, RetireOptions, core_create_vcpu,
    retire_guest_vcpu,
};
use crate::protocols::errors::SvsmReqError;
use crate::protocols::faultinj::{FAULT_BAD_VMPL, Fault, FaultStep, fault_arm};
use crate::protocols::watchdog::{
    RetryPolicy, SlotState, WATCHDOG_MAX_PAGES, WatchdogError, watchdog_arm, watchdog_disarm,
    watchdog_status, watchdog_tick,
};
use crate::sev::ghcb::{GhcbError, with_current_ghcb};
use crate::sev::utils::{RMPFlags, rmp_adjust};
use crate::types::{PAGE_SIZE, PageSize};
use crate::utils::MemoryRegion;
use alloc::vec::Vec;
use sha2::{Digest, Sha384};

/// Drop write for the guest VMPL on one 4K page, leaving read in place.
const SVSM_REQ_PARTTEST_REVOKE_WRITE: u32 = 0;
/// Restore read/write/execute for the guest VMPL on one 4K page.
const SVSM_REQ_PARTTEST_GRANT_WRITE: u32 = 1;
/// Do nothing and return success. Times the guest -> VMGEXIT -> hypervisor ->
/// VMPL0 -> back round trip with no work inside it, which is the relay cost any
/// hypervisor-mediated partition switch has to pay (F0X-93 vmsa_reentry). VEIL
/// reports 7,135 cycles for a relayed domain switch and NestedSGX 32-34k, so
/// this is the term to compare against those.
const SVSM_REQ_PARTTEST_NOOP: u32 = 2;
/// Drop every permission for the guest VMPL on one 4K page, read included.
/// REVOKE_WRITE keeps read because B runs read-only: page tables must stay
/// readable for the hardware walker at the partition's VMPL. This exists only
/// to tell whether the wedged-vCPU behaviour is specific to read-only or is
/// general to any RMP permission violation (F0X-301, experiment 2).
const SVSM_REQ_PARTTEST_REVOKE_ALL: u32 = 3;
/// Ask the PSP for a VMPL0 derived key and return the first 24 bytes of its
/// SHA-384 in rcx, rdx and r8. Input: rcx = GUEST_FIELD_SELECT, rdx =
/// ROOT_KEY_SELECT. This exists to check on silicon that the key the DICE
/// root will use (F0X-80) is stable for one image and changes with the
/// selected fields. The key itself is the monitor's root secret and never
/// leaves VMPL0; only a one-way hash does, and only in parttest builds.
const SVSM_REQ_PARTTEST_DERIVED_KEY_HASH: u32 = 4;
/// Write the monitor's hardware root endorsement (F0X-80) to the 4K guest
/// page at rcx: the alias public key (97-byte SEC1 point) followed by the
/// VMPL0 attestation report that binds it in REPORT_DATA. A verifier checks
/// the report against the VCEK chain and recomputes REPORT_DATA from the key.
/// Only public values leave VMPL0; the CDI and the alias private key do not.
#[cfg(feature = "dice")]
const SVSM_REQ_PARTTEST_DICE_ROOT: u32 = 5;
/// Measure a guest memory region as a partition and return its EAT
/// (F0X-80): the monitor token, signed by the hardware-rooted monitor alias,
/// carrying the partition's nested token and the monitor's endorsement of the
/// partition's measurement and alias key. With call 5's root it is a complete
/// version 2 evidence bundle.
///
/// rcx is a 4K request page: nonce (32 bytes), REPORT_DATA (64), region GPA
/// (u64), region page count (u32, at most `DICE_ATTEST_MAX_PAGES`), partition
/// ID length (u32, at most `DICE_ATTEST_MAX_ID`), then the ID in UTF-8. rdx
/// is a 4K output page that gets the token length (u32) followed by the token.
#[cfg(feature = "dice")]
const SVSM_REQ_PARTTEST_DICE_ATTEST: u32 = 6;
/// Retire a guest vCPU and revoke a partition's pages (F0X-75 watchdog
/// reclaim), through `core::retire_guest_vcpu`. The guest must take the
/// target CPU offline first.
///
/// In: rcx = the target's VMSA GPA. rdx = GPA of a 4K page list, or 0 for
/// none: a u64 count (at most `RETIRE_MAX_PAGES`) followed by that many 4K
/// GPAs. r8 bits 0-62 = FAIL_INUSE deadline in TSC ticks (0 for the
/// default), bit 63 = retire even if the host refuses AP_DESTROY.
///
/// Out, on success and on every error: rcx = AP_DESTROY ticks, rdx = ticks
/// from the first RMPADJUST on the VMSA to success, r8 = pages revoked
/// (bits 0-15) | FAIL_INUSE retries (16-47) | AP_DESTROY refused (63).
///
/// Errors: INVALID_PARAMETER (not a guest-created in-use VMSA, the caller's
/// own vCPU, or a bad list), INVALID_ADDRESS (a list entry is not a guest
/// page or is a VMSA or calling area, entry index in rdx), BUSY (FAIL_INUSE
/// past the deadline), and the protocol codes below, whose detail goes in
/// rdx. See `RetireError` for what each leaves behind.
const SVSM_REQ_PARTTEST_RETIRE_VCPU: u32 = 7;
const RETIRE_MAX_PAGES: usize = PAGE_SIZE / 8 - 1;
/// Host refused AP_DESTROY, nothing revoked. rdx = SW_EXITINFO1.
const RETIRE_ERR_AP_DESTROY: u64 = 0x100;
/// RMPADJUST on the VMSA failed, nothing revoked. rdx = status.
const RETIRE_ERR_VMSA_ADJUST: u64 = 0x101;
/// Retired, but revoking a partition page failed. rdx = index | status << 32.
const RETIRE_ERR_PAGE_REVOKE: u64 = 0x102;
/// Retired and pages revoked, but the scrubbed VMSA page stays revoked from
/// the guest. rdx = status.
const RETIRE_ERR_VMSA_RELEASE: u64 = 0x103;
/// The VMSA page could not be mapped. The host was already asked to halt it.
const RETIRE_ERR_NO_MAPPING: u64 = 0x104;
/// Start a guest-built VMSA on another vCPU (F0X-75 watchdog tests). In: rcx
/// = VMSA GPA, rdx = calling area GPA, r8 = APIC ID, as CREATE_VCPU. Runs
/// CREATE_VCPU's checks and registration, then sends GHCB AP_CREATE at the
/// guest VMPL from VMPL0, so the guest kernel needs no GHCB access of its own
/// to hand a vCPU it took offline to a partition payload. If AP_CREATE fails
/// (`START_ERR_AP_CREATE`) the VMSA stays registered but never ran, and call
/// 7 retires it.
const SVSM_REQ_PARTTEST_START_VCPU: u32 = 8;
/// Arm a watchdog slot (`watchdog::watchdog_arm`). rcx = 4K request page:
/// VMSA GPA, heartbeat GPA (8-byte aligned), deadline in TSC ticks, page
/// count (at most `WATCHDOG_MAX_PAGES`), then that many 4K GPAs to revoke on
/// reclaim, all u64. rdx = retry policy (`watchdog::RetryPolicy`): 0 for the
/// default, or `ARM_POLICY_SET` with bits 0-15 = plain retries and
/// `ARM_POLICY_NO_FORCE` to never force. r8 = with `ARM_POLICY_SET`, the
/// retry window in TSC ticks (0 for none). Out: rcx = slot index.
const SVSM_REQ_PARTTEST_WATCHDOG_ARM: u32 = 9;
/// One watchdog period (`watchdog::watchdog_tick`). Only the first vCPU that
/// ticks may tick again. Out: rcx = armed slots checked, rdx = slots whose
/// vCPU was retired in this tick (bitmask), r8 = VMPL0 ticks spent (bits
/// 0-31) | slots that finished with an error (bitmask) << 32 | slots whose
/// retire failed and is retried next tick (bitmask) << 48. A slot retired
/// with a clean-up error (`ReclaimedPartly`) is in both of the first two.
const SVSM_REQ_PARTTEST_WATCHDOG_TICK: u32 = 10;
/// Write one slot's record to a 4K guest page. In: rcx = slot, rdx = page.
/// The record is `WATCHDOG_STATUS_WORDS` u64s, laid out in `watchdog_status`.
const SVSM_REQ_PARTTEST_WATCHDOG_STATUS: u32 = 11;
/// Free a slot that is armed, reclaimed or failed. In: rcx = slot.
const SVSM_REQ_PARTTEST_WATCHDOG_DISARM: u32 = 12;
/// Arm one injected failure at a named step of the retire, START_VCPU or
/// watchdog paths (`faultinj`), so their error paths can be driven on
/// hardware. In: rcx bits 0-11 = `FaultStep` code, or 0 to clear, bits 12-63
/// = a page-aligned VMSA GPA the fault is keyed on (0 for any target; only
/// the retire steps name a target). rdx = the status the
/// failing step returns, where it returns one (for FAIL_INUSE, how many fake
/// results the retire sees, u64::MAX for all of them). r8 bits 0-31 = hits at the
/// step to let through first, bits 32-63 = how many hits fail (0 means 1).
/// Out: rcx = the step of the fault this replaces (0 if none), rdx = how
/// many times it fired, r8 = failures it had left.
const SVSM_REQ_PARTTEST_FAULT_INJECT: u32 = 13;
/// AP_CREATE refused by the host. rdx = SW_EXITINFO1, or u64::MAX if the GHCB
/// answer was malformed.
const START_ERR_AP_CREATE: u64 = 0x110;
/// The tick came from a vCPU other than the watchdog vCPU.
const WATCHDOG_ERR_NOT_WATCHDOG_CPU: u64 = 0x120;
/// No free slot.
const WATCHDOG_ERR_FULL: u64 = 0x121;
/// The VMSA is already watched.
const WATCHDOG_ERR_ALREADY_WATCHED: u64 = 0x122;
const WATCHDOG_STATUS_WORDS: usize = 26;
/// ARM rdx: a retry policy is given (otherwise the default).
const ARM_POLICY_SET: u64 = 1 << 63;
/// ARM rdx: never force, end `Failed` when the plain retries are used up.
const ARM_POLICY_NO_FORCE: u64 = 1 << 62;
#[cfg(feature = "dice")]
const DICE_ATTEST_MAX_PAGES: usize = 256;
#[cfg(feature = "dice")]
const DICE_ATTEST_MAX_ID: usize = 64;

#[cfg(feature = "dice")]
fn dice_root(params: &RequestParams) -> Result<(), SvsmReqError> {
    use crate::dice::{ALIAS_PUBLIC_SIZE, root_endorsement};
    use crate::greq::pld_report::AttestationReport;

    let paddr = PhysAddr::from(params.rcx);
    if !paddr.is_page_aligned() {
        return Err(SvsmReqError::invalid_parameter());
    }
    let (public, report) = root_endorsement().map_err(|e| {
        log::info!("parttest: DICE root endorsement failed: {e:?}");
        SvsmReqError::invalid_request()
    })?;
    // guest_ptr checks the region is guest memory before anything is written.
    let guard = PerCPUPageMappingGuard::create_4k(paddr)?;
    guard
        .guest_ptr::<[u8; ALIAS_PUBLIC_SIZE]>(0)?
        .write(public)?;
    guard
        .guest_ptr::<AttestationReport>(ALIAS_PUBLIC_SIZE)?
        .write(report)?;
    Ok(())
}

#[cfg(feature = "dice")]
fn dice_attest(params: &RequestParams) -> Result<(), SvsmReqError> {
    use crate::dice::attest_partition;

    let req_pa = PhysAddr::from(params.rcx);
    let out_pa = PhysAddr::from(params.rdx);
    if !req_pa.is_page_aligned() || !out_pa.is_page_aligned() {
        return Err(SvsmReqError::invalid_parameter());
    }

    let (nonce, report_data, region_pa, pages, id) = {
        let guard = PerCPUPageMappingGuard::create_4k(req_pa)?;
        let nonce = guard.guest_ptr::<[u8; 32]>(0)?.read()?;
        let report_data = guard.guest_ptr::<[u8; 64]>(32)?.read()?;
        let region_pa = guard.guest_ptr::<u64>(96)?.read()?;
        let pages = guard.guest_ptr::<u32>(104)?.read()? as usize;
        let id_len = guard.guest_ptr::<u32>(108)?.read()? as usize;
        if id_len == 0 || id_len > DICE_ATTEST_MAX_ID {
            return Err(SvsmReqError::invalid_parameter());
        }
        let id = guard.guest_slice::<u8>(112, id_len)?.read_to_vec()?;
        (nonce, report_data, PhysAddr::from(region_pa), pages, id)
    };
    let id = core::str::from_utf8(&id).map_err(|_| SvsmReqError::invalid_parameter())?;
    if !region_pa.is_page_aligned() || pages == 0 || pages > DICE_ATTEST_MAX_PAGES {
        return Err(SvsmReqError::invalid_parameter());
    }

    // The partition measurement is SHA-384 over the region, page by page.
    let mut hash = Sha384::new();
    for i in 0..pages {
        let pa = region_pa
            .checked_add(i * PAGE_SIZE)
            .ok_or_else(SvsmReqError::invalid_address)?;
        let guard = PerCPUPageMappingGuard::create_4k(pa)?;
        hash.update(guard.guest_slice::<u8>(0, PAGE_SIZE)?.read_to_vec()?);
    }
    let measurement: [u8; 48] = hash.finalize().into();

    let token = attest_partition(&nonce, &report_data, id, &measurement).map_err(|e| {
        log::info!("parttest: DICE attest failed: {e:?}");
        SvsmReqError::invalid_request()
    })?;
    let len = u32::try_from(token.len()).map_err(|_| SvsmReqError::invalid_request())?;
    if token.len() > PAGE_SIZE - 4 {
        return Err(SvsmReqError::invalid_request());
    }
    let guard = PerCPUPageMappingGuard::create_4k(out_pa)?;
    guard.guest_ptr::<u32>(0)?.write(len)?;
    guard
        .guest_slice::<u8>(4, token.len())?
        .write_from_slice(&token)?;
    Ok(())
}

pub const PARTTEST_PROTOCOL_VERSION_MIN: u32 = 1;
pub const PARTTEST_PROTOCOL_VERSION_MAX: u32 = 1;

/// Maps the guest page named by rcx and applies `flags` to it for the guest
/// VMPL. The address must be 4K aligned and inside a valid guest physical
/// region, which is the same check the core PVALIDATE handler applies, so a
/// guest cannot point this at SVSM-private or hypervisor memory.
fn adjust_guest_page(params: &RequestParams, flags: RMPFlags) -> Result<(), SvsmReqError> {
    let paddr = PhysAddr::from(params.rcx);

    if !paddr.is_page_aligned() {
        return Err(SvsmReqError::invalid_parameter());
    }

    let region =
        MemoryRegion::checked_new(paddr, PAGE_SIZE).ok_or_else(SvsmReqError::invalid_address)?;
    if !valid_phys_region(&region) {
        return Err(SvsmReqError::invalid_address());
    }

    // Serialize against PVALIDATE and CREATE_VCPU, which change RMP state on
    // guest pages under this lock. Without it a concurrent CREATE_VCPU on the
    // same page could see its permissions change between its own steps.
    let _lock = PVALIDATE_LOCK.lock_write();

    let guard = PerCPUPageMappingGuard::create_4k(paddr)?;
    let vaddr = guard.virt_addr();

    // SAFETY: the page is a guest page, mapped for the duration of the guard,
    // and the flags only ever narrow or restore the guest VMPL's own access to
    // it. Nothing here touches SVSM-private mappings, so it cannot affect
    // memory safety on this side.
    // The RMPADJUST status code is kept in the log line rather than folded
    // into the protocol error, which has no field for it. A failed lab run
    // otherwise says only "invalid request" when the interesting part is
    // whether the instruction returned FAIL_PERMISSION, FAIL_SIZEMISMATCH or
    // FAIL_INUSE.
    unsafe { rmp_adjust(vaddr, flags, PageSize::Regular) }.map_err(|e| match e {
        SvsmError::SevSnp(code) => {
            log::info!("parttest: RMPADJUST failed on {paddr:#x}, status {code:?}");
            SvsmReqError::invalid_request()
        }
        other => other.into(),
    })
}

fn derived_key_hash(params: &mut RequestParams) -> Result<(), SvsmReqError> {
    let root_key_select =
        u32::try_from(params.rdx).map_err(|_| SvsmReqError::invalid_parameter())?;
    let request = SnpDerivedKeyRequest::new(root_key_select, params.rcx, 0, 0)
        .map_err(|_| SvsmReqError::invalid_parameter())?;
    let key = get_derived_key(&request).map_err(|e| {
        log::info!("parttest: MSG_KEY_REQ failed: {e:?}");
        SvsmReqError::invalid_request()
    })?;
    let digest = Sha384::digest(&key[..]);
    let word = |i: usize| {
        u64::from_le_bytes(
            digest[i * 8..i * 8 + 8]
                .try_into()
                .expect("an 8-byte slice of a 48-byte digest"),
        )
    };
    params.rcx = word(0);
    params.rdx = word(1);
    params.r8 = word(2);
    Ok(())
}

fn retire_vcpu(params: &mut RequestParams) -> Result<(), SvsmReqError> {
    let vmsa_pa = PhysAddr::from(params.rcx);
    let list_pa = PhysAddr::from(params.rdx);
    let deadline = params.r8 & !(1u64 << 63);
    let opts = RetireOptions {
        inuse_deadline: if deadline == 0 {
            RETIRE_DEFAULT_INUSE_DEADLINE
        } else {
            deadline
        },
        force: params.r8 & (1u64 << 63) != 0,
        log: true,
    };
    params.rcx = 0;
    params.rdx = 0;
    params.r8 = 0;

    let mut pages: Vec<PhysAddr> = Vec::new();
    if !list_pa.is_null() {
        if !list_pa.is_page_aligned() {
            return Err(SvsmReqError::invalid_parameter());
        }
        let guard = PerCPUPageMappingGuard::create_4k(list_pa)?;
        let count = guard.guest_ptr::<u64>(0)?.read()?;
        let count = usize::try_from(count).map_err(|_| SvsmReqError::invalid_parameter())?;
        if count > RETIRE_MAX_PAGES {
            return Err(SvsmReqError::invalid_parameter());
        }
        let entries = guard.guest_slice::<u64>(8, count)?.read_to_vec()?;
        pages
            .try_reserve_exact(count)
            .map_err(|_| SvsmReqError::invalid_request())?;
        pages.extend(entries.into_iter().map(PhysAddr::from));
    }

    let (res, stats) = match retire_guest_vcpu(vmsa_pa, &pages, &opts) {
        Ok(stats) => (Ok(()), stats),
        Err((e, stats)) => (Err(e), stats),
    };
    params.rcx = stats.ap_destroy_ticks;
    params.rdx = stats.vmsa_ticks;
    params.r8 = (stats.pages_revoked as u64 & 0xffff)
        | (stats.inuse_retries & 0xffff_ffff) << 16
        | u64::from(stats.ap_destroy_refused) << 63;

    res.map_err(|e| {
        log::info!("parttest: retire of VMSA {vmsa_pa:#x} failed: {e:?}");
        match e {
            RetireError::InvalidTarget => SvsmReqError::invalid_parameter(),
            RetireError::InvalidPage(i) => {
                params.rdx = i as u64;
                SvsmReqError::invalid_address()
            }
            RetireError::NoMapping => SvsmReqError::protocol(RETIRE_ERR_NO_MAPPING),
            RetireError::ApDestroy(code) => {
                params.rdx = code;
                SvsmReqError::protocol(RETIRE_ERR_AP_DESTROY)
            }
            RetireError::Busy(_) => SvsmReqError::busy(),
            RetireError::VmsaAdjust(status) => {
                params.rdx = status;
                SvsmReqError::protocol(RETIRE_ERR_VMSA_ADJUST)
            }
            RetireError::PageRevoke { index, status } => {
                params.rdx = index as u64 | status << 32;
                SvsmReqError::protocol(RETIRE_ERR_PAGE_REVOKE)
            }
            RetireError::VmsaRelease(status) => {
                params.rdx = status;
                SvsmReqError::protocol(RETIRE_ERR_VMSA_RELEASE)
            }
        }
    })
}

fn start_vcpu(params: &mut RequestParams) -> Result<(), SvsmReqError> {
    let vmsa_pa = PhysAddr::from(params.rcx);
    let apic_id = params.r8 & 0xffff_ffff;
    core_create_vcpu(params)?;
    let vmpl = fault_inject!(ApCreate, _ => FAULT_BAD_VMPL, RMPFlags::GUEST_VMPL.bits());
    let sev_features = params.sev_features;
    with_current_ghcb(|ghcb| ghcb.ap_create(vmsa_pa, apic_id, vmpl, sev_features)).map_err(|e| {
        log::info!("parttest: AP_CREATE for APIC {apic_id} VMSA {vmsa_pa:#x} failed: {e:?}");
        // The documented detail, as call 7 reports a refused AP_DESTROY.
        params.rdx = match e {
            SvsmError::Ghcb(GhcbError::VmgexitError(info1, _)) => info1,
            _ => u64::MAX,
        };
        SvsmReqError::protocol(START_ERR_AP_CREATE)
    })
}

fn fault_inject_call(params: &mut RequestParams) -> Result<(), SvsmReqError> {
    let fault = match params.rcx & 0xfff {
        0 => None,
        code => {
            let step = FaultStep::from_code(code).ok_or_else(SvsmReqError::invalid_parameter)?;
            let count = (params.r8 >> 32) as u32;
            Some(Fault {
                step,
                value: params.rdx,
                skip: params.r8 as u32,
                count: count.max(1),
                key: params.rcx & !0xfff,
            })
        }
    };
    let prev = fault_arm(fault);
    if let Some(f) = fault {
        log::info!(
            "parttest: fault armed at {:?}, value {:#x}, skip {}, count {}, key {:#x}",
            f.step,
            f.value,
            f.skip,
            f.count,
            f.key
        );
    }
    params.rcx = prev.step;
    params.rdx = prev.fired;
    params.r8 = u64::from(prev.remaining);
    Ok(())
}

fn watchdog_err(e: WatchdogError, params: &mut RequestParams) -> SvsmReqError {
    match e {
        WatchdogError::BadPage(i) => {
            params.rdx = i as u64;
            SvsmReqError::invalid_address()
        }
        WatchdogError::BadHeartbeat => SvsmReqError::invalid_address(),
        WatchdogError::Read(err) => {
            log::info!("parttest: watchdog heartbeat read failed: {err:?}");
            SvsmReqError::invalid_address()
        }
        WatchdogError::BadTarget | WatchdogError::TooManyPages | WatchdogError::BadSlot => {
            SvsmReqError::invalid_parameter()
        }
        WatchdogError::AlreadyWatched => SvsmReqError::protocol(WATCHDOG_ERR_ALREADY_WATCHED),
        WatchdogError::Full => SvsmReqError::protocol(WATCHDOG_ERR_FULL),
        WatchdogError::NotWatchdogCpu => SvsmReqError::protocol(WATCHDOG_ERR_NOT_WATCHDOG_CPU),
    }
}

fn watchdog_arm_call(params: &mut RequestParams) -> Result<(), SvsmReqError> {
    let req_pa = PhysAddr::from(params.rcx);
    if !req_pa.is_page_aligned() {
        return Err(SvsmReqError::invalid_parameter());
    }
    let (vmsa_pa, hb_pa, deadline, pages) = {
        let guard = PerCPUPageMappingGuard::create_4k(req_pa)?;
        let head = guard.guest_slice::<u64>(0, 4)?.read_to_vec()?;
        let count = usize::try_from(head[3]).map_err(|_| SvsmReqError::invalid_parameter())?;
        if count > WATCHDOG_MAX_PAGES {
            return Err(SvsmReqError::invalid_parameter());
        }
        let pages = guard.guest_slice::<u64>(32, count)?.read_to_vec()?;
        (head[0], head[1], head[2], pages)
    };
    let pages: Vec<PhysAddr> = pages.into_iter().map(PhysAddr::from).collect();
    let policy = if params.rdx & ARM_POLICY_SET != 0 {
        RetryPolicy {
            retries: params.rdx as u16,
            window: params.r8,
            force: params.rdx & ARM_POLICY_NO_FORCE == 0,
        }
    } else {
        RetryPolicy::default()
    };
    params.rdx = 0;
    let slot = watchdog_arm(
        PhysAddr::from(vmsa_pa),
        PhysAddr::from(hb_pa),
        deadline,
        pages,
        policy,
    )
    .map_err(|e| watchdog_err(e, params))?;
    params.rcx = slot as u64;
    Ok(())
}

fn watchdog_tick_call(params: &mut RequestParams) -> Result<(), SvsmReqError> {
    let report = watchdog_tick().map_err(|e| watchdog_err(e, params))?;
    params.rcx = u64::from(report.checked);
    params.rdx = report.reclaimed;
    params.r8 = (report.ticks & 0xffff_ffff) | report.failed << 32 | report.retrying << 48;
    Ok(())
}

/// A retire error as status words carry it: code << 8 | detail << 32.
fn retire_error_code(e: RetireError) -> u64 {
    let (code, detail): (u64, u64) = match e {
        RetireError::InvalidTarget => (1, 0),
        RetireError::InvalidPage(i) => (2, i as u64),
        RetireError::NoMapping => (3, 0),
        RetireError::ApDestroy(c) => (4, c),
        RetireError::Busy(r) => (5, r),
        RetireError::VmsaAdjust(s) => (6, s),
        RetireError::PageRevoke { index, status } => (7, index as u64 | status << 16),
        RetireError::VmsaRelease(s) => (8, s),
    };
    code << 8 | (detail & 0xffff_ffff) << 32
}

/// Bits 0-7 the state, and for the states that carry an error, that error as
/// `retire_error_code` lays it out. Retrying sets bit 16 when the next
/// attempt is forced.
fn slot_state_code(state: SlotState, force_next: bool) -> u64 {
    match state {
        SlotState::Free => 0,
        SlotState::Armed => 1,
        SlotState::Reclaiming => 2,
        SlotState::Reclaimed => 3,
        SlotState::Failed(e) => 4 | retire_error_code(e),
        SlotState::Retrying(e) => 5 | retire_error_code(e) | u64::from(force_next) << 16,
        SlotState::ReclaimedPartly(e) => 6 | retire_error_code(e),
    }
}

fn watchdog_status_call(params: &mut RequestParams) -> Result<(), SvsmReqError> {
    let index = usize::try_from(params.rcx).map_err(|_| SvsmReqError::invalid_parameter())?;
    let out_pa = PhysAddr::from(params.rdx);
    if !out_pa.is_page_aligned() {
        return Err(SvsmReqError::invalid_parameter());
    }
    let s = watchdog_status(index).map_err(|e| watchdog_err(e, params))?;
    let words: [u64; WATCHDOG_STATUS_WORDS] = [
        slot_state_code(s.state, s.force_next),
        u64::from(s.vmsa_pa),
        s.cpu_index as u64,
        s.deadline,
        s.last_value,
        s.last_change,
        s.armed_at,
        s.changes_seen,
        s.max_gap,
        s.read_errors,
        s.reclaim_decided,
        s.stale_at_reclaim,
        s.retire.ap_destroy_ticks,
        s.retire.lock_ticks,
        s.retire.vmsa_ticks,
        s.retire.total_ticks,
        s.retire.pages_revoked as u64 | s.retire.inuse_retries << 16,
        s.watchdog_ticks,
        s.last_tick,
        rdtsc(),
        s.retire_started,
        s.max_still,
        // 22: retries | an attempt used force << 32 | the host refused
        // AP_DESTROY on the last attempt << 33.
        u64::from(s.retries)
            | u64::from(s.forced) << 32
            | u64::from(s.retire.ap_destroy_refused) << 33,
        // 23: the policy as ARM takes it.
        ARM_POLICY_SET
            | u64::from(s.policy.retries)
            | if s.policy.force {
                0
            } else {
                ARM_POLICY_NO_FORCE
            },
        // 24: retry window, TSC ticks.
        s.policy.window,
        // 25: first failed attempt's error, 0 if none.
        s.first_error.map_or(0, retire_error_code),
    ];
    let guard = PerCPUPageMappingGuard::create_4k(out_pa)?;
    guard
        .guest_slice::<u64>(0, WATCHDOG_STATUS_WORDS)?
        .write_from_slice(&words)?;
    Ok(())
}

pub fn parttest_protocol_request(
    request: u32,
    params: &mut RequestParams,
) -> Result<(), SvsmReqError> {
    match request {
        // These touch GUEST_VMPL alone, where upstream's
        // rmp_revoke_guest_access sweeps GUEST_VMPL through VMPL3. Deliberate:
        // the test is about what the partition's own VMPL can reach, and
        // narrowing one level is what idea B does on a switch. Sweeping the
        // rest would also revoke levels the test never granted, which would
        // make a failure harder to attribute rather than easier.
        SVSM_REQ_PARTTEST_REVOKE_WRITE => {
            adjust_guest_page(params, RMPFlags::GUEST_VMPL | RMPFlags::READ)
        }
        SVSM_REQ_PARTTEST_GRANT_WRITE => {
            adjust_guest_page(params, RMPFlags::GUEST_VMPL | RMPFlags::RWX)
        }
        SVSM_REQ_PARTTEST_REVOKE_ALL => {
            adjust_guest_page(params, RMPFlags::GUEST_VMPL | RMPFlags::NONE)
        }
        SVSM_REQ_PARTTEST_NOOP => Ok(()),
        SVSM_REQ_PARTTEST_DERIVED_KEY_HASH => derived_key_hash(params),
        SVSM_REQ_PARTTEST_RETIRE_VCPU => retire_vcpu(params),
        SVSM_REQ_PARTTEST_START_VCPU => start_vcpu(params),
        SVSM_REQ_PARTTEST_WATCHDOG_ARM => watchdog_arm_call(params),
        SVSM_REQ_PARTTEST_WATCHDOG_TICK => watchdog_tick_call(params),
        SVSM_REQ_PARTTEST_WATCHDOG_STATUS => watchdog_status_call(params),
        SVSM_REQ_PARTTEST_WATCHDOG_DISARM => {
            let index =
                usize::try_from(params.rcx).map_err(|_| SvsmReqError::invalid_parameter())?;
            watchdog_disarm(index).map_err(|e| watchdog_err(e, params))
        }
        SVSM_REQ_PARTTEST_FAULT_INJECT => fault_inject_call(params),
        #[cfg(feature = "dice")]
        SVSM_REQ_PARTTEST_DICE_ROOT => dice_root(params),
        #[cfg(feature = "dice")]
        SVSM_REQ_PARTTEST_DICE_ATTEST => dice_attest(params),
        _ => Err(SvsmReqError::unsupported_call()),
    }
}
