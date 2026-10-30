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

use crate::address::{Address, PhysAddr};
use crate::error::SvsmError;
use crate::greq::pld_key::SnpDerivedKeyRequest;
use crate::greq::services::get_derived_key;
use crate::mm::{PerCPUPageMappingGuard, valid_phys_region};
use crate::protocols::RequestParams;
use crate::protocols::core::PVALIDATE_LOCK;
use crate::protocols::errors::SvsmReqError;
use crate::sev::utils::{RMPFlags, rmp_adjust};
use crate::types::{PAGE_SIZE, PageSize};
use crate::utils::MemoryRegion;
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
        #[cfg(feature = "dice")]
        SVSM_REQ_PARTTEST_DICE_ROOT => dice_root(params),
        #[cfg(feature = "dice")]
        SVSM_REQ_PARTTEST_DICE_ATTEST => dice_attest(params),
        _ => Err(SvsmReqError::unsupported_call()),
    }
}
