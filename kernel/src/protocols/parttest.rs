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
use crate::mm::{PerCPUPageMappingGuard, valid_phys_region};
use crate::protocols::RequestParams;
use crate::protocols::errors::SvsmReqError;
use crate::sev::utils::{RMPFlags, rmp_adjust};
use crate::types::{PAGE_SIZE, PageSize};
use crate::utils::MemoryRegion;

/// Drop write for the guest VMPL on one 4K page, leaving read in place.
const SVSM_REQ_PARTTEST_REVOKE_WRITE: u32 = 0;
/// Restore read/write/execute for the guest VMPL on one 4K page.
const SVSM_REQ_PARTTEST_GRANT_WRITE: u32 = 1;

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

    let guard = PerCPUPageMappingGuard::create_4k(paddr)?;
    let vaddr = guard.virt_addr();

    // SAFETY: the page is a guest page, mapped for the duration of the guard,
    // and the flags only ever narrow or restore the guest VMPL's own access to
    // it. Nothing here touches SVSM-private mappings, so it cannot affect
    // memory safety on this side.
    unsafe { rmp_adjust(vaddr, flags, PageSize::Regular) }.map_err(|e| match e {
        SvsmError::SevSnp(_) => SvsmReqError::invalid_request(),
        other => other.into(),
    })
}

pub fn parttest_protocol_request(
    request: u32,
    params: &mut RequestParams,
) -> Result<(), SvsmReqError> {
    match request {
        SVSM_REQ_PARTTEST_REVOKE_WRITE => {
            adjust_guest_page(params, RMPFlags::GUEST_VMPL | RMPFlags::READ)
        }
        SVSM_REQ_PARTTEST_GRANT_WRITE => {
            adjust_guest_page(params, RMPFlags::GUEST_VMPL | RMPFlags::RWX)
        }
        _ => Err(SvsmReqError::unsupported_call()),
    }
}
