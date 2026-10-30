// SPDX-License-Identifier: MIT OR Apache-2.0

//! `SNP_GUEST_REQUEST` command to request a key derived by the PSP.
//!
//! The key is derived from a chip-unique root (VCEK or VMRK) mixed with the
//! VMPL and whichever guest fields the request selects, so a different
//! measurement, policy or TCB yields a different key. Layouts match
//! `struct snp_derived_key_req` / `snp_derived_key_resp` in Linux
//! `include/uapi/linux/sev-guest.h` and the SEV-SNP firmware ABI.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::error::SvsmError;

/// `ROOT_KEY_SELECT` value: derive from the VCEK, which is chip-unique.
pub const ROOT_KEY_VCEK: u32 = 0;
/// `ROOT_KEY_SELECT` value: derive from the VM root key, which is per launch.
pub const ROOT_KEY_VMRK: u32 = 1;

/// `GUEST_FIELD_SELECT` bits: which guest fields are mixed into the key.
pub const FIELD_GUEST_POLICY: u64 = 1 << 0;
pub const FIELD_IMAGE_ID: u64 = 1 << 1;
pub const FIELD_FAMILY_ID: u64 = 1 << 2;
pub const FIELD_MEASUREMENT: u64 = 1 << 3;
pub const FIELD_GUEST_SVN: u64 = 1 << 4;
pub const FIELD_TCB_VERSION: u64 = 1 << 5;
const FIELD_DEFINED: u64 = (1 << 6) - 1;

/// Size of the derived key.
pub const DERIVED_KEY_SIZE: usize = 32;

/// MSG_KEY_REQ payload format
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, FromBytes, KnownLayout, Immutable, IntoBytes)]
pub struct SnpDerivedKeyRequest {
    /// 0: derive from the VCEK, 1: derive from the VMRK
    pub root_key_select: u32,
    /// Reserved, must be zero
    rsvd: u32,
    /// Bitmask of guest fields mixed into the key, see the `FIELD_*` constants
    pub guest_field_select: u64,
    /// VMPL mixed into the key. Must be at least the caller's VMPL.
    pub vmpl: u32,
    /// Guest SVN mixed into the key. Must not exceed the SVN at launch.
    pub guest_svn: u32,
    /// TCB version mixed into the key. Must not exceed the committed TCB.
    pub tcb_version: u64,
}

impl SnpDerivedKeyRequest {
    /// A request for a VMPL0 key. Refuses field bits the ABI does not define,
    /// so a typo cannot silently drop a field from the binding.
    pub fn new(
        root_key_select: u32,
        guest_field_select: u64,
        guest_svn: u32,
        tcb_version: u64,
    ) -> Result<Self, SvsmError> {
        if root_key_select > ROOT_KEY_VMRK || guest_field_select & !FIELD_DEFINED != 0 {
            return Err(SvsmError::InvalidParameter);
        }
        Ok(Self {
            root_key_select,
            rsvd: 0,
            guest_field_select,
            vmpl: 0,
            guest_svn,
            tcb_version,
        })
    }
}

/// MSG_KEY_RSP payload format
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, FromBytes, KnownLayout, Immutable, IntoBytes)]
pub struct SnpDerivedKeyResponse {
    /// 0 on success, 0x16 for invalid parameters
    status: u32,
    /// Reserved
    _reserved: [u8; 0x1c],
    /// The derived key
    key: [u8; DERIVED_KEY_SIZE],
}

impl SnpDerivedKeyResponse {
    /// The derived key, if the PSP reported success.
    pub fn key(&self) -> Result<[u8; DERIVED_KEY_SIZE], SvsmError> {
        if self.status != 0 {
            return Err(SvsmError::SnpGuestRequest(self.status));
        }
        Ok(self.key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    #[test]
    fn layouts_match_the_abi() {
        assert_eq!(size_of::<SnpDerivedKeyRequest>(), 0x20);
        assert_eq!(offset_of!(SnpDerivedKeyRequest, guest_field_select), 0x08);
        assert_eq!(offset_of!(SnpDerivedKeyRequest, vmpl), 0x10);
        assert_eq!(offset_of!(SnpDerivedKeyRequest, guest_svn), 0x14);
        assert_eq!(offset_of!(SnpDerivedKeyRequest, tcb_version), 0x18);
        assert_eq!(size_of::<SnpDerivedKeyResponse>(), 0x40);
        assert_eq!(offset_of!(SnpDerivedKeyResponse, key), 0x20);
    }

    #[test]
    fn refuses_undefined_fields_and_root_keys() {
        assert!(SnpDerivedKeyRequest::new(ROOT_KEY_VCEK, FIELD_MEASUREMENT, 0, 0).is_ok());
        assert!(SnpDerivedKeyRequest::new(ROOT_KEY_VCEK, 1 << 6, 0, 0).is_err());
        assert!(SnpDerivedKeyRequest::new(2, FIELD_MEASUREMENT, 0, 0).is_err());
    }

    #[test]
    fn requests_are_always_vmpl0() {
        let r = SnpDerivedKeyRequest::new(ROOT_KEY_VCEK, FIELD_MEASUREMENT, 0, 0).unwrap();
        assert_eq!({ r.vmpl }, 0);
    }
}
