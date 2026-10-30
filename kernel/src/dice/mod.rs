// SPDX-License-Identifier: MIT OR Apache-2.0

//! DICE layer 0 for the CMVM monitor (F0X-80).
//!
//! The monitor's Compound Device Identifier comes from a VMPL0 key the PSP
//! derives from the VCEK, mixed with this guest's launch measurement, guest
//! policy and reported TCB. The same SVSM image, launched with the same policy
//! on the same chip at the same TCB, gets the same CDI; any of those changing
//! gives a different one. The VMRK is not used: it changes on every launch.
//!
//! The alias key is derived from the CDI the same way `cmvm-dice` derives it
//! in uPolaris, and bound into a VMPL0 attestation report as
//! `REPORT_DATA = SHA-512(REPORT_DATA_TAG || SEC1 point)`. That report, signed
//! by the VCEK, is the hardware root endorsement: it says this measurement
//! holds this alias key.
//!
//! The CDI and the alias private key never leave VMPL0.

extern crate alloc;

mod cbor;
pub mod eat;

use alloc::vec::Vec;
use core::mem::size_of;

use hkdf::Hkdf;
use p384::ecdsa::SigningKey;
use sha2::{Digest, Sha384, Sha512};
use zerocopy::{FromBytes, IntoBytes};
use zeroize::Zeroizing;

use crate::error::SvsmError;
use crate::greq::pld_key::{
    FIELD_GUEST_POLICY, FIELD_MEASUREMENT, FIELD_TCB_VERSION, ROOT_KEY_VCEK, SnpDerivedKeyRequest,
};
use crate::greq::pld_report::{AttestationReport, SnpReportRequest, SnpReportResponse};
use crate::greq::services::{get_derived_key, get_regular_report};

const INFO_MONITOR_CDI: &[u8] = b"cmvm monitor cdi v1";
const INFO_PARTITION_CDI: &[u8] = b"cmvm partition cdi v1";
const INFO_ALIAS: &[u8] = b"cmvm alias v1";

/// Domain tag hashed in front of the alias public key to form REPORT_DATA.
pub const REPORT_DATA_TAG: &[u8] = b"cmvm monitor alias v1\0";

/// Size of the uncompressed SEC1 encoding of a P-384 public key.
pub const ALIAS_PUBLIC_SIZE: usize = 97;

/// Size of an SNP attestation report.
pub const REPORT_SIZE: usize = size_of::<AttestationReport>();

/// The guest fields the monitor CDI is bound to.
const CDI_FIELDS: u64 = FIELD_MEASUREMENT | FIELD_GUEST_POLICY | FIELD_TCB_VERSION;

/// A VMPL0 attestation report carrying `report_data`.
fn vmpl0_report(report_data: [u8; 64]) -> Result<AttestationReport, SvsmError> {
    let mut buffer = [0u8; size_of::<SnpReportResponse>()];
    let request = SnpReportRequest::new(report_data, 0, 0);
    buffer[..size_of::<SnpReportRequest>()].copy_from_slice(request.as_bytes());
    get_regular_report(&mut buffer)?;
    let (response, _rest) =
        SnpReportResponse::ref_from_prefix(&buffer).map_err(|_| SvsmError::InvalidParameter)?;
    Ok(*response.report())
}

/// The monitor CDI: HKDF-SHA384 over the VCEK-rooted derived key.
fn monitor_cdi() -> Result<Zeroizing<[u8; 48]>, SvsmError> {
    Ok(monitor_cdi_and_measurement()?.0)
}

/// The monitor CDI and the monitor's launch measurement, read from the same
/// VMPL0 report.
fn monitor_cdi_and_measurement() -> Result<(Zeroizing<[u8; 48]>, [u8; 48]), SvsmError> {
    // TCB_VERSION mixes in the value the request carries, so pass the TCB the
    // VCEK is derived at, read from this guest's own report. It is never above
    // the committed TCB, which the PSP requires.
    let own = vmpl0_report([0u8; 64])?;
    let tcb = own.reported_tcb();
    let request = SnpDerivedKeyRequest::new(ROOT_KEY_VCEK, CDI_FIELDS, 0, tcb)?;
    let key = get_derived_key(&request)?;
    let mut cdi = Zeroizing::new([0u8; 48]);
    Hkdf::<Sha384>::new(None, &key[..])
        .expand(INFO_MONITOR_CDI, &mut cdi[..])
        .map_err(|_| SvsmError::InvalidParameter)?;
    Ok((cdi, *own.measurement()))
}

/// A partition's CDI: HKDF-SHA384 with the monitor CDI as input keying
/// material and the partition measurement as salt, as `cmvm-dice` derives it.
fn partition_cdi(monitor: &[u8; 48], measurement: &[u8]) -> Result<Zeroizing<[u8; 48]>, SvsmError> {
    let mut cdi = Zeroizing::new([0u8; 48]);
    Hkdf::<Sha384>::new(Some(measurement), monitor)
        .expand(INFO_PARTITION_CDI, &mut cdi[..])
        .map_err(|_| SvsmError::InvalidParameter)?;
    Ok(cdi)
}

fn sec1(key: &SigningKey) -> Result<[u8; ALIAS_PUBLIC_SIZE], SvsmError> {
    key.verifying_key()
        .to_sec1_point(false)
        .as_bytes()
        .try_into()
        .map_err(|_| SvsmError::InvalidFormat)
}

/// Measure, derive and sign for one partition: the monitor token, carrying the
/// partition's nested token and the monitor's endorsement of its measurement
/// and alias key. With [`root_endorsement`] it is a complete version 2
/// evidence bundle.
///
/// Prototype limit: the nested token is signed here, in VMPL0, with the alias
/// key derived for `partition_measurement`. In the eventual design the
/// partition holds its own key and signs it; where it does depends on how
/// partitions run (F0X-76).
pub fn attest_partition(
    nonce: &[u8],
    report_data: &[u8],
    partition_id: &str,
    partition_measurement: &[u8; 48],
) -> Result<Vec<u8>, SvsmError> {
    let (cdi, monitor_measurement) = monitor_cdi_and_measurement()?;
    let monitor_alias = alias_key(&cdi)?;
    let partition = partition_cdi(&cdi, partition_measurement)?;
    let partition_alias = alias_key(&partition)?;
    let nested = eat::nested_token(&partition_alias, nonce, partition_measurement, report_data);
    Ok(eat::monitor_token(
        &monitor_alias,
        nonce,
        &monitor_measurement,
        partition_id,
        partition_measurement,
        &sec1(&partition_alias)?,
        &nested,
    ))
}

/// The alias key for a CDI. Re-draws with a counter in the negligible case the
/// derived bytes are not a valid P-384 scalar, as `cmvm-dice` does.
fn alias_key(cdi: &[u8; 48]) -> Result<SigningKey, SvsmError> {
    let hk = Hkdf::<Sha384>::new(None, cdi);
    for ctr in 0u8..=u8::MAX {
        let mut info = Vec::from(INFO_ALIAS);
        if ctr > 0 {
            info.push(ctr);
        }
        let mut okm = Zeroizing::new([0u8; 48]);
        hk.expand(&info, &mut okm[..])
            .map_err(|_| SvsmError::InvalidParameter)?;
        if let Ok(key) = SigningKey::from_slice(&okm[..]) {
            return Ok(key);
        }
    }
    Err(SvsmError::InvalidParameter)
}

/// `REPORT_DATA` for an alias public key.
pub fn alias_report_data(alias_public: &[u8]) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(REPORT_DATA_TAG);
    h.update(alias_public);
    h.finalize().into()
}

/// The hardware root endorsement: the monitor's alias public key and a VMPL0
/// report binding it. The verifier recomputes `REPORT_DATA` from the key and
/// checks the report against the VCEK chain.
pub fn root_endorsement() -> Result<([u8; ALIAS_PUBLIC_SIZE], AttestationReport), SvsmError> {
    let cdi = monitor_cdi()?;
    let public = sec1(&alias_key(&cdi)?)?;
    let report = vmpl0_report(alias_report_data(&public))?;
    Ok((public, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_key_is_deterministic_in_the_cdi() {
        let a = alias_key(&[7u8; 48]).unwrap();
        let b = alias_key(&[7u8; 48]).unwrap();
        assert_eq!(a.verifying_key(), b.verifying_key());
        assert_ne!(
            a.verifying_key(),
            alias_key(&[8u8; 48]).unwrap().verifying_key()
        );
    }

    #[test]
    fn report_data_binds_the_whole_key() {
        let k = [1u8; ALIAS_PUBLIC_SIZE];
        let mut other = k;
        other[ALIAS_PUBLIC_SIZE - 1] ^= 1;
        assert_ne!(alias_report_data(&k), alias_report_data(&other));
    }
}
