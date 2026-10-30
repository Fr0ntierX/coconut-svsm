// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CMVM EAT tokens (F0X-80), built and signed in VMPL0.
//!
//! The layout is the `tag:fr0ntierx.com,2026:cmvm-eat-v1` profile that
//! `cmvm-dice` in uPolaris defines and `cmvm-verify` appraises: a monitor
//! token signed by the monitor alias key, carrying the nonce, the monitor
//! measurement, each partition's nested token under `submods`, and the
//! monitor's endorsement of each partition's measurement and alias key. Claim
//! keys are the RFC 9711 registrations plus the profile's private keys.
//!
//! Signatures are COSE_Sign1 (RFC 9052) with ES384, untagged.

extern crate alloc;

use alloc::vec::Vec;

use p384::ecdsa::signature::Signer;
use p384::ecdsa::{Signature, SigningKey};

use super::cbor;

pub const PROFILE: &str = "tag:fr0ntierx.com,2026:cmvm-eat-v1";

const CLAIM_NONCE: i64 = 10;
const CLAIM_PROFILE: i64 = 265;
const CLAIM_SUBMODS: i64 = 266;
const CLAIM_MEASUREMENTS: i64 = 273;
const CLAIM_PARTITION_ENDORSEMENTS: i64 = -70001;
const CLAIM_REPORT_DATA: i64 = -70002;
const CLAIM_ALIAS_KEY: i64 = -70003;

/// Measurement content type from the CoAP experimental range, for a CBOR
/// `[alg, digest]` record. `ALG_SHA384` is the profile's tag for SHA-384.
const CT_DIGEST: i64 = 65000;
const ALG_SHA384: i64 = 1;

/// COSE labels and values.
const COSE_HEADER_ALG: i64 = 1;
const COSE_ALG_ES384: i64 = -35;
const COSE_KEY_KTY: i64 = 1;
const COSE_KEY_ALG: i64 = 3;
const COSE_KTY_EC2: i64 = 2;
const COSE_EC2_CRV: i64 = -1;
const COSE_EC2_X: i64 = -2;
const COSE_EC2_Y: i64 = -3;
const COSE_CRV_P384: i64 = 2;

/// RFC 9711 measurements: `[[content-type, bstr .cbor [alg, digest]]]`.
fn measurements(out: &mut Vec<u8>, digest: &[u8]) {
    let mut record = Vec::new();
    cbor::array(&mut record, 2);
    cbor::int(&mut record, ALG_SHA384);
    cbor::bytes(&mut record, digest);
    cbor::array(out, 1);
    cbor::array(out, 2);
    cbor::int(out, CT_DIGEST);
    cbor::bytes(out, &record);
}

/// The COSE_Key for a P-384 public key given as an uncompressed SEC1 point.
pub fn cose_key(sec1: &[u8; 97]) -> Vec<u8> {
    let mut out = Vec::new();
    cbor::map(&mut out, 5);
    cbor::int(&mut out, COSE_KEY_KTY);
    cbor::int(&mut out, COSE_KTY_EC2);
    cbor::int(&mut out, COSE_KEY_ALG);
    cbor::int(&mut out, COSE_ALG_ES384);
    cbor::int(&mut out, COSE_EC2_CRV);
    cbor::int(&mut out, COSE_CRV_P384);
    cbor::int(&mut out, COSE_EC2_X);
    cbor::bytes(&mut out, &sec1[1..49]);
    cbor::int(&mut out, COSE_EC2_Y);
    cbor::bytes(&mut out, &sec1[49..97]);
    out
}

/// An untagged COSE_Sign1 over `payload` with ES384.
pub fn sign1(key: &SigningKey, payload: &[u8]) -> Vec<u8> {
    let mut protected = Vec::new();
    cbor::map(&mut protected, 1);
    cbor::int(&mut protected, COSE_HEADER_ALG);
    cbor::int(&mut protected, COSE_ALG_ES384);

    // Sig_structure = ["Signature1", protected, external_aad, payload]
    let mut tbs = Vec::new();
    cbor::array(&mut tbs, 4);
    cbor::text(&mut tbs, "Signature1");
    cbor::bytes(&mut tbs, &protected);
    cbor::bytes(&mut tbs, &[]);
    cbor::bytes(&mut tbs, payload);
    let signature: Signature = key.sign(&tbs);

    let mut out = Vec::new();
    cbor::array(&mut out, 4);
    cbor::bytes(&mut out, &protected);
    cbor::map(&mut out, 0);
    cbor::bytes(&mut out, payload);
    cbor::bytes(&mut out, &signature.to_bytes());
    out
}

/// The partition's nested token: nonce, its measurement and `report_data`,
/// signed by the partition alias key.
pub fn nested_token(
    partition_alias: &SigningKey,
    nonce: &[u8],
    partition_measurement: &[u8],
    report_data: &[u8],
) -> Vec<u8> {
    let mut claims = Vec::new();
    cbor::map(&mut claims, 3);
    cbor::int(&mut claims, CLAIM_NONCE);
    cbor::bytes(&mut claims, nonce);
    cbor::int(&mut claims, CLAIM_MEASUREMENTS);
    measurements(&mut claims, partition_measurement);
    cbor::int(&mut claims, CLAIM_REPORT_DATA);
    cbor::bytes(&mut claims, report_data);
    sign1(partition_alias, &claims)
}

/// The monitor token for one partition, signed by the monitor alias key.
pub fn monitor_token(
    monitor_alias: &SigningKey,
    nonce: &[u8],
    monitor_measurement: &[u8],
    partition_id: &str,
    partition_measurement: &[u8],
    partition_alias_public: &[u8; 97],
    nested: &[u8],
) -> Vec<u8> {
    let mut claims = Vec::new();
    cbor::map(&mut claims, 5);
    cbor::int(&mut claims, CLAIM_PROFILE);
    cbor::text(&mut claims, PROFILE);
    cbor::int(&mut claims, CLAIM_NONCE);
    cbor::bytes(&mut claims, nonce);
    cbor::int(&mut claims, CLAIM_MEASUREMENTS);
    measurements(&mut claims, monitor_measurement);

    cbor::int(&mut claims, CLAIM_SUBMODS);
    cbor::map(&mut claims, 1);
    cbor::text(&mut claims, partition_id);
    cbor::bytes(&mut claims, nested);

    cbor::int(&mut claims, CLAIM_PARTITION_ENDORSEMENTS);
    cbor::map(&mut claims, 1);
    cbor::text(&mut claims, partition_id);
    cbor::map(&mut claims, 2);
    cbor::int(&mut claims, CLAIM_MEASUREMENTS);
    measurements(&mut claims, partition_measurement);
    cbor::int(&mut claims, CLAIM_ALIAS_KEY);
    cbor::bytes(&mut claims, &cose_key(partition_alias_public));

    sign1(monitor_alias, &claims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign1_verifies_over_the_rfc9052_sig_structure() {
        use p384::ecdsa::signature::Verifier;
        let key = SigningKey::from_slice(&[7u8; 48]).unwrap();
        let msg = sign1(&key, b"payload");
        // [bstr(protected a10138 22), {}, bstr(payload), bstr(96-byte sig)]
        assert_eq!(&msg[..6], &[0x84, 0x44, 0xa1, 0x01, 0x38, 0x22]);
        assert_eq!(msg[6], 0xa0);
        let sig = Signature::from_slice(&msg[msg.len() - 96..]).unwrap();
        let mut tbs = Vec::new();
        cbor::array(&mut tbs, 4);
        cbor::text(&mut tbs, "Signature1");
        cbor::bytes(&mut tbs, &[0xa1, 0x01, 0x38, 0x22]);
        cbor::bytes(&mut tbs, &[]);
        cbor::bytes(&mut tbs, b"payload");
        assert!(key.verifying_key().verify(&tbs, &sig).is_ok());
    }
}
