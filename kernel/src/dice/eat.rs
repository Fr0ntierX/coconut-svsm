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
//!
//! A nested token can also carry the partition's wrap key (F0X-898) in a CWT
//! confirmation claim (RFC 8747), `8: {1: COSE_Key}`, byte for byte as
//! `cmvm-dice`'s `nested_token_with_wrap_key` writes it. The wrap key is the
//! HPKE key a KMS seals a released key to.

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
/// RFC 8747 confirmation claim, and its COSE_Key member.
const CLAIM_CNF: i64 = 8;
const CNF_COSE_KEY: i64 = 1;

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

/// The COSE_Key for a P-384 signing (alias) public key given as an
/// uncompressed SEC1 point. Labeled ES384.
pub fn cose_key(sec1: &[u8; 97]) -> Vec<u8> {
    let mut out = Vec::new();
    ec2_p384_key(&mut out, sec1, Some(COSE_ALG_ES384));
    out
}

/// The COSE_Key for a P-384 wrap (HPKE) public key: no `alg`, as `cmvm-dice`
/// `WrapPublicKey::to_cose` encodes it. Its bytes are the `wrap_pubkey` that
/// `report_data` hashes. `cmvm-dice` refuses an ES384-labeled key as a wrap
/// key, so an alias key can't be passed off as one.
pub fn wrap_cose_key(sec1: &[u8; 97]) -> Vec<u8> {
    let mut out = Vec::new();
    ec2_p384_key(&mut out, sec1, None);
    out
}

/// An EC2 P-384 COSE_Key in the order coset writes it: kty, alg, crv, x, y.
fn ec2_p384_key(out: &mut Vec<u8>, sec1: &[u8; 97], alg: Option<i64>) {
    cbor::map(out, if alg.is_some() { 5 } else { 4 });
    cbor::int(out, COSE_KEY_KTY);
    cbor::int(out, COSE_KTY_EC2);
    if let Some(alg) = alg {
        cbor::int(out, COSE_KEY_ALG);
        cbor::int(out, alg);
    }
    cbor::int(out, COSE_EC2_CRV);
    cbor::int(out, COSE_CRV_P384);
    cbor::int(out, COSE_EC2_X);
    cbor::bytes(out, &sec1[1..49]);
    cbor::int(out, COSE_EC2_Y);
    cbor::bytes(out, &sec1[49..97]);
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

/// The partition's nested token claims: nonce, its measurement,
/// `report_data` and, if given, the wrap key in `cnf`. The order is the one
/// `cmvm-dice` writes, so the same inputs give the same bytes.
fn nested_claims(
    nonce: &[u8],
    partition_measurement: &[u8],
    report_data: &[u8],
    wrap_key: Option<&[u8; 97]>,
) -> Vec<u8> {
    let mut claims = Vec::new();
    cbor::map(&mut claims, if wrap_key.is_some() { 4 } else { 3 });
    cbor::int(&mut claims, CLAIM_NONCE);
    cbor::bytes(&mut claims, nonce);
    cbor::int(&mut claims, CLAIM_MEASUREMENTS);
    measurements(&mut claims, partition_measurement);
    cbor::int(&mut claims, CLAIM_REPORT_DATA);
    cbor::bytes(&mut claims, report_data);
    if let Some(wrap) = wrap_key {
        // The COSE_Key goes in as a CBOR map, not a byte string.
        cbor::int(&mut claims, CLAIM_CNF);
        cbor::map(&mut claims, 1);
        cbor::int(&mut claims, CNF_COSE_KEY);
        claims.extend_from_slice(&wrap_cose_key(wrap));
    }
    claims
}

/// The partition's nested token, signed by the partition alias key. With
/// `wrap_key` (an uncompressed SEC1 P-384 point) it carries that key in
/// `cnf`. The caller checks the point is on the curve.
pub fn nested_token(
    partition_alias: &SigningKey,
    nonce: &[u8],
    partition_measurement: &[u8],
    report_data: &[u8],
    wrap_key: Option<&[u8; 97]>,
) -> Vec<u8> {
    sign1(
        partition_alias,
        &nested_claims(nonce, partition_measurement, report_data, wrap_key),
    )
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

    // Known vectors from uPolaris cmvm-dice (main 2e70873), printed by a test
    // that builds them with `WrapKey::from_scalar_bytes`,
    // `Monitor::new(Cdi::from_bytes([7; 48]), stub_measure(b"monitor"))`,
    // `partition(stub_measure(b"image"))`, `nested_token_with_wrap_key` and
    // `token`. Nonce is 32 bytes of 9, partition ID "p0". ECDSA here and there
    // is RFC 6979 deterministic, so the signed tokens match byte for byte.

    /// SEC1 point of the wrap key with scalar 0x0102..30, the one cmvm-dice's
    /// HPKE interop test uses.
    const WRAP_SEC1: &str = "04c76f2283dda95cd49b0ed9e733d2904474e37216f124e13d2c9ab4cf01021c49ad9cabb3d0b97499aef2f0ab313fa02826bc1f83451b5c8962a75caff73588d4400a6296436154fb343c393e91048a6c7bcbadc83cd8a5f26feae883156f92a1";
    /// `WrapPublicKey::to_cose()` for that key.
    const WRAP_COSE: &str = "a401022002215830c76f2283dda95cd49b0ed9e733d2904474e37216f124e13d2c9ab4cf01021c49ad9cabb3d0b97499aef2f0ab313fa02822583026bc1f83451b5c8962a75caff73588d4400a6296436154fb343c393e91048a6c7bcbadc83cd8a5f26feae883156f92a1";
    /// `compute_report_data(nonce, WRAP_COSE)`: SHA-512(nonce || COSE_Key).
    const REPORT_DATA: &str = "09eef9ccd06b511205d08969d2eac1305f635ea4dc417a06531a120c68a1e1559a9739daa91438517cd054889e31d95877c8b81ae3391d0382569edbffc0b11b";
    const MONITOR_MEASUREMENT: &str = "9d0514b37dee26bb60aee45ba5a54174520be70b772d1b46a4f87cdfec073ced5312dd6085c3f346ee8109f2872ea427";
    const PARTITION_MEASUREMENT: &str = "7158862be7ff7d2ae0a585b872f415cf09b7fe8c6ce170ef944061e7788d73c1f5835652d8afe9939b01905d5aa7c48c";
    /// `nested_token_with_wrap_key`.
    const NESTED: &str = "8444a1013822a0590117a40a58200909090909090909090909090909090909090909090909090909090909090909190111818219fde85834820158307158862be7ff7d2ae0a585b872f415cf09b7fe8c6ce170ef944061e7788d73c1f5835652d8afe9939b01905d5aa7c48c3a00011171584009eef9ccd06b511205d08969d2eac1305f635ea4dc417a06531a120c68a1e1559a9739daa91438517cd054889e31d95877c8b81ae3391d0382569edbffc0b11b08a101a401022002215830c76f2283dda95cd49b0ed9e733d2904474e37216f124e13d2c9ab4cf01021c49ad9cabb3d0b97499aef2f0ab313fa02822583026bc1f83451b5c8962a75caff73588d4400a6296436154fb343c393e91048a6c7bcbadc83cd8a5f26feae883156f92a15860e3534eb9de1a029648c154a410b94ebd3147e2bda70e94a43b9af7e7373bbe04612f415503e96c9cc0308d2a1ab9a013586f74ce6dae98556d9d314c68f39daeef030a41d03e49b2903f64f12dbd56ba90963fe464e98510c45b194f245bbf2a";
    /// `Monitor::token` carrying that nested token.
    const MONITOR_TOKEN: &str = "8444a1013822a05902d3a519010978227461673a6672306e74696572782e636f6d2c323032363a636d766d2d6561742d76310a58200909090909090909090909090909090909090909090909090909090909090909190111818219fde85834820158309d0514b37dee26bb60aee45ba5a54174520be70b772d1b46a4f87cdfec073ced5312dd6085c3f346ee8109f2872ea42719010aa16270305901838444a1013822a0590117a40a58200909090909090909090909090909090909090909090909090909090909090909190111818219fde85834820158307158862be7ff7d2ae0a585b872f415cf09b7fe8c6ce170ef944061e7788d73c1f5835652d8afe9939b01905d5aa7c48c3a00011171584009eef9ccd06b511205d08969d2eac1305f635ea4dc417a06531a120c68a1e1559a9739daa91438517cd054889e31d95877c8b81ae3391d0382569edbffc0b11b08a101a401022002215830c76f2283dda95cd49b0ed9e733d2904474e37216f124e13d2c9ab4cf01021c49ad9cabb3d0b97499aef2f0ab313fa02822583026bc1f83451b5c8962a75caff73588d4400a6296436154fb343c393e91048a6c7bcbadc83cd8a5f26feae883156f92a15860e3534eb9de1a029648c154a410b94ebd3147e2bda70e94a43b9af7e7373bbe04612f415503e96c9cc0308d2a1ab9a013586f74ce6dae98556d9d314c68f39daeef030a41d03e49b2903f64f12dbd56ba90963fe464e98510c45b194f245bbf2a3a00011170a1627030a2190111818219fde85834820158307158862be7ff7d2ae0a585b872f415cf09b7fe8c6ce170ef944061e7788d73c1f5835652d8afe9939b01905d5aa7c48c3a00011172586ea50102033822200221583064690e21801e03645c5dc3e508019a0af83f76fb23481a2d0c2aca61b6d88e5eb5885337c47d123069ebb7f96d9e676a225830db9f890b0586e7bb4254df43cba9e397ff4065627f5ce6f65b0a3002d730eaf3c3c4589297fa3eb46a95101da2274513586083778a15ee356010ebf52c5fa2fb792b6e68084a3781ad0d616561db108b273136278cab1367917547de1639e17c16d7d7953f31c4e410357d38007a62645f30d07913eb9ee8da61793df805af1f1a1ba136656f83813a663a807d4af2e1183a";

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn wrap_sec1() -> [u8; 97] {
        unhex(WRAP_SEC1).try_into().unwrap()
    }

    #[test]
    fn wrap_cose_key_matches_cmvm_dice() {
        assert_eq!(wrap_cose_key(&wrap_sec1()), unhex(WRAP_COSE));
        // The alias form differs only by the ES384 label: {1: 2, 3: -35, ...}.
        let alias = cose_key(&wrap_sec1());
        assert_eq!(&alias[..5], &[0xa5, 0x01, 0x02, 0x03, 0x38]);
        assert_eq!(&alias[6..], &unhex(WRAP_COSE)[3..]);
    }

    #[test]
    fn cnf_claim_is_rfc8747_cose_key_form() {
        let nonce = [9u8; 32];
        let pm = unhex(PARTITION_MEASUREMENT);
        let rd = unhex(REPORT_DATA);
        let with = nested_claims(&nonce, &pm, &rd, Some(&wrap_sec1()));
        let without = nested_claims(&nonce, &pm, &rd, None);
        // 8: {1: COSE_Key}, appended after the three claims, map head 3 -> 4.
        let mut expected = without.clone();
        expected[0] = 0xa4;
        expected.extend_from_slice(&[0x08, 0xa1, 0x01]);
        expected.extend_from_slice(&unhex(WRAP_COSE));
        assert_eq!(without[0], 0xa3);
        assert_eq!(with, expected);
    }

    #[test]
    fn nested_and_monitor_tokens_match_cmvm_dice() {
        use crate::dice::{alias_key, partition_cdi, sec1};
        let monitor_cdi = [7u8; 48];
        let nonce = [9u8; 32];
        let pm: [u8; 48] = unhex(PARTITION_MEASUREMENT).try_into().unwrap();
        let partition_alias = alias_key(&partition_cdi(&monitor_cdi, &pm).unwrap()).unwrap();
        let nested = nested_token(
            &partition_alias,
            &nonce,
            &pm,
            &unhex(REPORT_DATA),
            Some(&wrap_sec1()),
        );
        assert_eq!(nested, unhex(NESTED));
        let token = monitor_token(
            &alias_key(&monitor_cdi).unwrap(),
            &nonce,
            &unhex(MONITOR_MEASUREMENT),
            "p0",
            &pm,
            &sec1(&partition_alias).unwrap(),
            &nested,
        );
        assert_eq!(token, unhex(MONITOR_TOKEN));
    }
}
