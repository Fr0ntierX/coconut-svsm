// SPDX-License-Identifier: MIT OR Apache-2.0

//! A minimal CBOR (RFC 8949) encoder for the fixed structures the DICE
//! tokens use: integers, byte and text strings, arrays and maps of known
//! length. Definite lengths only, shortest-form heads, which is what the
//! deterministic encoding rules ask for. Decoding is the verifier's job.

extern crate alloc;

use alloc::vec::Vec;

const UINT: u8 = 0;
const NINT: u8 = 1;
const BYTES: u8 = 2;
const TEXT: u8 = 3;
const ARRAY: u8 = 4;
const MAP: u8 = 5;

fn head(out: &mut Vec<u8>, major: u8, value: u64) {
    let m = major << 5;
    match value {
        0..=23 => out.push(m | value as u8),
        24..=0xff => out.extend_from_slice(&[m | 24, value as u8]),
        0x100..=0xffff => {
            out.push(m | 25);
            out.extend_from_slice(&(value as u16).to_be_bytes());
        }
        0x1_0000..=0xffff_ffff => {
            out.push(m | 26);
            out.extend_from_slice(&(value as u32).to_be_bytes());
        }
        _ => {
            out.push(m | 27);
            out.extend_from_slice(&value.to_be_bytes());
        }
    }
}

/// A signed integer: major type 0 for non-negative, 1 for negative.
pub fn int(out: &mut Vec<u8>, value: i64) {
    if value >= 0 {
        head(out, UINT, value as u64);
    } else {
        // -1 - value, computed without overflowing on i64::MIN.
        head(out, NINT, !(value as u64));
    }
}

pub fn bytes(out: &mut Vec<u8>, value: &[u8]) {
    head(out, BYTES, value.len() as u64);
    out.extend_from_slice(value);
}

pub fn text(out: &mut Vec<u8>, value: &str) {
    head(out, TEXT, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

/// The head of an array of `len` items; the caller writes the items.
pub fn array(out: &mut Vec<u8>, len: usize) {
    head(out, ARRAY, len as u64);
}

/// The head of a map of `len` pairs; the caller writes key, value, key, ...
pub fn map(out: &mut Vec<u8>, len: usize) {
    head(out, MAP, len as u64);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(f: impl Fn(&mut Vec<u8>)) -> Vec<u8> {
        let mut v = Vec::new();
        f(&mut v);
        v
    }

    // Vectors from RFC 8949 Appendix A.
    #[test]
    fn integers_match_rfc8949() {
        assert_eq!(enc(|o| int(o, 0)), [0x00]);
        assert_eq!(enc(|o| int(o, 23)), [0x17]);
        assert_eq!(enc(|o| int(o, 24)), [0x18, 0x18]);
        assert_eq!(enc(|o| int(o, 1000)), [0x19, 0x03, 0xe8]);
        assert_eq!(enc(|o| int(o, 1_000_000)), [0x1a, 0x00, 0x0f, 0x42, 0x40]);
        assert_eq!(enc(|o| int(o, -1)), [0x20]);
        assert_eq!(enc(|o| int(o, -10)), [0x29]);
        assert_eq!(enc(|o| int(o, -100)), [0x38, 0x63]);
        assert_eq!(enc(|o| int(o, -1000)), [0x39, 0x03, 0xe7]);
        assert_eq!(enc(|o| int(o, i64::MIN))[0], 0x3b);
    }

    #[test]
    fn strings_and_containers_match_rfc8949() {
        assert_eq!(enc(|o| bytes(o, &[1, 2, 3, 4])), [0x44, 1, 2, 3, 4]);
        assert_eq!(enc(|o| text(o, "IETF")), [0x64, b'I', b'E', b'T', b'F']);
        assert_eq!(enc(|o| array(o, 3)), [0x83]);
        assert_eq!(enc(|o| map(o, 2)), [0xa2]);
        // The private claim key the profile uses.
        assert_eq!(enc(|o| int(o, -70001)), [0x3a, 0x00, 0x01, 0x11, 0x70]);
    }
}
