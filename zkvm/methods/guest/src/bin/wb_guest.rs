// methods/guest/src/bin/wb_guest.rs
#![no_main]
#![no_std]

extern crate alloc;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use risc0_zkvm::guest::{entry, env};
entry!(main);

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tiny_keccak::{Hasher, Keccak};

#[derive(Serialize, Deserialize, Debug)]
pub struct Public {
    pub h_a: [u8; 32],          // SHA256(A_int.bin)
    pub h_mu: [u8; 32],         // SHA256(mu_int.bin)
    pub sig_msg_hex: String,    // 공개키/메시지 HEX (0x 접두 허용)
    pub l: u32,                 // 비트 길이
    pub tau: u32,               // 허용 해밍 거리
    pub scale: i64,             // 정보용(연산에는 영향 없음)
    pub sign_zero_rule: String, // "ge_zero_is_one"
    pub version: u8,            // 프로토콜 버전
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Secret {
    pub a_flat: Vec<i64>, // A_int: (l*d) 평탄화
    pub l: u32,
    pub d: u32,
    pub mu: Vec<i64>,     // 길이 d
}

// 저널은 PH(32바이트)만 커밋

fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new(); h.update(data); h.finalize().into()
}

fn first_n_bits_msb(digest32: [u8; 32], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    for byte in digest32 {
        for i in (0..8).rev() {
            out.push(((byte >> i) & 1) as u8);
        }
    }
    out.truncate(n);
    out
}

fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(x,y)| if x!=y {1} else {0}).sum()
}

const DOMAIN: &[u8] = b"PK-PoMLO:v1";

fn rule_id(name: &str) -> u8 {
    match name {
        "ge_zero_is_one" => 0,
        _ => 0,
    }
}

fn keccak256_packed(chunks: &[&[u8]]) -> [u8; 32] {
    let mut k = Keccak::v256();
    for c in chunks { k.update(c); }
    let mut out = [0u8; 32];
    k.finalize(&mut out);
    out
}

#[inline] fn u32_be(n: u32) -> [u8; 4] { n.to_be_bytes() }
#[inline] fn u8_be(n: u8) -> [u8; 1] { [n] }

pub fn main() {
    // 1) 공개/비밀 입력
    let public: Public = env::read();
    let secret: Secret = env::read();

    // Bind A and mu to their public commitments (SHA256(i64 LE bytes))
    fn hash_i64_le(vec: &Vec<i64>) -> [u8; 32] {
        let mut h = Sha256::new();
        for v in vec {
            h.update(&v.to_le_bytes());
        }
        h.finalize().into()
    }
    let h_a_calc = hash_i64_le(&secret.a_flat);
    assert_eq!(h_a_calc, public.h_a, "A hash mismatch");
    let h_mu_calc = hash_i64_le(&secret.mu);
    assert_eq!(h_mu_calc, public.h_mu, "mu hash mismatch");

    // 2) b = first_l_bits(SHA256(sig_msg_hex))
    let sig_msg_clean = public.sig_msg_hex.trim_start_matches("0x");
    let sig_msg_bytes = hex::decode(sig_msg_clean).expect("bad sig_msg hex");
    let digest = sha256_bytes(&sig_msg_bytes);
    let l = public.l as usize;
    let b = first_n_bits_msb(digest, l);

    // 3) A(u)와 μ로 b′ 생성: z = A * μ (정수 내적) → b′ = sign(z)
    let l_u = secret.l as usize;
    let d_u = secret.d as usize;
    assert!(l_u == l, "l mismatch");
    assert!(secret.a_flat.len() == l_u * d_u, "A_flat shape mismatch");

    let mut b_prime = vec![0u8; l_u];
    for i in 0..l_u {
        let mut acc: i128 = 0;
        for j in 0..d_u {
            let a_ij = secret.a_flat[i * d_u + j] as i128;
            let mu_j = secret.mu[j] as i128;
            acc += a_ij * mu_j;
        }
        b_prime[i] = if acc >= 0 { 1 } else { 0 }; // sign(0) = 1
    }

    // 4) Hamming(b, b′) ≤ τ
    let hd = hamming(&b, &b_prime);
    assert!(hd <= public.tau, "HD {} > tau {}", hd, public.tau);

    // 5) PH = keccak256(DOMAIN, version, hA, hMu, keccak(sig_msg), L_BE, tau_BE, ruleId)
    let sig_hash = keccak256_packed(&[&sig_msg_bytes]);
    let rid = rule_id(&public.sign_zero_rule);
    let ph = keccak256_packed(&[
        DOMAIN,
        &u8_be(public.version),
        &public.h_a,
        &public.h_mu,
        &sig_hash,
        &u32_be(public.l),
        &u32_be(public.tau),
        &u8_be(rid),
    ]);

    // 저널엔 PH(32바이트)만 커밋
    env::commit_slice(&ph);
}
