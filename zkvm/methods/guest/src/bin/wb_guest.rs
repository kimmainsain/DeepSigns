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

#[derive(Serialize, Deserialize, Debug)]
pub struct Public {
    pub h_a: [u8; 32],          // SHA256(A_int.bin)
    pub pk_hex: String,         // 공개키 HEX (0x 접두 허용)
    pub l: u32,                 // 비트 길이
    pub tau: u32,               // 허용 해밍 거리
    pub scale: i64,             // 정보용(연산에는 영향 없음)
    pub sign_zero_rule: String, // "ge_zero_is_one"
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Secret {
    pub a_flat: Vec<i64>, // A_int: (l*d) 평탄화
    pub l: u32,
    pub d: u32,
    pub mu: Vec<i64>,     // 길이 d
}

// 저널에 커밋할 출력 형식
#[derive(Serialize, Deserialize, Debug)]
pub struct Out {
    pub pass: u8,                  // 1이면 통과
    pub public_hash: [u8; 32],     // 공개입력 바인딩 해시
    pub hd: u32,                   // (선택) 해밍거리
}


fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new(); h.update(data); h.finalize().into()
}
fn first_n_bits_msb(digest32: [u8; 32], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    for byte in digest32 { for i in (0..8).rev() { out.push(((byte >> i) & 1) as u8); } }
    out.truncate(n); out
}
fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(x,y)| if x!=y {1} else {0}).sum()
}

// 공개입력 바인딩 해시 = SHA256(pk || h_a || L_le || tau_le)
fn hash_public(pk_bytes: &[u8], h_a: &[u8; 32], l: u32, tau: u32) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(pk_bytes);
    h.update(h_a);
    h.update(&l.to_le_bytes());
    h.update(&tau.to_le_bytes());
    h.finalize().into()
}


pub fn main() {
    // 1) 공개/비공개 입력
    let public: Public = env::read();
    let secret: Secret = env::read();

    // 2) H(A) 커밋 확인
    let mut a_bytes: Vec<u8> = Vec::with_capacity(secret.a_flat.len()*8);
    for v in &secret.a_flat { a_bytes.extend_from_slice(&v.to_le_bytes()); }
    let h_a_calc = sha256_bytes(&a_bytes);
    assert!(h_a_calc == public.h_a, "H(A) mismatch");

    // 3) b = first L bits of SHA256(pk)
    let pk_clean = public.pk_hex.trim_start_matches("0x");
    let pk_bytes = hex::decode(pk_clean).expect("bad pk hex");
    let digest = sha256_bytes(&pk_bytes);
    let l = public.l as usize;
    let b = first_n_bits_msb(digest, l);

    // 4) z = A * mu  (정수 내적) → b' = sign(z) (z>=0 → 1)
    let l_u = secret.l as usize;
    let d_u = secret.d as usize;
    assert!(l_u == l, "public.l != secret.l");
    assert!(secret.mu.len() == d_u, "mu dim");
    assert!(secret.a_flat.len() == l_u*d_u, "A dim");

    let mut b_prime = vec![0u8; l_u];
    for i in 0..l_u {
        let mut acc: i128 = 0;
        let row = &secret.a_flat[i*d_u .. (i+1)*d_u];
        for (aij, muj) in row.iter().zip(secret.mu.iter()) {
            acc += (*aij as i128) * (*muj as i128);
        }
        b_prime[i] = if acc >= 0 { 1 } else { 0 }; // sign(0) = 1
    }

    // 5) Hamming(b, b′) ≤ τ
    let hd = hamming(&b, &b_prime);
    assert!(hd <= public.tau, "HD {} > tau {}", hd, public.tau);

    // 6) 저널에 PASS만 기록
    let public_hash = hash_public(&pk_bytes, &public.h_a, public.l, public.tau);
    let out = Out { pass: 1, public_hash, hd };
    env::commit(&out);
}

