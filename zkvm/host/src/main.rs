// host/src/main.rs
use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

use sha2::{Digest, Sha256};
use risc0_zkvm::{default_prover, ExecutorEnv};
use methods::{WB_GUEST_ELF, WB_GUEST_ID};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// secret: A_int.bin/mu_int.bin, public: public.json -> receipt.bin
    Prove {
        #[arg(long)] a: PathBuf,
        #[arg(long)] mu: PathBuf,
        #[arg(long)] public: PathBuf,
        #[arg(long)] out: PathBuf,
    },
    /// verify receipt.bin against public.json
    Verify {
        #[arg(long)] public: PathBuf,
        #[arg(long)] receipt: PathBuf,
    },
}

#[derive(Serialize, Deserialize, Debug)]
struct PublicJson {
    h_a: String,         // hex(64)
    pk_hex: String,
    l: u32,
    tau: u32,
    scale: i64,
    sign_zero_rule: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct PublicGuest {
    h_a: [u8; 32],
    pk_hex: String,
    l: u32,
    tau: u32,
    scale: i64,
    sign_zero_rule: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct SecretGuest {
    a_flat: Vec<i64>,
    l: u32,
    d: u32,
    mu: Vec<i64>,
}

#[derive(Deserialize, Debug)]
struct Out {
    pass: u8,
    public_hash: [u8; 32],
    hd: u32,
}

fn hex32(hexstr: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    let bytes = hex::decode(hexstr).expect("bad hex");
    assert!(bytes.len() == 32, "H(A) must be 32 bytes");
    out.copy_from_slice(&bytes);
    out
}

// 공개입력 바인딩 해시를 호스트에서도 동일하게 계산
fn calc_public_hash(pk_hex: &str, h_a: [u8; 32], l: u32, tau: u32) -> [u8; 32] {
    let pk_clean = pk_hex.trim_start_matches("0x");
    let pk_bytes = hex::decode(pk_clean).expect("bad pk hex");
    let mut h = Sha256::new();
    h.update(&pk_bytes);
    h.update(&h_a);
    h.update(&l.to_le_bytes());
    h.update(&tau.to_le_bytes());
    h.finalize().into()
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Prove { a, mu, public, out } => {
            // 파일 읽기
            let a_bytes = fs::read(&a)?;
            let mu_bytes = fs::read(&mu)?;
            let a_flat: Vec<i64> = a_bytes.chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
            let mu_vec: Vec<i64> = mu_bytes.chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();

            // public.json 파싱
            let pj: PublicJson = serde_json::from_slice(&fs::read(&public)?)?;
            let h_a = hex32(&pj.h_a);

            // l, d 유추
            let d = mu_vec.len() as u32;
            let l = (a_flat.len() as u32) / d;

            let pub_guest = PublicGuest {
                h_a,
                pk_hex: pj.pk_hex,
                l: pj.l,
                tau: pj.tau,
                scale: pj.scale,
                sign_zero_rule: pj.sign_zero_rule,
            };
            let sec_guest = SecretGuest { a_flat, l, d, mu: mu_vec };

            // 게스트 실행 환경
            let env = ExecutorEnv::builder()
                .write(&pub_guest)?
                .write(&sec_guest)?
                .build()?;

            // 증명
            let prover = default_prover();
            let receipt = prover.prove(env, WB_GUEST_ELF)?;

            // 저장
            fs::write(out, bincode::serialize(&receipt)?)?;
            println!("OK: receipt written");
        }

        Cmd::Verify { public, receipt } => {
            // public.json 읽기
            let pj: PublicJson = serde_json::from_slice(&fs::read(&public)?)?;

            // receipt 검증
            let r_bytes = fs::read(&receipt)?;
            let receipt: risc0_zkvm::Receipt = bincode::deserialize(&r_bytes)?;
            receipt.verify(WB_GUEST_ID)?;

            // 저널 디코드
            let out: Out = receipt.journal.decode().expect("decode journal");
            assert_eq!(out.pass, 1, "journal says not pass");

            // 공개입력 바인딩 해시 재계산 → 대조
            let h_a = hex32(&pj.h_a);
            let calc = calc_public_hash(&pj.pk_hex, h_a, pj.l, pj.tau);
            assert_eq!(out.public_hash, calc, "public_hash mismatch");

            println!("OK: verified (hd={}, tau={})", out.hd, pj.tau);
        }
    }

    Ok(())
}

