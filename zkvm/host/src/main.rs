// host/src/main.rs
use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};
use sha3::{Digest as KeccakDigest, Keccak256};

use sha2::Sha256;
use methods::{WB_GUEST_ELF, WB_GUEST_ID};
use risc0_zkvm::{default_prover, ExecutorEnv, ProverOpts, ReceiptKind};

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
    /// verify: public.json + receipt.bin
    Verify {
        #[arg(long)] public: PathBuf,
        #[arg(long)] receipt: PathBuf,
    },
}

#[derive(Serialize, Deserialize, Debug)]
struct PublicJson {
    h_a: String,          // hex(64)
    h_mu: String,         // hex(64) of mu
    sig_msg_hex: String,
    l: u32,
    tau: u32,
    scale: i64,
    sign_zero_rule: String,
    // 새 필드(옵션): 없으면 기본 1로 동작
    #[serde(default)]
    version: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug)]
struct SecretGuest {
    a_flat: Vec<i64>,
    l: u32,
    d: u32,
    mu: Vec<i64>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PublicGuest {
    h_a: [u8; 32],
    h_mu: [u8; 32],
    sig_msg_hex: String,
    l: u32,
    tau: u32,
    scale: i64,
    sign_zero_rule: String,
    version: u8,
}

// 저널은 PH(32바이트)만 커밋하므로 별도 Out 구조체 불필요

fn hex32(s: &str) -> [u8; 32] {
    let s = s.trim_start_matches("0x");
    let v = hex::decode(s).expect("bad hex32");
    assert_eq!(v.len(), 32);
    let mut out = [0u8; 32];
    out.copy_from_slice(&v);
    out
}

const DOMAIN: &[u8] = b"PK-PoMLO:v1";

fn rule_id(name: &str) -> u8 {
    match name {
        "ge_zero_is_one" => 0,
        _ => 0,
    }
}

fn calc_ph_keccak(
    sig_msg_hex: &str,
    h_a: [u8; 32],
    h_mu: [u8; 32],
    l: u32,
    tau: u32,
    version: u8,
    rule_id: u8,
) -> [u8; 32] {
    let sig_bytes = hex::decode(sig_msg_hex.trim_start_matches("0x")).expect("bad sig_msg hex");
    let sig_hash = Keccak256::digest(&sig_bytes); // bytes32
    let mut k = Keccak256::new();
    k.update(DOMAIN);
    k.update([version]);
    k.update(h_a);
    k.update(h_mu);
    k.update(sig_hash);
    k.update(l.to_be_bytes());
    k.update(tau.to_be_bytes());
    k.update([rule_id]);
    k.finalize().into()
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Prove { a, mu, public, out } => {
            // 파일 읽기
            let a_bytes = fs::read(&a)?;
            let mu_bytes = fs::read(&mu)?;
            let a_flat: Vec<i64> = a_bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let mu_vec: Vec<i64> = mu_bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();

            // public.json 파싱
            let pj: PublicJson = serde_json::from_slice(&fs::read(&public)?)?;
            let h_a = hex32(&pj.h_a);
            let h_mu = hex32(&pj.h_mu);

            // host-side consistency checks: h(A), h(mu) 검증
            let calc_h_a: [u8; 32] = {
                let mut h = Sha256::new();
                h.update(&a_bytes);
                h.finalize().into()
            };
            assert_eq!(h_a, calc_h_a, "public.h_a doesn't match SHA256(a)");
            let calc_h_mu: [u8; 32] = {
                let mut h = Sha256::new();
                h.update(&mu_bytes);
                h.finalize().into()
            };
            assert_eq!(h_mu, calc_h_mu, "public.h_mu doesn't match SHA256(mu)");

            // l, d 유추
            let d = mu_vec.len() as u32;
            let l = (a_flat.len() as u32) / d;

            // 공개 입력
            let pub_guest = PublicGuest {
                h_a,
                h_mu,
                sig_msg_hex: pj.sig_msg_hex,
                l,
                tau: pj.tau,
                scale: pj.scale,
                sign_zero_rule: pj.sign_zero_rule,
                version: pj.version.unwrap_or(1),
            };
            let sec_guest = SecretGuest { a_flat, l, d, mu: mu_vec };

            let env = ExecutorEnv::builder()
                .write(&pub_guest)?
                .write(&sec_guest)?
                .build()?;

            // 증명: Groth16 영수증으로 생성
            let prover = default_prover();
            let opts = ProverOpts::default().with_receipt_kind(ReceiptKind::Groth16);
            let info = prover.prove_with_opts(env, WB_GUEST_ELF, &opts)?;
            let receipt = info.receipt;

            fs::write(out, bincode::serialize(&receipt)?)?;
            println!("OK: Groth16 receipt written");
        }

        Cmd::Verify { public, receipt } => {
            // public.json 읽기
            let pj: PublicJson = serde_json::from_slice(&fs::read(&public)?)?;
            let version = pj.version.unwrap_or(1);
            let rid = rule_id(&pj.sign_zero_rule);

            // receipt 복원
            let receipt: risc0_zkvm::Receipt = bincode::deserialize(&fs::read(&receipt)?)?;

            // 프로그램 ID 검사
            receipt.verify(WB_GUEST_ID)?;

            // 게스트 저널은 PH(32B)만 커밋 → 그대로 비교
            let journal_bytes: Vec<u8> = receipt.journal.bytes.clone();
            assert_eq!(journal_bytes.len(), 32, "journal must be 32-byte PH");

            let h_a = hex32(&pj.h_a);
            let h_mu = hex32(&pj.h_mu);
            let ph = calc_ph_keccak(&pj.sig_msg_hex, h_a, h_mu, pj.l, pj.tau, version, rid);

            assert_eq!(&journal_bytes[..], &ph, "PH mismatch between journal and local calc");
            // 프로그램 ID를 big-endian 바이트32로 출력
            let mut id_bytes = [0u8; 32];
            for (i, w) in WB_GUEST_ID.iter().enumerate() {
                id_bytes[i*4..i*4+4].copy_from_slice(&w.to_be_bytes());
            }
            println!("programId (imageId) = 0x{}", hex::encode(id_bytes));
            println!("Public Data Hash = 0x{}", hex::encode(ph));
            println!("OK: verified");
        }
    }

    Ok(())
}
