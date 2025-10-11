use anyhow::{Context, Result};
use clap::Parser;
use sha2::{Digest as Sha256Digest, Sha256};
use std::{fs, path::PathBuf};

use risc0_ethereum_contracts::encode_seal;       // seal 인코딩
use risc0_zkvm::Receipt;
use methods::WB_GUEST_ID;

/// on-chain 검증 입력을 준비: seal(EVM) 추출 + SHA256(PH) 계산 + programId 출력
#[derive(Parser, Debug)]
struct Cli {
    /// path to receipt.bin (bincode-serialized risc0_zkvm::Receipt)
    #[arg(long)]
    receipt: PathBuf,

    /// PH (0x + 64 hex), 게스트 저널로 커밋한 32바이트 public data hash
    #[arg(long)]
    ph: String,

    /// 출력할 EVM용 seal 파일 경로 (예: data/seal.evm)
    #[arg(long)]
    out_seal: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // 1) receipt 로드
    let receipt_bytes = fs::read(&cli.receipt)
        .with_context(|| format!("read {:?}", cli.receipt))?;
    let receipt: Receipt = bincode::deserialize(&receipt_bytes)
        .context("bincode::deserialize(receipt)")?;

    // 2) EVM 검증기 입력용 seal 인코딩 & 저장
    let seal = encode_seal(&receipt)
        .context("encode_seal")?;
    fs::write(&cli.out_seal, &seal)
        .with_context(|| format!("write {:?}", cli.out_seal))?;

    // 3) PH(32바이트 원시값) → SHA256(PH) 계산 (저널 다이제스트)
    let ph_hex = cli.ph.trim_start_matches("0x");
    let ph_bytes = hex::decode(ph_hex).context("bad PH hex")?;
    assert_eq!(ph_bytes.len(), 32, "PH must be 32 bytes");
    let jd: [u8; 32] = Sha256::digest(&ph_bytes).into();

    // 4) programId(bytes32) 출력 (WB_GUEST_ID: [u32; 8] → big-endian bytes32)
    let mut pid = [0u8; 32];
    for (i, w) in WB_GUEST_ID.iter().enumerate() {
        pid[i*4..i*4+4].copy_from_slice(&w.to_be_bytes());
    }

    println!("programId (imageId)  = 0x{}", hex::encode(pid));
    println!("PH (as given)        = 0x{}", hex::encode(&ph_bytes));
    println!("journalDigest(SHA256(PH)) = 0x{}", hex::encode(jd));
    println!("seal.evm bytes       = {} bytes -> {:?}",
             seal.len(), cli.out_seal);

    // cast/curl 등에서 바로 쓰기 편하게 헬퍼도 같이 출력
    println!("SEAL_HEX=0x{}", hex::encode(seal));
    Ok(())
}

