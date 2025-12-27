// host/src/bin/claim_digest.rs
use anyhow::{Context, Result};
use clap::Parser;
use std::{fs, path::PathBuf};

use risc0_zkvm::{Receipt, MaybePruned, sha}; // sha::Impl 사용
use risc0_binfmt::Digestible;                // .digest() 트레이트
use sha2::{Digest as Sha256Digest, Sha256};  // journal 해시용(선택)

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long)]
    receipt: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let bytes = fs::read(&cli.receipt).with_context(|| format!("read {:?}", cli.receipt))?;
    let r: Receipt = bincode::deserialize(&bytes).context("bincode::deserialize(receipt)")?;

    // MaybePruned<ReceiptClaim> 매칭
    let cd = match r.claim()? {
        MaybePruned::Value(claim) => claim.digest::<sha::Impl>(), // ✅ 제네릭 지정
        MaybePruned::Pruned(d)    => d,
    };
    println!("CLAIM_DIGEST=0x{}", hex::encode(cd.as_bytes()));

    // (참고) journal 바이트 SHA-256 → on-chain 비교용
    let jd: [u8; 32] = Sha256::digest(&r.journal.bytes).into();
    println!("JOURNAL_DIGEST=0x{}", hex::encode(jd));
    println!("journal_len={} bytes", r.journal.bytes.len());
    Ok(())
}

