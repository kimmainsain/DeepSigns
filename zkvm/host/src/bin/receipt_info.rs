use anyhow::{Context, Result};
use clap::Parser;
use std::{fs, path::PathBuf};
use risc0_zkvm::{Receipt, InnerReceipt};

#[derive(Parser, Debug)]
struct Cli {
    /// path to bincode-serialized receipt, e.g. data/receipt.bin
    #[arg(long)]
    receipt: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let bytes = fs::read(&cli.receipt)
        .with_context(|| format!("read {:?}", cli.receipt))?;
    let r: Receipt = bincode::deserialize(&bytes)
        .context("bincode::deserialize(receipt)")?;

    // 타입 판별
    let kind = match &r.inner {
        InnerReceipt::Groth16(_) => "Groth16",
        InnerReceipt::Succinct(_) => "Succinct",
        InnerReceipt::Composite(_) => "Composite",
        InnerReceipt::Fake(_) => "Fake",
        // non_exhaustive 대비
        _ => "Unknown",
    };

    println!("kind: {}", kind);
    println!("seal_size: {} bytes", r.inner.seal_size());
    // 저널 길이 등도 참고 가능
    println!("journal_len: {} bytes", r.journal.bytes.len());

    Ok(())
}

