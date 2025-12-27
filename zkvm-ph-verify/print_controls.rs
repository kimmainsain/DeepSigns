// print_controls.rs
use hex;
fn main() {
    // 같은 crate를 onchain_prep에서 쓰는 버전과 동일하게 링크해야 합니다.
    // Cargo.toml에 risc0-ethereum-contracts 버전을 onchain_prep와 맞추고 빌드하세요.
    let control_root = risc0_ethereum_contracts::CONTROL_ROOT;
    let control_id   = risc0_ethereum_contracts::BN254_CONTROL_ID;
    println!("CONTROL_ROOT=0x{}", hex::encode(control_root));
    println!("BN254_CONTROL_ID=0x{}", hex::encode(control_id));
}

