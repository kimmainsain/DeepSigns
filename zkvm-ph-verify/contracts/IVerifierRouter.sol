// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IVerifierRouter
 * @notice 실제 ZK Verifier(예: RISC Zero Groth16)로 위임 호출하는 라우터 인터페이스
 * @dev 인자 순서는 (imageId, journalDigest, seal) 고정.
 *      실패 시 반드시 revert, 성공 시 true 반환(편의용).
 */
interface IVerifierRouter {
    /**
     * @notice 증명 검증. 실패하면 revert, 성공하면 true 반환.
     * @param imageId        RISC Zero program/image ID (bytes32)
     * @param journalDigest  SHA-256(journal) (bytes32)
     * @param seal           Groth16/EVM proof 바이트
     * @return ok            검증 성공 시 true
     */
    function verify(
        bytes32 imageId,
        bytes32 journalDigest,
        bytes calldata seal
    ) external returns (bool);
}

