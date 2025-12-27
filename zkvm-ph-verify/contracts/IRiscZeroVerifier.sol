// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @notice RISC Zero Groth16 verifier 최소 인터페이스
/// @dev 실패 시 revert, 성공 시 아무 것도 반환하지 않음
interface IRiscZeroVerifier {
    /// @param imageId        program/image ID (bytes32)
    /// @param journalDigest  SHA-256(journal) (bytes32)
    /// @param seal           Groth16/EVM proof bytes
    function verify(
        bytes32 imageId,
        bytes32 journalDigest,
        bytes calldata seal
    ) external view;
}

