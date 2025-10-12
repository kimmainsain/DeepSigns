// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @notice RISC Zero Verifier Router와 동일한 인터페이스 (성공 시 revert 없음)
interface IVerifierRouter {
    function verify(bytes calldata seal, bytes32 imageId, bytes32 journalDigest) external;
}

