// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./IVerifierRouter.sol";

/// @notice 로컬 경로 점검용 Mock (암호학적 검증 없음)
contract MockVerifierRouter is IVerifierRouter {
    event Called(bytes seal, bytes32 imageId, bytes32 journalDigest);

    function verify(bytes calldata seal, bytes32 imageId, bytes32 journalDigest) external override {
        // 실제 라우터는 여기서 증명을 검증하지만,
        // Mock은 단지 이벤트만 내보내고 revert하지 않음.
        emit Called(seal, imageId, journalDigest);
    }
}

