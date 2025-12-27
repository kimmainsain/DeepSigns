// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// RISC Zero 실제 Verifier 인터페이스(verify(bytes,bytes32,bytes32))
import {IRiscZeroVerifier as RZVerifier} from
    "risc0-ethereum/contracts/src/IRiscZeroVerifier.sol";

/**
 * Registry가 기대하는 시그니처(verify(bytes,bytes32,bytes32))를 그대로 제공하고
 * 실제 RISC Zero Verifier로 위임하는 어댑터 라우터.
 */
contract RouterCompat {
    RZVerifier public immutable verifier;

    event Verified(bytes32 indexed imageId, bytes32 indexed journalDigest);

    constructor(address verifier_) {
        require(verifier_ != address(0), "zero verifier");
        verifier = RZVerifier(verifier_);
    }

    // Registry가 호출하는 그대로: (proofSeal, programId, journalDigest)
    function verify(bytes calldata proofSeal, bytes32 programId, bytes32 journalDigest)
        external
        returns (bool)
    {
        // 실제 검증기는 (seal, imageId, journalDigest)
        verifier.verify(proofSeal, programId, journalDigest); // 실패시 revert
        emit Verified(programId, journalDigest);
        return true;
    }
}

