// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// RISC Zero 라이브러리의 인터페이스를 직접 사용 (verify(bytes,bytes32,bytes32))
import {IRiscZeroVerifier as RZVerifier} from "risc0-ethereum/contracts/src/IRiscZeroVerifier.sol";

contract Router {
    RZVerifier public immutable verifier;

    event Verified(bytes32 indexed imageId, bytes32 indexed journalDigest);

    constructor(address verifier_) {
        require(verifier_ != address(0), "zero verifier");
        verifier = RZVerifier(verifier_);
    }

    // 바깥에 노출하는 시그니처는 기존 유지:
    // verify(imageId, journalDigest, seal) returns (bool)
    function verify(bytes32 imageId, bytes32 journalDigest, bytes calldata seal)
        external
        returns (bool)
    {
        // 실제 RISC Zero Verifier는 (seal, imageId, journalDigest) 순서
        verifier.verify(seal, imageId, journalDigest); // 실패 시 revert
        emit Verified(imageId, journalDigest);
        return true;
    }
}

