// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// RISC Zero 라이브러리의 인터페이스를 직접 사용 (verify(bytes,bytes32,bytes32))
import {IRiscZeroVerifier as RZVerifier} from "risc0-ethereum/contracts/src/IRiscZeroVerifier.sol";

contract Router {
    RZVerifier public immutable verifier;

    event Verified(bytes32 indexed programId, bytes32 indexed journalDigest);

    constructor(address verifier_) {
        require(verifier_ != address(0), "zero verifier");
        verifier = RZVerifier(verifier_);
    }

    function verify(bytes32 programId, bytes32 journalDigest, bytes calldata seal)
        external
        returns (bool)
    {
        verifier.verify(seal, programId, journalDigest);
        emit Verified(programId, journalDigest);
        return true;
    }
}

