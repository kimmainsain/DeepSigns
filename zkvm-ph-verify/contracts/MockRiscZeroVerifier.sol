// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IRiscZeroVerifier} from "./IRiscZeroVerifier.sol";

contract MockRiscZeroVerifier is IRiscZeroVerifier {
    function verify(bytes calldata receipt, bytes32 /*imageId*/)
        external
        pure
        returns (bytes32 journalDigest)
    {
        require(receipt.length >= 32, "short receipt");

        // calldata의 마지막 32바이트를 레지스터로 로드
        bytes32 last32;
        assembly {
            last32 := calldataload(add(receipt.offset, sub(receipt.length, 32)))
        }
        // 실제 on-chain verifier가 반환하는 journalDigest(= keccak(journalBytes))를 흉내
        return keccak256(abi.encodePacked(last32));
    }
}
