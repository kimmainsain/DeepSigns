// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IRiscZeroVerifier {
    function verify(bytes calldata receipt, bytes32 imageId)
        external
        view
        returns (bytes32 journalDigest);
}
