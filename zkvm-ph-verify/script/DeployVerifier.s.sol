// script/DeployVerifier.s.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import {RiscZeroGroth16Verifier}
  from "risc0-ethereum/contracts/src/groth16/RiscZeroGroth16Verifier.sol";

// grep로 확인한 상수값을 직접 주입 (라이브러리 버전 독립)
bytes32 constant CONTROL_ROOT        =
  0xa54dc85ac99f851c92d7c96d7318af41dbe7c0194edfcc37eb4d422a998c1f56;
bytes32 constant BN254_CONTROL_ID    =
  0x04446e66d300eb7fb45c9726bb53c793dda407a62e9601618bb43c5c14657ac0;

contract DeployVerifier is Script {
    function run() external returns (address) {
        vm.startBroadcast();
        RiscZeroGroth16Verifier v =
            new RiscZeroGroth16Verifier(CONTROL_ROOT, BN254_CONTROL_ID);
        vm.stopBroadcast();

        console2.log("VERIFIER", address(v));
        return address(v);
    }
}

