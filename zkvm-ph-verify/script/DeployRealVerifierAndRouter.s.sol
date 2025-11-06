
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "forge-std/console2.sol";

// RISC Zero 라이브러리 (경로 확인됨)
import {RiscZeroGroth16Verifier} from "risc0-ethereum/contracts/src/groth16/RiscZeroGroth16Verifier.sol";
import {ControlID}                 from "risc0-ethereum/contracts/src/groth16/ControlID.sol";

// 우리가 쓰는 Router (외부 시그니처 유지)
import {Router} from "../contracts/Router.sol";

contract DeployRealVerifierAndRouter is Script {
    function run() external {
        vm.startBroadcast();

        // ✅ 함수 호출 아님: 괄호 없이 '상수' 접근
        bytes32 controlRoot = ControlID.CONTROL_ROOT;
        bytes32 bn254       = ControlID.BN254_CONTROL_ID;

        RiscZeroGroth16Verifier verifier = new RiscZeroGroth16Verifier(controlRoot, bn254);
        Router router = new Router(address(verifier));

        vm.stopBroadcast();

        console2.log("Verifier:", address(verifier));
        console2.log("Router:", address(router));
    }
}

