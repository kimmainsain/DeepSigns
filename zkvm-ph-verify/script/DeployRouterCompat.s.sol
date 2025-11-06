// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "forge-std/console2.sol";
import {RouterCompat} from "../contracts/RouterCompat.sol";

contract DeployRouterCompat is Script {
    function run() external {
        // 환경변수로 Verifier 주소 받기
        address verifier = vm.envAddress("VERIFIER_ADDR");
        require(verifier != address(0), "VERIFIER_ADDR not set");

        vm.startBroadcast();
        RouterCompat router = new RouterCompat(verifier);
        vm.stopBroadcast();

        console2.log("RouterCompat:", address(router));
        console2.log("Verifier:", verifier);
    }
}

