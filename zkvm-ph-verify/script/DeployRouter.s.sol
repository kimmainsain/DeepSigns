// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../contracts/MockVerifierRouter.sol";
import "forge-std/console2.sol";

contract DeployRouter is Script {
    function run() external {
        vm.startBroadcast();
        MockVerifierRouter router = new MockVerifierRouter();
        console2.log("Router deployed:", address(router));
        vm.stopBroadcast();
    }
}

