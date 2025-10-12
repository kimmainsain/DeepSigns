// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../contracts/Registry.sol";
import "forge-std/console2.sol";

contract DeployRegistry is Script {
    function run() external {
        address router = vm.envAddress("ROUTER"); // 환경변수로 주입
        vm.startBroadcast();
        Registry reg = new Registry(router);
        console2.log("Registry deployed:", address(reg));
        vm.stopBroadcast();
    }
}

