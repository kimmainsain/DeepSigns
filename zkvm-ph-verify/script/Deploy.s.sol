// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../contracts/Registry.sol";
import "forge-std/console2.sol";

contract Deploy is Script {
    function run() external {
        address router = vm.envAddress("ROUTER");
        vm.startBroadcast();                // --unlocked 또는 --private-key 로 서명
        Registry reg = new Registry(router);
        console2.log("Registry deployed:", address(reg));
        vm.stopBroadcast();
    }
}

