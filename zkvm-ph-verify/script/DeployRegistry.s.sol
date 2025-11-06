// script/DeployRegistry.s.sol
pragma solidity ^0.8.20;
import "forge-std/Script.sol";
import {Registry} from "../contracts/Registry.sol";

contract DeployRegistry is Script {
    function run() external returns (address) {
        address router = vm.envAddress("ROUTER");
        vm.startBroadcast();
		Registry reg = new Registry(router);        
		vm.stopBroadcast();
        console2.log("REGISTRY", address(reg));
        return address(reg);
    }
}

