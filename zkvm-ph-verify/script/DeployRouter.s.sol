// script/DeployRouter.s.sol
pragma solidity ^0.8.20;
import "forge-std/Script.sol";
import {Router} from "../contracts/Router.sol";

contract DeployRouter is Script {
    function run() external returns (address) {
        address verifier = vm.envAddress("VERIFIER"); // ← 환경변수로 전달
        vm.startBroadcast();
        Router r = new Router(verifier);
        vm.stopBroadcast();
        console2.log("ROUTER", address(r));
        return address(r);
    }
}

