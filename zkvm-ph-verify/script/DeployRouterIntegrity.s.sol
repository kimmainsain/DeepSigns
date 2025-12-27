// script/DeployRouterIntegrity.s.sol
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
// 파일명만 변경되었고 컨트랙트명은 Router라고 가정
import {Router} from "../contracts/Router_Integrity.sol";

contract DeployRouterIntegrity is Script {
    function run() external returns (address) {
        address verifier = vm.envAddress("VERIFIER"); // Sepolia Router 등 환경변수로 주입
        vm.startBroadcast();
        Router r = new Router(verifier);
        vm.stopBroadcast();
        console2.log("ROUTER_INTEGRITY", address(r));
        return address(r);
    }
}

