// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../contracts/Registry.sol";
import "../contracts/MockRiscZeroVerifier.sol";

contract RegistryTest is Test {
    Registry reg;
    MockRiscZeroVerifier mv;

    function setUp() public {
        mv = new MockRiscZeroVerifier();
        reg = new Registry(IRiscZeroVerifier(address(mv)));
    }

    function test_Verify_OK() public {
        bytes32 programId = 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA; // WB_GUEST_ID (image ID)
        bytes32 ph        = 0xBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB; // host verify에 찍힌 PH

        bytes32 claimId = keccak256(abi.encode(programId, ph));
        reg.register(claimId, programId, ph, address(this));

        // 모의 receipt: "끝 32B가 PH"라고 가정 (Mock이 이걸 keccak해서 journalDigest 반환)
        bytes memory fakeReceipt = abi.encodePacked(hex"00", ph);

        reg.verifyReceipt(claimId, fakeReceipt); // require 통과하면 테스트 성공
    }
}

