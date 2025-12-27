// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Receipt를 별칭으로 같이 import 해야 함 (인터페이스 내부 네임스페이스 아님)
import {
    IRiscZeroVerifier as RZVerifier,
    Receipt as RZReceipt
} from "risc0-ethereum/contracts/src/IRiscZeroVerifier.sol";

contract Router {
    RZVerifier public immutable verifier;

    event VerifiedIntegrity(bytes32 indexed claimDigest, bytes32 indexed ph, address indexed sender);

    constructor(address verifier_) {
        require(verifier_ != address(0), "zero verifier");
        verifier = RZVerifier(verifier_);
    }

    /// view 점검용: 성공 시 true(eth_call 결과 0x01...), 실패 시 revert
    function check(bytes calldata seal, bytes32 claimDigest) external view returns (bool) {
        RZReceipt memory r = RZReceipt({seal: seal, claimDigest: claimDigest});
        verifier.verifyIntegrity(r); // 원 함수는 반환값 없음
        return true;
    }

    /// 검증 + 이벤트 앵커
    function submit(bytes calldata seal, bytes32 claimDigest, bytes32 ph) external returns (bool) {
        RZReceipt memory r = RZReceipt({seal: seal, claimDigest: claimDigest});
        verifier.verifyIntegrity(r); // 실패 시 revert
        emit VerifiedIntegrity(claimDigest, ph, msg.sender);
        return true;
    }
}

