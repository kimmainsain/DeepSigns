// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IRiscZeroVerifier} from "./IRiscZeroVerifier.sol";

contract Registry {
    struct Claim {
        bytes32 programId;  // image ID
        bytes32 ph;         // public params hash (PH)
        address owner;      // 선택: 소유자
        bool exists;
    }

    IRiscZeroVerifier public verifier;
    mapping(bytes32 => Claim) public claims;

    event Registered(bytes32 indexed claimId, bytes32 programId, bytes32 ph, address owner);
    event Verified(bytes32 indexed claimId, address indexed submitter);

    constructor(IRiscZeroVerifier _verifier) {
        verifier = _verifier;
    }

    function register(bytes32 claimId, bytes32 programId, bytes32 ph, address owner) external {
        require(!claims[claimId].exists, "already");
        claims[claimId] = Claim({programId: programId, ph: ph, owner: owner, exists: true});
        emit Registered(claimId, programId, ph, owner);
    }

    function verifyReceipt(bytes32 claimId, bytes calldata receipt) external {
        Claim memory c = claims[claimId];
        require(c.exists, "no claim");

        // 1) (모의) zkVM 검증
        bytes32 jd = verifier.verify(receipt, c.programId);

        // 2) 저널다이제스트 == keccak(PH) 확인
        require(jd == keccak256(abi.encodePacked(c.ph)), "journalDigest != keccak(PH)");
        emit Verified(claimId, msg.sender);
    }
}

