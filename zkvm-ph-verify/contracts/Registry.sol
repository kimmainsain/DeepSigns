// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./IVerifierRouter.sol";

/**
 * @title Registry
 * @notice (programId, sha256(PH)) 앵커 저장 및 검증 호출 라우팅
 * @dev 용어:
 *      - programId: RISC Zero program/image ID (bytes32)
 *      - PH: 저널로 쓰는 32바이트 공개값 (bytes32)
 *      - journalDigest: sha256(PH) (bytes32)
 */
contract Registry {
    IVerifierRouter public router;
    address public owner;

    // anchored[programId][journalDigest] = true
    mapping(bytes32 => mapping(bytes32 => bool)) public anchored;

    event Registered(bytes32 indexed programId, bytes32 indexed journalDigest, address indexed registrant);
    event Verified(bytes32 indexed programId, bytes32 indexed journalDigest, address indexed submitter);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address router_) {
        require(router_ != address(0), "zero router");
        owner = msg.sender;
        router = IVerifierRouter(router_);
    }

    function setRouter(address router_) external onlyOwner {
        require(router_ != address(0), "zero router");
        router = IVerifierRouter(router_);
    }

    /// @notice 앵커 고정: (programId, sha256(PH))를 저장
    function register(bytes32 programId, bytes32 ph /* PH raw (32B) */) external {
        bytes32 digest = sha256(abi.encodePacked(ph)); // journalDigest = sha256(PH)
        anchored[programId][digest] = true;
        emit Registered(programId, digest, msg.sender);
    }

    /// @notice 검증: Router를 통해 on-chain 검증을 수행
    /// @param programId    RISC Zero programId
    /// @param PH    호출자가 주장하는 PH (32B)
    /// @param journalRaw 검증에 사용될 저널 원값(32B) — 설계상 PH와 동일해야 함
    /// @param seal       Groth16/EVM proof 바이트
    /// @return ok        검증 성공 시 true (실패 시 Router/Verifier에서 revert)
	function verify(
		bytes32 programId,
		bytes32 PH,
		bytes32 journalRaw,
		bytes calldata seal
	) external returns (bool ok) {
	    require(journalRaw == PH, "journal != PH");

		bytes32 journalDigest = sha256(abi.encodePacked(journalRaw));
		require(anchored[programId][journalDigest], "not anchored");

		ok = router.verify(programId, journalDigest, seal);

		emit Verified(programId, journalDigest, msg.sender);
	}

    /// @notice 보조: (programId, ph)에 대한 앵커 여부 조회
    function isAnchored(bytes32 programId, bytes32 ph) external view returns (bool) {
        bytes32 digest = sha256(abi.encodePacked(ph));
        return anchored[programId][digest];
    }
}
