// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./IVerifierRouter.sol";

/**
 * @title Registry
 * @notice (imageId, sha256(PH)) 앵커 저장 및 검증 호출 라우팅
 * @dev 용어:
 *      - imageId: RISC Zero program/image ID (bytes32)
 *      - PH: 저널로 쓰는 32바이트 공개값 (bytes32)
 *      - journalDigest: sha256(PH) (bytes32)
 */
contract Registry {
    IVerifierRouter public router;
    address public owner;

    // anchored[imageId][journalDigest] = true
    mapping(bytes32 => mapping(bytes32 => bool)) public anchored;

    event Registered(bytes32 indexed imageId, bytes32 indexed journalDigest, address indexed registrant);
    event Verified(bytes32 indexed imageId, bytes32 indexed journalDigest, address indexed submitter);

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

    /// @notice 앵커 고정: (imageId, sha256(PH))를 저장
    function register(bytes32 imageId, bytes32 ph /* PH raw (32B) */) external {
        bytes32 digest = sha256(abi.encodePacked(ph)); // journalDigest = sha256(PH)
        anchored[imageId][digest] = true;
        emit Registered(imageId, digest, msg.sender);
    }

    /// @notice 검증: Router를 통해 on-chain 검증을 수행
    /// @param imageId    RISC Zero imageId
    /// @param phClaim    호출자가 주장하는 PH (32B)
    /// @param journalRaw 검증에 사용될 저널 원값(32B) — 설계상 PH와 동일해야 함
    /// @param seal       Groth16/EVM proof 바이트
    /// @return ok        검증 성공 시 true (실패 시 Router/Verifier에서 revert)
    function verify(
        bytes32 imageId,
        bytes32 phClaim,
        bytes32 journalRaw,
        bytes calldata seal
    ) external returns (bool ok) {
        // [STEP 2] 주석 해제하여 "저널 == PH 주장값" 강제
        // require(journalRaw == phClaim, "journal != PH");

        // journalDigest = sha256(journalRaw)
        bytes32 digest = sha256(abi.encodePacked(journalRaw));

        // [STEP 3] 주석 해제하여 "사전 앵커 필수" 강제
        // require(anchored[imageId][digest], "not anchored");

        ok = router.verify(imageId, digest, seal); // 성공 시 true, 실패 시 revert
        emit Verified(imageId, digest, msg.sender);
    }

    /// @notice 보조: (imageId, ph)에 대한 앵커 여부 조회
    function isAnchored(bytes32 imageId, bytes32 ph) external view returns (bool) {
        bytes32 digest = sha256(abi.encodePacked(ph));
        return anchored[imageId][digest];
    }
}
