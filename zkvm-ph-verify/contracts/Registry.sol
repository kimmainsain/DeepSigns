// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./IVerifierRouter.sol";

/// @notice 간단한 앵커 + 검증 경로 컨트랙트
/// - (Program ID, Public Data Hash) 조합을 등록(앵커) 후
/// - verify에서 Proof Seal + Journal(= Public Data Hash 32바이트)을 받아
///   1) 앵커 존재 확인
///   2) Journal 길이/동일성 체크
///   3) Journal Digest = SHA-256(Journal) 계산
///   4) Verifier Router.verify 호출(성공 = revert 없음)
contract Registry {
    // -------------------------------
    // [MOCK ↔ REAL 라우터 교체 안내]
    // -------------------------------
    // 1) 로컬(Mock) 모드:
    //    - MockVerifierRouter를 배포하고, 그 주소를 constructor(router_)에 넘겨서 배포
    //    - 나중에 실제 라우터로 바꾸고 싶으면 setRouter() 호출
    //
    // 2) 실제(테스트넷/메인넷) 모드:
    //    - constructor(router_)에 해당 체인의 "실제 Verifier Router 주소"를 넣어 배포
    //    - 또는 배포 후 setRouter()로 교체
    //
    // 👉 핵심: "router" 주소만 바꾸면 나머지 로직 변경 없이 동일하게 동작
    IVerifierRouter public router;

    address public owner;
    modifier onlyOwner() { require(msg.sender == owner, "not owner"); _; }

    // (Program ID, Public Data Hash) → 등록 여부
    mapping(bytes32 => mapping(bytes32 => bool)) public anchored;

    event Registered(bytes32 programId, bytes32 publicDataHash, address indexed owner);
    event Verified(bytes32 programId, bytes32 publicDataHash, address indexed submitter);

    constructor(address router_) {
        owner = msg.sender;
        router = IVerifierRouter(router_);
    }

    /// @notice 라우터 교체 (Mock ↔ 실제)
    function setRouter(address router_) external onlyOwner {
        router = IVerifierRouter(router_);
    }

    /// @notice 앵커 등록 (Claim ID 없이, (Program ID, Public Data Hash)로 식별)
    function register(bytes32 programId, bytes32 publicDataHash) external {
        anchored[programId][publicDataHash] = true;
        emit Registered(programId, publicDataHash, msg.sender);
    }

    /// @notice 검증: Proof Seal + Journal(= Public Data Hash 32바이트 원문)
    function verify(
        bytes32 programId,
        bytes32 publicDataHash,
        bytes calldata proofSeal,
        bytes calldata journal
    ) external {
        require(anchored[programId][publicDataHash], "not anchored");
        require(journal.length == 32, "journal len");

        // bytes → bytes32 (길이 32 미만이면 abi.decode에서 revert)
        bytes32 journal32 = abi.decode(journal, (bytes32));
        require(journal32 == publicDataHash, "journal != PDH");

        // 검증기 입력 규약: SHA-256(Journal)
        bytes32 journalDigest = sha256(journal);

        // (Mock or Real) Verifier Router 호출
        // - Mock: 이벤트만 발생, revert 없음 → 정상 흐름 확인용
        // - Real: 실제 Groth16 증명 검증 수행 → 성공 시 revert 없음
        router.verify(proofSeal, programId, journalDigest);

        emit Verified(programId, publicDataHash, msg.sender);
    }
}

