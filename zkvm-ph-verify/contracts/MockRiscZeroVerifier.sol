// contracts/MockRiscZeroVerifier.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./MockVerifierRouter.sol";

/// @notice 예전 파일명을 임포트하는 코드와 호환을 위한 래퍼.
/// 기존 코드가 기대하는 이름(MockRiscZeroVerifier)을 그대로 제공.
contract MockRiscZeroVerifier is MockVerifierRouter {}

