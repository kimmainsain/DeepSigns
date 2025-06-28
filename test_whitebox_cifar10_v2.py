"""
test_whitebox_cifar10_v2.py
───────────────────────────────────────────────────────────────────────────────
DeepSigns – CIFAR-10 화이트박스(N-bit) 워터마크 **추출 전용** 스크립트

● 논문 단계 대응
    Alg. 3  Step I   :  키 서브셋(특정 클래스 데이터) 선택
            Step II  :  선택 층 활성값 f(x) 수집
            Step III :  μ′ (평균) 계산
            Step IV  :  투영행렬 A·μ′ → Sigmoid → 0/1
            Step V   :  BER = |b′ ⊕ b| / N 계산

───────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchsummary import summary     # (선택) 모델 구조 출력용

from models.resnet import ResNet18
from utils import *                  # subsample / get_activations / BER 등

# ── 상위 경로(프로젝트 루트) 추가 ─────────────────────────────────── #
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))


def run(args):
    # ───────────────────────────────────────────────────────── #
    # 0. 환경 및 CIFAR-10 데이터 로드                            #
    # ───────────────────────────────────────────────────────── #
    device = torch.device('cuda')              # CPU 사용 시 'cpu'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    # ───────────────────────────────────────────────────────── #
    # 1. 논문 Figure 1 ③에서 저장된 비트열 b 로드                 #
    #    b ∈ {0,1}^{N × 10}   (N = embed_bits)                   #
    # ───────────────────────────────────────────────────────── #
    b = np.load('logs/whitebox/marked/b.npy')       # 원본 워터마크

    # ───────────────────────────────────────────────────────── #
    # 2. 모델 파라미터 로드 (※ 현재는 ‘ummarked’ 예시)           #
    #    • 실제 검증하려면  marked 모델 경로로 교체 필요          #
    # ───────────────────────────────────────────────────────── #
    sd_path = 'logs/whitebox/ummarked/resnet18.pth'  # ← 언마크드 모델
    marked_model = ResNet18().to(device)
    # summary(marked_model, input_size=(3, 32, 32))  # 구조 확인용
    marked_model.load_state_dict(torch.load(sd_path))

    # ───────────────────────────────────────────────────────── #
    # 3. (Alg. 3 Step I)  키용 mini-set :                       #
    #    · target_class 데이터 중 50 % 무작 추출                #
    # ───────────────────────────────────────────────────────── #
    x_train_subset_loader = subsample_training_data(
        trainset, args.target_class)

    # ───────────────────────────────────────────────────────── #
    # 4. (Step II) 선택 층(ResNet18 feat) 활성값 수집            #
    #    반환 shape ≈ [M, 512]                                  #
    # ───────────────────────────────────────────────────────── #
    marked_activations = get_activations(
        marked_model, x_train_subset_loader)
    print("Collected activations of marked dense layer")

    # ───────────────────────────────────────────────────────── #
    # 5. (Step III–IV) A·μ′ → Bit 복원                           #
    #    • 투영행렬 A 는 train_whitebox 단계에서 저장됨           #
    # ───────────────────────────────────────────────────────── #
    A = np.load('logs/whitebox/marked/projection_matrix.npy')
    print('Projection matrix A shape:', A.shape)
    decoded_WM = extract_WM_from_activations(marked_activations, A)

    # ───────────────────────────────────────────────────────── #
    # 6. (Step V) BER 계산                                      #
    #    BER = 0 → 완전 일치, 0.5 수준 → 랜덤                    #
    # ───────────────────────────────────────────────────────── #
    BER = compute_BER(decoded_WM, b[:, args.target_class])
    print(f"BER for class {args.target_class} = {BER}")

    # ─ 참고:  여기서는 ‘ummarked’ 모델이므로 BER ≈ 0.5 예상 ─── #


# ────────────────────────────────────────────────────────────── #
#  CLI 인자 정의 (주요 하이퍼파라미터 표기)                       #
# ────────────────────────────────────────────────────────────── #
def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes in data')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='(unused here) Placeholder for symmetry')
    parser.add_argument('--epochs', type=int, default=50,
                        help='(unused) kept for CLI consistency')
    parser.add_argument('--scale', type=float, default=0.01,
                        help='λ₁ in training script (not used here)')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='λ₂ in training script (not used here)')
    parser.add_argument('--target_dense_idx', type=int, default=2,
                        help='(unused) Dense-layer index placeholder')
    parser.add_argument('--embed_bits', type=int, default=16,
                        help='N : number of watermark bits per class')
    parser.add_argument('--target_class', type=int, default=0,
                        help='Class used for extraction/verification')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
