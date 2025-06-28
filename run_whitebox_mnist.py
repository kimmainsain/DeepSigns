"""
run_whitebox_mnist.py
───────────────────────────────────────────────────────────────────────────────
DeepSigns – MNIST 화이트박스(N-bit) 워터마크 **삽입 + 추출** 실험 스크립트

● 논문 대응 흐름
    Fig. 1 ③     :   b(비트열) 생성 ➊  +  MLP + centers(μ) 초기화 ➋
                     train_whitebox() 호출 → 식 (1)(3) 학습 ➌
    테스트        :   원본 정확도 확인
    Alg. 3       :   키 서브셋 → μ′ → A·μ′ → BER 계산 ➍

───────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchsummary import summary          # (선택) 네트워크 요약
from models.mlp import MLP                # 두 층 512-FC MLP + feat 반환
from utils import *                       # train_whitebox, BER 등 유틸

# ── 프로젝트 루트 경로 추가 ───────────────────────────────── #
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))


def run(args):
    # ───────────────────────────────────────────────────────── #
    # 0. 환경 설정 + MNIST 데이터 로드                           #
    # ───────────────────────────────────────────────────────── #
    device = torch.device('cpu')  # CUDA 사용 시 'cuda'
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(
        root="./data/", transform=transform, train=True, download=True)
    testset = torchvision.datasets.MNIST(
        root="./data/", transform=transform, train=False, download=True)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=512, shuffle=False)

    # ───────────────────────────────────────────────────────── #
    # 1. (Fig. 1 ③-①)  워터마크 비트열 b 생성                   #
    #    b ∈ {0,1}^{embed_bits × 10} (N-bit × 클래스)           #
    # ───────────────────────────────────────────────────────── #
    b = np.random.randint(2, size=(args.embed_bits, args.n_classes))
    # (저장은 필요 시 utils 내부에서 진행하므로 여기선 메모리 보관)

    # ───────────────────────────────────────────────────────── #
    # 2. (Fig. 1 ③-②)  모델 & 센터 μ 초기화                    #
    # ───────────────────────────────────────────────────────── #
    model = MLP().to(device)                        # forward → (logits, feat)
    centers = torch.nn.Parameter(
        torch.rand(args.n_classes, 512, device=device), requires_grad=True)

    optimizer = torch.optim.RMSprop(
        [{'params': model.parameters()},
         {'params': centers}],
        lr=args.lr, alpha=0.9, eps=1e-8, weight_decay=1e-3)

    # ───────────────────────────────────────────────────────── #
    # 3. (Fig. 1 ③-③)  워터마크 삽입 학습                       #
    #    train_whitebox() :  CE + λ₁(loss1+2+3) + λ₂·loss4      #
    #    투영행렬 A 는 logs/whitebox/mlp/marked/projection_matrix.npy 에 저장
    # ───────────────────────────────────────────────────────── #
    train_whitebox(model, optimizer, trainloader,
                   b=b,
                   centers=centers,
                   args=args,
                   save_path='./logs/whitebox/mlp/marked/projection_matrix.npy')

    # ───────────────────────────────────────────────────────── #
    # 4. 모델 성능 확인 (정확도·Loss)                           #
    # ───────────────────────────────────────────────────────── #
    model.eval()
    loss_meter, acc_meter = 0, 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            logits, _ = model(data)
            loss_meter += F.cross_entropy(logits, target, reduction='sum').item()
            acc_meter += logits.argmax(1).eq(target).sum().item()
    print('Test loss    :', loss_meter / len(testloader.dataset))
    print('Test accuracy:', acc_meter  / len(testloader.dataset))

    # 모델 파라미터(워터마크 포함) 저장
    sd_path = 'logs/whitebox/mlp/marked/mlp.pth'
    torch.save(model.state_dict(), sd_path)

    # ───────────────────────────────────────────────────────── #
    # 5. 워터마크 추출·검증 (Alg. 3)                            #
    # ───────────────────────────────────────────────────────── #
    # 5-1. 마크드 모델 로드
    marked_model = MLP().to(device)
    # summary(marked_model, input_size=(1, 28, 28))  # 네트워크 구조 보기
    marked_model.load_state_dict(torch.load(sd_path))

    # 5-2. (Step I)  target_class(=digit) 데이터 절반 서브샘플
    subset_loader = subsample_training_data(trainset, args.target_class)

    # 5-3. (Step II) feat 활성값 수집
    activations = get_activations(marked_model, subset_loader)
    print("Collected activations of first WM-carrying dense layer")

    # 5-4. (Step III–IV) A·μ′ → 비트 복원
    A = np.load('logs/whitebox/mlp/marked/projection_matrix.npy')
    decoded_bits = extract_WM_from_activations(activations, A)

    # 5-5. (Step V) BER 계산
    BER = compute_BER(decoded_bits, b[:, args.target_class])
    print(f"BER for class {args.target_class} = {BER}")


# ────────────────────────────────────────────────────────────── #
#  CLI 인수 정의                                                  #
# ────────────────────────────────────────────────────────────── #
def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # 일반 파라미터
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes in MNIST (10 digits)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for WM training')

    # 워터마킹 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=1,
                        help='embed_epoch : train_whitebox epochs')
    parser.add_argument('--scale', type=float, default=0.01,
                        help='λ₁ (loss1+2+3 weight)')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='λ₂ (watermark CE weight)')
    parser.add_argument('--target_dense_idx', type=int, default=2,
                        help='(unused for this MLP) Dense layer index placeholder')
    parser.add_argument('--embed_bits', type=int, default=16,
                        help='N : number of watermark bits per class')
    parser.add_argument('--target_class', type=int, default=0,
                        help='Digit (0-9) chosen for extraction demo')
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
