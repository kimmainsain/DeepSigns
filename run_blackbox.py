"""
run_blackbox.py
───────────────────────────────────────────────────────────────────────────────
DeepSigns – 블랙박스 워터마킹 데모 스크립트

● 흐름 요약
    1) CIFAR-10 학습 데이터를 내려받고 전처리
    2) ***워터마킹되지 않은(unmarked)*** ResNet-18 파라미터 로드
       └ 모델을 key_generation()에 넘겨 **키(Trigger Set) 삽입**
    3) 생성된 키 셋으로 **동일 모델**(unmarked) 응답을 받아
       맞은 개수(acc) → mismatch 임계값 θ 와 비교
       → 논문 Alg. 4와 동일한 검증 단계

● 주의
    • 현재 파일 경로는 ‘logs/blackbox/ummarked/...’ → 워터마킹이
      실제로 삽입되지 않은 상태에서 검증을 수행함  
      (논문 실험을 재현하려면 marked 모델 경로로 교체해야 함)
    • mismatch = key_len − acc_meter 가 아니라
      acc_meter 자체를 θ와 직접 비교하는 구현임
      (= ‘맞은 개수 < θ’ 조건). 원 논문 식(4)에서는
      “틀린 개수 < θ”로 정의되므로 해석에 유의.
───────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from utils import *                       # key_generation, θ계산 등
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))
from models.resnet import ResNet18        # ResNet-18 백본


# ────────────────────────────────────────────────────────────── #
#  핵심 실행부 – 전체 파이프라인                                    #
# ────────────────────────────────────────────────────────────── #
def run(args):
    # ── 0. 디바이스 설정 (기본 CPU) ───────────────────────────── #
    device = torch.device('cpu')

    # ── 1. CIFAR-10 학습 데이터 (전처리 포함) ────────────────── #
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),          # 데이터 증강
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), # 채널 평균·표준편차
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    # ── 2. (Un)marked 모델 로드 & 옵티마이저 ────────────────── #
    #     ※ 경로명에 'ummarked' → 워터마크가 삽입되지 않은 상태
    model = ResNet18().to(device)
    model.load_state_dict(torch.load('logs/blackbox/ummarked/resnet18.pth'))
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=0.9, weight_decay=0.0, nesterov=True
    )

    # ── 3. 키(Trigger Set) 생성 + 미세조정(Fig. 1 ②③, Alg. 2) ── #
    #     • key_generation():  
    #       - 무작위 이미지 40×key_len 개 생성  
    #       - 모델을 fine-tune 하여 일부 샘플만 ‘정답 예측’하도록 만듦  
    #       - 그중 err∩correct 인덱스 K개를 최종 키로 저장
    x_key, y_key = key_generation(model, optimizer,
                                  original_data=trainset,
                                  desired_key_len=args.key_len,
                                  img_size=32,
                                  num_classes=args.n_classes,
                                  embed_epoch=args.epochs)

    key_data   = torch.utils.data.TensorDataset(x_key, y_key)
    key_loader = DataLoader(key_data,
                            batch_size=128,
                            shuffle=False,
                            num_workers=2)

    # ── 4. 블랙박스 검증 (Alg. 4 Step 2-3) ───────────────────── #
    #     *여기서는* 재학습 이전 unmarked 모델을 그대로 사용
    marked_model = ResNet18().to(device)
    marked_model.load_state_dict(torch.load('logs/blackbox/ummarked/resnet18.pth'))

    acc_meter = 0  # 키 셋에 대해 ‘맞게’ 예측한 개수
    with torch.no_grad():
        for load in key_loader:
            data, target = load[:2]
            data   = data.to(device)
            target = target.to(device)
            pred, _ = marked_model(data)          # (logits, feat)
            pred = pred.max(1, keepdim=True)[1]   # 라벨 추출
            acc_meter += pred.eq(target.view_as(pred)).sum().item()

    # ── 5. mismatch 임계치 θ 계산 (논문 식 4) ────────────────── #
    theta = compute_mismatch_threshold(
        c=args.n_classes,
        kp=args.key_len,
        p=args.th
    )

    # ── 6. 결과 출력 ───────────────────────────────────────── #
    print('probability threshold p is ', args.th)
    print('Mismatch threshold θ   : ', theta)
    print('“맞은” 개수 (acc_meter)         : ', acc_meter)
    print("Authentication (acc_meter < θ) :", acc_meter < theta)
    # ※ acc_meter 는 ‘맞음’ 개수.  
    #    원 논문에서는 mismatch(틀림) 개수를 θ와 비교.


# ────────────────────────────────────────────────────────────── #
#  CLI 인수 정의                                                   #
# ────────────────────────────────────────────────────────────── #
def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_classes', type=int, default=10,
                        help='데이터셋 클래스 개수 (C)')
    parser.add_argument('--key_len', type=int, default=20,
                        help='원하는 키 길이 |K|')
    parser.add_argument('--lr',        type=float, default=0.00005,
                        help='fine-tune learning rate')
    parser.add_argument('--th',        type=float, default=0.1,
                        help='식 (4)의 허용 확률 p')
    parser.add_argument('--epochs',    type=int,   default=2,
                        help='key embed fine-tune epoch 수')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
