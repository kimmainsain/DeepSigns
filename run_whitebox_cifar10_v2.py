"""
run_whitebox_cifar10_v2.py
───────────────────────────────────────────────────────────────────────────────
DeepSigns – CIFAR-10 화이트박스(N-bit) 워터마킹 실험 스크립트

● 수행 단계 (논문 기준)
    Fig. 1 ③     :   WM 비트열(b) 생성 + 모델/센터 초기화
    train_whitebox:   식 (1)(3) → μ·A·b 학습
    Alg. 3        :   키 서브셋 → 활성값 μ′ → A·μ′ → BER 계산

───────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torchvision
from torchvision import transforms
from torchsummary import summary
import numpy as np

from models.resnet import ResNet18
from utils import *

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))


def run(args):
    # ───────────────────────────────────────────────────────── #
    # 0. 환경 세팅 / CIFAR-10 로드                               #
    # ───────────────────────────────────────────────────────── #
    device = torch.device('cpu')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),             # Data aug.
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # ───────────────────────────────────────────────────────── #
    # 1. 워터마킹 설정 : 비트열 b, 투영행렬 A 저장 위치           #
    # ───────────────────────────────────────────────────────── #
    # b  ∈ {0,1}^{embed_bits × 10}  (논문: N-bit × 클래스)
    b = np.random.randint(2, size=(args.embed_bits, args.n_classes))
    np.save('logs/whitebox/resnet18/marked/b.npy', b)     # 재현용 기록

    # ───────────────────────────────────────────────────────── #
    # 2. 모델 & trainable 센터(μ) 초기화                           #
    # ───────────────────────────────────────────────────────── #
    model = ResNet18().to(device)                         # logits, feat 반환
    # centers(μ) 를 trainable buffer로 두고 MSE + cosine으로 정렬
    centers = torch.nn.Parameter(torch.rand(args.n_classes, 512).to(device), requires_grad=True)
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
        {'params': centers}
    ], lr=args.lr,
        momentum=0.9, weight_decay=5e-4)

    # ───────────────────────────────────────────────────────── #
    # 3. 화이트박스 워터마킹 학습 (Fig. 1 ③, train_whitebox)     #
    #    - loss = CE + λ₁(loss1+2+3) + λ₂·loss4                 #
    # ───────────────────────────────────────────────────────── #
    # train_whitebox() 함수 호출 - 화이트박스 워터마킹 학습
    train_whitebox(model, optimizer, trainloader, b, centers, args, save_path='./logs/whitebox/resnet18/marked/projection_matrix.npy')

    # ───────────────────────────────────────────────────────── #
    # 4. 모델 성능 테스트 (정확도·손실)                           #
    # ───────────────────────────────────────────────────────── #
    # 모델 성능 평가
    model.eval()
    loss_meter = 0
    acc_meter = 0
    with torch.no_grad():
        for d, t in testloader:
            data = d.to(device)
            target = t.to(device)
            pred, _ = model(data)
            loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
            pred = pred.max(1, keepdim=True)[1]
            acc_meter += pred.eq(target.view_as(pred)).sum().item()
    print('Test loss:', loss_meter / len(testloader.dataset))
    print('Test accuracy:', acc_meter / len(testloader.dataset))

    # 모델 파라미터 저장 (워터마크 포함)
    sd_path = 'logs/whitebox/resnet18/marked/resnet18.pth'
    torch.save(model.state_dict(), sd_path)

    # ───────────────────────────────────────────────────────── #
    # 5. 화이트박스 워터마크 추출 & BER 계산 (Alg. 3)            #
    # ───────────────────────────────────────────────────────── #
    # ---- 화이트박스 워터마크 검증 (Alg. 3) ---- #
    marked_model = ResNet18().to(device)
    # summary(marked_model, input_size=(1, 28, 28))
    marked_model.load_state_dict(torch.load(sd_path))
    
    # (Step I)   : 키용 mini subset (특정 클래스 50%)
    # subsample_training_data() - 키용 mini subset 생성
    x_train_subset_loader = subsample_training_data(trainset, args.target_class)
    
    # (Step II)  : 선택 층 활성값 수집
    # get_activations() - 선택 층 출력 획득
    marked_activations = get_activations(marked_model, x_train_subset_loader)
    print("Get activations of marked FC layer")
    # choose the activations from first wmarked dense layer
    marked_FC_activations = marked_activations
    
    # (Step III-IV) : μ′ → 투영행렬 A 로 비트 복원
    # extract_WM_from_activations() - 워터마크 추출
    A = np.load('logs/whitebox/resnet18/marked/projection_matrix.npy')
    print('A = ', A)
    decoded_WM = extract_WM_from_activations(marked_FC_activations, A)
    
    # (Step V)   : BER 계산
    BER = compute_BER(decoded_WM, b[:, args.target_class])
    print("BER in class {} is {}: ".format(args.target_class, BER))

    # ──────────────────────────────── 〈추가 로그/덤프 구간〉 ──────────────────────────────── #
    import hashlib

    # 1) A·b 내용∙형상 확인
    print("A shape :", A.shape)
    print("first 3 rows of A\n", A[:3])
    print("b shape :", b.shape)
    print("b[:, {}] =".format(args.target_class), b[:, args.target_class])

    # 2) 휴먼-리더블 TXT 저장
    np.savetxt('A_matrix.txt', A, fmt='%.6f', delimiter=',')
    np.savetxt('b_bits.txt', b, fmt='%d', delimiter='')

    # 3) SHA-256 해시 출력
    def sha256_of_npy(path):
        data = np.load(path)
        return hashlib.sha256(data.tobytes()).hexdigest()

    hash_A = sha256_of_npy('logs/whitebox/resnet18/marked/projection_matrix.npy')
    hash_b = sha256_of_npy('logs/whitebox/resnet18/marked/b.npy')

    print("SHA256(A) :", hash_A)
    print("SHA256(b) :", hash_b)
    # ─────────────────────────────────────────────────────────────────────────────────────── #


# ────────────────────────────────────────────────────────────── #
#  CLI 인자 정의                                                  #
# ────────────────────────────────────────────────────────────── #
def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # 데이터·네트워크 일반 파라미터
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes in data')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

    # 워터마킹 하이퍼파라미터 (논문 λ₁·λ₂·N 등)
    parser.add_argument('--epochs', default=50, type=int, help='embed_epoch')
    parser.add_argument('--scale', default=0.01, type=float, help='for loss1')  # args.scale→λ₁
    parser.add_argument('--gamma', default=0.01, type=float, help='for loss2')  # args.gamma→λ₂
    parser.add_argument('--target_dense_idx', default=2, type=int, help='target layer to carry WM')
    parser.add_argument('--embed_bits', default=16, type=int)
    parser.add_argument('--target_class', default=0, type=int)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
