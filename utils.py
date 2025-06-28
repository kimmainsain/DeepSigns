"""
utils.py
───────────────────────────────────────────────────────────────────────────────
DeepSigns – 공통 유틸리티 모음

함수 분류
────────
1. 블랙박스(출력층 1-bit) 키 생성·검증
   • key_generation        – Algorithm 2 전반 (키 생성 + fine-tune)
   • fine_tune             – 키 정확도 확보용 소규모 재학습
   • test                  – 모델 예측 / 맞음·틀림 인덱스 산출
   • compute_mismatch_threshold – 논문 식 (4) 구현

2. 화이트박스(N-bit) 워터마킹
   • train_whitebox        – Figure 1 ③, 식 (1)(3) 구현
   • get_activations       – 중간층 활성값 수집 (Alg. 3 Step II)
   • extract_WM_from_activations – μ·A 투영 → 비트 추출 (Alg. 3 Step IV)
   • compute_BER           – BER 계산 (Alg. 3 Step V)

3. 보조 함수
   • subsample_training_data – 특정 클래스 데이터의 50 % 무작 추출
───────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import comb
from torch.utils.data import ConcatDataset, DataLoader, Subset


# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 1. 블랙박스 키 생성 (Algorithm 2)                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
def key_generation(marked_model,
                   optimizer,
                   original_data,
                   desired_key_len,
                   img_size=32,
                   num_classes=10,
                   embed_epoch=20):
    """
    키(Trigger Set) 자동 생성 + 모델 fine-tune

    1) 난수 이미지 40 × |K| 생성  ── (논문 기본: 20 ×; 여기선 40 ×로 넉넉히)
    2) fine_tune() 으로 모델을 재학습해 ‘일부 샘플만 정답으로 바뀌게’ 만듦
    3) 재학습 前 틀리고(err) 재학습 後 맞은(correct) 교집합만 키로 채택
       – usable_key_len < desired_key_len  ▸  루프 재시도
       – 충분하면 K개 저장 + 워터마크드 모델 체크포인트 저장
    """
    key_len   = 40 * desired_key_len            # ★ 논문값(20×)보다 2배 크게 설정
    batch_sz  = 1024
    key_flag  = True

    while key_flag:
        # ── (1) 무작위 이미지·라벨 생성 ───────────────────── #
        x_rand = torch.randn(key_len, 3, img_size, img_size)
        y_rand = torch.randint(num_classes, size=[key_len])

        rand_loader = DataLoader(
            torch.utils.data.TensorDataset(x_rand, y_rand),
            batch_size=batch_sz, shuffle=False, num_workers=2
        )

        # ── err_idx : 재학습 前 틀린 샘플 인덱스 ─────────────── #
        _, err_idx, _ = test(marked_model, rand_loader)

        # ── (2) 원본 데이터 + 랜덤 샘플 결합 → fine-tune ─────── #
        retrain_data = ConcatDataset([original_data,
                                      torch.utils.data.TensorDataset(x_rand, y_rand)])
        retrain_loader = DataLoader(retrain_data,
                                    batch_size=batch_sz, shuffle=False)
        fine_tune(marked_model, optimizer, retrain_loader, embed_epoch)

        # ── correct_idx : 재학습 後 맞은 샘플 인덱스 ─────────── #
        _, _, correct_idx = test(marked_model, rand_loader)

        # ── (3) err ∩ correct 교집합만 사용 ────────────────── #
        key_idx   = np.intersect1d(err_idx, correct_idx)
        x_keys    = x_rand[key_idx.astype(int)]
        y_keys    = y_rand[key_idx.astype(int)]
        usable_k  = x_keys.shape[0]
        print('usable key len is:', usable_k)

        if usable_k < desired_key_len:
            print(f'Desired key length {desired_key_len} unmet → retry.')
            key_flag = True
        else:
            key_flag = False
            # 앞쪽 K개만 사용
            x_keys   = x_keys[:desired_key_len]
            y_keys   = y_keys[:desired_key_len]

            # ── 결과 저장 (키 & 워터마크드 모델) ─────────────── #
            np.save(f'logs/blackbox/keyRandomImage_keyLength{desired_key_len}.npy', x_keys)
            np.savetxt(f'logs/blackbox/keyRandomLabel_keyLength{desired_key_len}.txt',
                       y_keys, fmt='%i', delimiter=',')
            torch.save(marked_model.state_dict(),
                       'logs/blackbox/marked/resnet18.pth')
            print('WM key generation finished. Save watermarked model.')

    return x_keys, y_keys


def fine_tune(model, optimizer, dataloader, epochs):
    """
    소규모 fine-tuning (Alg. 2 Step 3)
    – 손실: CrossEntropy **단독** (λ₁·λ₂ 보정 없음)
    – 목적: 새로 삽입된 random 키를 ‘정답’으로 만들기
    """
    model.train()
    device    = next(model.parameters()).device
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(data)
            if isinstance(logits, tuple):          # (logits, feat) 형태 대응
                logits = logits[0]
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()


def test(model, dataloader):
    """
    모델 예측 + 맞음·틀림 인덱스 반환 (Alg. 4 Step 2-3 공용)
    반환값:
        loss_meter  – 총 CE 손실
        err_idx     – 틀린 인덱스 리스트
        correct_idx – 맞은 인덱스 리스트
    """
    model.eval()
    device     = next(model.parameters()).device
    loss_meter = 0
    err_idx, correct_idx = [], []
    offset = 0                                            # 누적 인덱스 오프셋

    with torch.no_grad():
        for data, target, *_ in dataloader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss_meter += F.cross_entropy(logits, target, reduction='sum').item()

            pred = logits.argmax(dim=1, keepdim=False)
            correct_idx += (pred.eq(target)).nonzero(as_tuple=True)[0].cpu().add(offset).tolist()
            err_idx     += (pred.ne(target)).nonzero(as_tuple=True)[0].cpu().add(offset).tolist()
            offset      += data.size(0)

    return loss_meter, err_idx, correct_idx


def compute_mismatch_threshold(c=10, kp=50, p=0.05):
    """
    논문 식 (4) 구현  
    ▸ θ = 최소 i  (누적확률 > p)  
      s.t. P(N_k > i | O)

    - p_err = 1 − 1/C  (임의 모델이 키 하나를 ‘틀릴’ 확률)
    - kp    = |K|      (키 길이)
    """
    prob_sum = 0
    p_err    = 1.0 - 1.0 / c
    theta    = 0
    for i in range(kp + 1):                              # ≤ kp 포함
        cur = comb(kp, i) * (p_err ** i) * ((1 - p_err) ** (kp - i))
        prob_sum += cur
        if prob_sum > p:
            theta = i
            break
    return theta


# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 2. 화이트박스 워터마킹                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝
def extract_WM_from_activations(activs, A):
    """
    Alg. 3 Step III-IV  
    : μ′ 평균 → A·μ′ → Sigmoid → Hard 0/1
    """
    mu     = np.mean(activs, axis=0).reshape(-1, 1)      # μ′
    proj   = A @ mu                                      # A·μ′
    probs  = 1.0 / (1 + np.exp(-proj))                   # Sigmoid
    bits   = (probs > 0.5).astype(int)                   # Hard threshold 0.5
    return bits


def get_activations(model, input_loader):
    """
    선택 층(ResNet18 구현상: model forward 반환 tuple 중 feat)을 모아 numpy 배열 반환
    """
    acts   = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for data, *_ in input_loader:
            feat = model(data.to(device))[1]             # (logits, feat)
            acts.append(feat.cpu().numpy())
    return np.concatenate(acts, axis=0)


def compute_BER(decoded_bits, gt_bits):
    """
    Bit Error Rate = (|b' ⊕ b|) / N
    """
    diff = np.abs(decoded_bits - gt_bits.reshape(-1, 1))
    return diff.sum() / diff.size


# ╔══════════════════════════════════════════════════════════════════════╗
# ║ 3. 보조 함수                                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
def subsample_training_data(dataset, target_class):
    """
    특정 클래스(target_class) 데이터 중 50 %를 샘플링 → DataLoader 반환
    (Alg. 3 Step I – ‘입력 키’용 subset)
    """
    idx_all   = (np.array(dataset.targets) == target_class).nonzero()[0]
    pick_len  = int(0.5 * len(idx_all))
    picked    = np.random.choice(idx_all, size=pick_len, replace=False)
    subset    = Subset(dataset, picked)
    return DataLoader(subset, batch_size=128, shuffle=False)


def train_whitebox(model,
                   optimizer,
                   dataloader,
                   b, centers,
                   args,
                   is_attack=False,
                   save_path=None):
    """
    Figure 1 ③ + 식 (1)(3) :  
    손실 = CE + λ₁(loss1+loss2+loss3) + λ₂·loss4

    - loss1 : Intra-class MSE (feat ↔ μ<sub>class</sub>)
    - loss2 : Inter-class 분리 (cosine × distance 방식, *논문 식과 유사*)
    - loss3 : Center 벡터 정규화 ‖μ‖² ≈ 1
    - loss4 : Watermark CE  (Sigmoid(A·μ<sub>K</sub>), b<sub>K</sub>)
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    device    = next(model.parameters()).device

    # ── 투영행렬 A 생성 & 저장 (랜덤 → 재현성 없음) ───────────── #
    x_value = np.random.randn(args.embed_bits, 512)
    if save_path:
        np.save(save_path, x_value)
    x_value = torch.tensor(x_value, dtype=torch.float32).to(device)
    b       = torch.tensor(b, dtype=torch.float32).to(device)

    for ep in range(args.epochs):
        print(ep)
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            logits, feat = model(data)                 # ResNet18 → (logits, feat)
            loss  = criterion(logits, target)          # 기본 CE 손실

            # ── loss1 : Intra-class compactness ───────────── #
            centers_batch = centers[target]
            loss1 = F.mse_loss(feat, centers_batch, reduction='sum') / 2

            # ── loss2 : Inter-class separability (cosine·distance) ─ #
            c_batch = centers_batch.unsqueeze(1)       # [B,1,512]
            c_all   = centers.unsqueeze(0)             # [1,10,512]
            pair_d  = ((c_batch - c_all) ** 2).sum(-1)
            arg     = torch.topk(-pair_d, k=2)[1][:, -1]  # 두 번째로 가까운 클래스
            near_c  = centers[arg]
            dists   = ((centers_batch - near_c) ** 2).sum(-1)
            cosines = (centers_batch * near_c).sum(-1)
            loss2   = (cosines * dists - dists).mean()

            # ── loss3 : Center normalization ‖μ‖² ≈ 1 ─────────── #
            loss3   = (1 - centers.pow(2).sum(1)).abs().sum()

            # ── loss4 : Watermark embedding for target_class ─── #
            loss4   = 0
            idx_k   = (target == args.target_class).nonzero(as_tuple=True)[0]
            if idx_k.numel() > 0:
                acts_k   = centers_batch[idx_k]
                mu_k     = acts_k.mean(0)
                logits_w = torch.sigmoid(x_value.matmul(mu_k))
                b_k      = b[:, args.target_class]
                loss4    = F.binary_cross_entropy(logits_w, b_k, reduction='sum')

            # ── 최종 손실 & 업데이트 ─────────────────────────── #
            total_loss = (loss +
                          args.scale * (loss1 + loss2 + loss3) +
                          args.gamma * loss4)
            total_loss.backward()
            optimizer.step()
