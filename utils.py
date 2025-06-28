import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import comb
from torch.utils.data import ConcatDataset, DataLoader, Subset


def key_generation(marked_model, optimizer, original_data, desired_key_len, img_size=32, num_classes=10,
                   embed_epoch=20):
    """
    블랙박스 키 생성 함수 (Alg. 2 단계 2-4)
    1️⃣ 무작위 이미지 40 × K 생성
    2️⃣ fine_tune() 으로 모델 재학습(키 주입)
    3️⃣ "재학습 전에는 틀리고, 후에는 맞는" 샘플 교집합(err_idx ∩ correct_idx)만 추출 → K개 저장
    """
    key_len = 40 * desired_key_len  # 논문의 K₀ = 20 × K와 동일 아이디어
    batch_size = 1024
    key_gen_flag = 1
    while key_gen_flag:
        # 1️⃣ 무작위 이미지 40 × K 생성
        x_retrain_rand = torch.randn(key_len, 3, img_size, img_size)
        y_retrain_rand_vec = torch.randint(num_classes, size=[key_len])
        retrain_rand_data = torch.utils.data.TensorDataset(x_retrain_rand, y_retrain_rand_vec)
        retrain_rand_loader = DataLoader(retrain_rand_data,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=2)
        
        # 재학습 전 테스트 - 틀린 샘플 인덱스 추출
        _, err_idx, _ = test(marked_model, retrain_rand_loader)
        
        # 2️⃣ fine_tune() 으로 모델 재학습(키 주입)
        retrain_data = ConcatDataset([original_data, retrain_rand_data])
        retrain_loader = DataLoader(retrain_data,
                                    batch_size=batch_size,
                                    shuffle=False)
        fine_tune(marked_model, optimizer, retrain_loader, embed_epoch)
        
        # 재학습 후 테스트 - 맞는 샘플 인덱스 추출
        _, _, correct_idx = test(marked_model, retrain_rand_loader)
        
        # 3️⃣ "재학습 전에는 틀리고, 후에는 맞는" 샘플 교집합(err_idx ∩ correct_idx)만 추출
        selected_key_idx = np.intersect1d(err_idx, correct_idx)
        selected_keys = x_retrain_rand[np.array(selected_key_idx).astype(int), :]
        selected_keys_labels = y_retrain_rand_vec[np.array(selected_key_idx).astype(int)]
        usable_key_len = selected_keys.shape[0]
        print('usable key len is: ', usable_key_len)
        
        # 사용 길이 부족 → while 루프 재시도
        if usable_key_len < desired_key_len:
            key_gen_flag = 1
            print(' Desire key length is {}, Need longer key, skip this test. '.format(desired_key_len))
        else:
            key_gen_flag = 0
            selected_keys = selected_keys[0:desired_key_len, :]
            selected_keys_labels = selected_keys_labels[0:desired_key_len]
            np.save('logs/blackbox/keyRandomImage' + '_keyLength' + str(desired_key_len) + '.npy', selected_keys)
            np.savetxt('logs/blackbox/keyRandomLabel' + '_keyLength' + str() + '.txt', selected_keys_labels, fmt='%i',
                       delimiter=',')
            torch.save(marked_model.state_dict(), f'logs/blackbox/marked/resnet18.pth')
            print('WM key generation finished. Save watermarked model. ')
    return selected_keys, selected_keys_labels


def fine_tune(model, optimizer, dataloader, epochs):
    """
    블랙박스용 키가 정확히 분류되도록 소규모 fine-tuning (Alg. 2 Step 3)
    λ₁·λ₂는 건드리지 않음 → 모델 정확도 거의 불변
    """
    model.train()
    device = next(model.parameters()).device
    criterion = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        for d, t in dataloader:
            d = d.to(device)
            t = t.to(device)
            optimizer.zero_grad()
            pred = model(d)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = criterion(pred, t)
            loss.backward()
            optimizer.step()


def test(model, dataloader):
    """
    블랙박스 검증 함수 (Alg. 4 Steps 2-3)
    예측, 손실, 맞음/틀림 인덱스 반환
    err_idx, correct_idx 를 그대로 키 생성에도 재사용
    """
    model.eval()
    device = next(model.parameters()).device
    loss_meter = 0
    err_idx = []
    correct_idx = []
    runcount = 0
    with torch.no_grad():
        for load in dataloader:
            data, target = load[:2]
            data = data.to(device)
            target = target.to(device)
            pred = model(data)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
            pred = pred.max(1, keepdim=True)[1]
            correct_idx += (pred.view_as(target) == target).nonzero(as_tuple=True)[0].cpu() + runcount
            err_idx += (pred.view_as(target) != target).nonzero(as_tuple=True)[0].cpu() + runcount
            runcount += data.size(0)
    return loss_meter, err_idx, correct_idx


def compute_mismatch_threshold(c=10, kp=50, p=0.05):
    """
    식 (4) 구현: 클래스수 C·키길이 K·허용 p(=5 %) → 임계치 θ 계산
    누락: ceil·> vs ≥ 경계 조건 검토 필요
    """
    prob_sum = 0
    p_err = 1 - 1.0 / c  # 각 클래스에서 틀릴 확률
    theta = 0
    for i in range(kp):
        cur_prob = comb(kp, i, exact=False) * np.power(p_err, i) * np.power(1 - p_err, kp - i)
        prob_sum = prob_sum + cur_prob
        if prob_sum > p:
            theta = i
            break
    return theta


def extract_WM_from_activations(activs, A):
    """
    워터마크 추출 함수 (Alg. 3)
    선택 층 출력 → μ′ 평균 → X·μ′ → Sigmoid→0/1 → 추출 비트
    하드 threshold 0.5 고정 ⇒ 논문과 동일
    """
    activ_classK = activs
    activ_centerK = np.mean(activ_classK, axis=0)  # μ′ 평균
    activ_centerK = np.reshape(activ_centerK, (-1, 1))
    X_Ck = np.dot(A, activ_centerK)  # X·μ′ (x_value가 논문의 투영행렬 A)
    X_Ck_sigmoid = 1 / (1 + np.exp(-X_Ck))  # Sigmoid
    decode_wmark = (X_Ck_sigmoid > 0.5) * 1  # 하드 threshold 0.5 고정
    return decode_wmark


def get_activations(model, input_loader):
    """
    워터마크 추출을 위한 활성화값 획득 (Alg. 3)
    선택 층 출력 획득
    """
    activations = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for d, t in input_loader:
            d = d.to(device)
            _, feat = model(d)  # 특징 추출
            activations.extend(feat.detach().cpu().numpy())
    return np.stack(activations, 0)


def compute_BER(decode_wmark, b_classK):
    """
    비트 오류율 계산 (Alg. 3 Step 5)
    원본 b 와 추출 b′ 차이 비율 - 논문의 BER
    """
    b_classK = np.reshape(b_classK, (-1, 1))
    diff = np.abs(decode_wmark - b_classK)
    BER = np.sum(diff) / b_classK.size
    return BER


def subsample_training_data(dataset, target_class):
    """
    데이터 유틸 함수 (Alg. 3 Step I - 키용 mini subset)
    특정 클래스 데이터 중 50 % 무작 추출
    np 방식으로 가속
    """
    # train_indices = []
    # for i in range(len(dataset.targets)):
    #     if dataset.targets[i] == target_class:
    #         train_indices.append(i)
    train_indices = (np.array(dataset.targets) == target_class).nonzero()[0]  # .reshape(-1)
    subsample_len = int(np.floor(0.5 * len(train_indices)))  # 50% 무작 추출
    subset_idx = np.random.randint(len(train_indices), size=subsample_len)
    train_subset = Subset(dataset, train_indices[subset_idx])
    dataloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=False)
    return dataloader


def train_whitebox(model, optimizer, dataloader, b, centers, args, is_attack=False, save_path=None):
    """
    화이트박스 워터마킹 학습 함수 (Fig. 1 ③, 식 (1)(3) + GMM 가정)
    중심 μ·투영행렬 X_value·비트 b에 대해
    loss = CE + scale·(loss1+2+3) + γ·loss4
    loss1 = Intra/Inter compact,
    loss4 = 투영 μ→Sigmoid→b 와의 CE
    """
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    
    # x_value가 논문의 투영행렬 A
    x_value = np.random.randn(args.embed_bits, 512)
    np.save(save_path, x_value)
    x_value = torch.tensor(x_value, dtype=torch.float32).to(device)
    b = torch.tensor(b).to(device)
    
    for ep in range(args.epochs):
        print(ep)
        for d, t in dataloader:
            d = d.to(device)
            t = t.to(device)
            optimizer.zero_grad()
            pred = model(d)
            feat = None
            if isinstance(pred, tuple):
                pred, feat = pred
            else:
                feat = pred
            
            # 기본 분류 손실
            loss = criterion(pred, t)
            
            # centers(μ) 를 trainable buffer로 두고 MSE + cosine으로 정렬
            centers_batch = torch.gather(centers, 0, t.unsqueeze(1).repeat(1, feat.shape[1]))
            
            # loss1: Intra-class compactness (MSE)
            loss1 = F.mse_loss(feat, centers_batch, reduction='sum') / 2
            
            # loss2: Inter-class separability
            centers_batch_reshape = torch.unsqueeze(centers_batch, 1)
            centers_reshape = torch.unsqueeze(centers, 0)
            # shape 맞추기: centers_batch_reshape [B,1,512], centers_reshape [1,10,512]
            if centers_batch_reshape.shape[2] != centers_reshape.shape[2]:
                min_dim = min(centers_batch_reshape.shape[2], centers_reshape.shape[2])
                centers_batch_reshape = centers_batch_reshape[..., :min_dim]
                centers_reshape = centers_reshape[..., :min_dim]
            pairwise_dists = (centers_batch_reshape - centers_reshape) ** 2
            pairwise_dists = torch.sum(pairwise_dists, dim=-1)
            arg = torch.topk(-pairwise_dists, k=2)[1]
            arg = arg[:, -1]
            closest_cents = torch.gather(centers, 0, arg.unsqueeze(1).repeat(1, feat.shape[1]))
            dists = torch.sum((centers_batch - closest_cents) ** 2, dim=-1)
            cosines = torch.mul(closest_cents, centers_batch)
            cosines = torch.sum(cosines, dim=-1)
            loss2 = (cosines * dists - dists).mean()
            
            # loss3: Center normalization
            loss3 = (1 - torch.sum(centers ** 2, dim=1)).abs().sum()
            
            # loss4: Watermark embedding loss
            loss4 = 0
            embed_center_idx = args.target_class
            idx_classK = (t == embed_center_idx).nonzero(as_tuple=True)
            if len(idx_classK[0]) >= 1:
                idx_classK = idx_classK[0]
                activ_classK = torch.gather(centers_batch, 0,
                                            idx_classK.unsqueeze(1).repeat(1, feat.shape[1]))
                center_classK = torch.mean(activ_classK, dim=0)
                Xc = torch.matmul(x_value, center_classK)  # 투영 μ→Sigmoid→b
                bk = b[:, embed_center_idx]
                bk_float = bk.float()
                probs = torch.sigmoid(Xc)
                entropy_tensor = F.binary_cross_entropy(target=bk_float, input=probs, reduce=False)
                loss4 += entropy_tensor.sum()
            
            # args.scale→λ₁, args.gamma→λ₂ 대응
            (loss + args.scale * (loss1 + loss2 + loss3) + args.gamma * loss4).backward()
            optimizer.step()
