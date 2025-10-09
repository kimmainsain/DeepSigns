# common_wm.py
import os, numpy as np, torch, hashlib, ecdsa
from torchvision import datasets, transforms

# ---- 1) 비트 유틸 ----
def bits_from_pk(pk_uncompressed_65B: bytes, T: int) -> np.ndarray:
    """SHA256(pk) → 앞 T비트 (0/1) ndarray (T,)"""
    h = hashlib.sha256(pk_uncompressed_65B).digest()
    bits256 = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
    return bits256[:T].astype(np.uint8)

def ber(x_bits: np.ndarray, y_bits: np.ndarray) -> float:
    assert x_bits.shape == y_bits.shape
    return float(np.mean(x_bits ^ y_bits))

def majority_vote(bits_TC: np.ndarray) -> np.ndarray:
    """(T, C) → (T,) 클래스 복제된 비트의 다수결 축약"""
    T, C = bits_TC.shape
    s = bits_TC.sum(axis=1)
    return (s >= (C/2)).astype(np.uint8)

# ---- 2) b' 추출(DeepSigns-화이트박스) ----
def extract_bprime_unique(model, A_TxD: np.ndarray, device="cuda", 
                          batch_size=512) -> np.ndarray:
    """
    모델이 (logits, feat) 반환한다고 가정.
    A: shape (T, D). D는 feat 차원(예: 512).
    절차: MNIST 테스트셋→클래스별 feat 평균 μ_c → s_{:,c} = A @ μ_c → b'_{:,c} = (s>0)
         마지막에 majority_vote로 (T,)로 축약.
    """
    transform = transforms.ToTensor()
    ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval(); model.to(device)
    # C=10, D=feat_dim
    with torch.no_grad():
        feats = [[] for _ in range(10)]
        for x, y in loader:
            x = x.to(device)
            logits, feat = model(x)     # feat: (B, D)
            for i, yi in enumerate(y.tolist()):
                feats[yi].append(feat[i].detach().cpu().numpy())
    mus = np.stack([np.mean(np.vstack(f), axis=0) for f in feats], axis=1)  # (D, C)
    s = A_TxD @ mus                                # (T, C)
    bprime_TC = (s > 0).astype(np.uint8)           # (T, C)
    return majority_vote(bprime_TC)                # (T,)
