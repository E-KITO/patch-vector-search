"""L2 正規化コサインの前段に挟む線形変換(異方性除去)。

`experiments/0024` で、UNI 埋め込みの L2 正規化コサインには 2 つの弱点があり
(異方性: 平均ベクトル μ のノルムが大きく共分散が非等方 / hubness)、self-retrieval の
n_hits_ratio ランキングでは **ZCA 白色化** または **all-but-the-top (ABTT)** が
median best_rank を約半分に下げると判明した(髄外造血 1000→35、granular 160→11 等)。

ここで扱うのはどちらも単位ベクトルへのアフィン変換 `y = A(x − μ)`(+ 事後 L2 再正規化):
  - whiten (ZCA): A = (Σ + εI)^{-1/2}    全主方向を等分散化
  - abtt(d):      A = I − U_d U_d^T       上位 d 主成分を除去 (Mu & Viswanath 2018)

`make_faiss_pretransform` がこれを `faiss.LinearTransform` + `faiss.NormalizationTransform`
に変換するので、索引にベイクすれば `faiss.read_index` した全消費者(クエリ側も含む)が
自動でこの空間で検索する(`lib.faiss_index.build_faiss_index` の `pretransform` 引数)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _fit_pca(X: np.ndarray, fit_sample: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L2 正規化済み X から (mu, eigenvalues_desc, eigvecs_rows_desc) を返す。"""
    rng = np.random.default_rng(seed)
    if fit_sample and len(X) > fit_sample:
        X = X[rng.choice(len(X), size=fit_sample, replace=False)]
    mu = X.mean(axis=0)
    Xc = X - mu
    # SVD: Xc = U S Vt。共分散 Σ = Vt^T diag(S^2/n) Vt。
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    eig = (s.astype(np.float64) ** 2) / len(Xc)
    return mu.astype(np.float32), eig, vt.astype(np.float32)


def fit_whiten(X: np.ndarray, *, eps: float = 1e-3, fit_sample: int = 200_000,
               seed: int = 42) -> dict:
    """ZCA 白色化 `A = V (Λ + εI)^{-1/2} V^T`。X は L2 正規化済みを想定。"""
    mu, eig, vt = _fit_pca(X, fit_sample, seed)
    scale = 1.0 / np.sqrt(eig + eps)
    A = (vt.T * scale) @ vt                     # (1024, 1024)
    return {"kind": "whiten", "eps": float(eps), "mu": mu,
            "A": A.astype(np.float32), "explained_var_ratio": (eig / eig.sum())[:16]}


def fit_whiten_shrink(X: np.ndarray, *, alpha: float = 0.5, eps: float = 1e-3,
                      fit_sample: int = 200_000, seed: int = 42) -> dict:
    """収縮白色化: 固有値を平均へ向けて内挿してから逆平方根を取る。

        λ_i^shrunk = α·λ_i + (1−α)·mean(λ)
        A = V diag(1/√(λ_i^shrunk + ε)) V^T

    α=1 は full ZCA(fit_whiten と一致)、α=0 は全固有値が等しくなり A ∝ I
    (後段の L2 再正規化で baseline と等価)。`experiments/0025` で full 白色化が
    小固有値方向を 1/√λ 倍に増幅して atlas クエリのドメインギャップノイズまで
    増幅したため、その増幅を α で抑える(`experiments/0026`)。
    """
    mu, eig, vt = _fit_pca(X, fit_sample, seed)
    lam_mean = float(eig.mean())
    eig_shrunk = alpha * eig + (1.0 - alpha) * lam_mean
    scale = 1.0 / np.sqrt(eig_shrunk + eps)
    A = (vt.T * scale) @ vt
    return {"kind": f"whiten_shrink_a{alpha}", "alpha": float(alpha), "eps": float(eps),
            "mu": mu, "A": A.astype(np.float32),
            "explained_var_ratio": (eig / eig.sum())[:16]}


def fit_abtt(X: np.ndarray, *, d: int = 4, fit_sample: int = 200_000, seed: int = 42) -> dict:
    """all-but-the-top: `A = I − U_d U_d^T`(上位 d 主成分を除去)。"""
    mu, eig, vt = _fit_pca(X, fit_sample, seed)
    U = vt[:d]                                   # (d, 1024)
    A = np.eye(len(mu), dtype=np.float32) - U.T @ U
    return {"kind": f"abtt{d}", "d": int(d), "mu": mu,
            "A": A.astype(np.float32), "explained_var_ratio": (eig / eig.sum())[:16]}


def apply_transform(params: dict, V: np.ndarray) -> np.ndarray:
    """numpy 側で `normalize(A (V − μ))` を適用(索引を使わない検証用)。"""
    Y = (V.astype(np.float32) - params["mu"]) @ params["A"].T
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Y / n


def save_transform(params: dict, path: str | Path) -> None:
    np.savez(path, **{k: v for k, v in params.items() if isinstance(v, (np.ndarray, int, float, str))})


def load_transform(path: str | Path) -> dict:
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    for k in ("eps", "d", "alpha"):
        if k in out:
            out[k] = out[k].item()
    return out


def make_faiss_pretransform(params: dict):
    """params から faiss の変換チェーン [LinearTransform, NormalizationTransform] を作る。

    LinearTransform は `y = A x + b`。ここでは `y = A (x − μ) = A x − A μ` なので
    b = −A μ。続く NormalizationTransform(L2) で単位ベクトルに戻す。
    索引にこの 2 つを prepend すれば、add / search 双方で自動適用される。
    """
    import faiss

    A = np.ascontiguousarray(params["A"], dtype=np.float32)
    mu = np.ascontiguousarray(params["mu"], dtype=np.float32)
    d = A.shape[0]
    b = -(A @ mu)

    lt = faiss.LinearTransform(d, d, True)
    faiss.copy_array_to_vector(np.ascontiguousarray(A.ravel(), dtype=np.float32), lt.A)
    faiss.copy_array_to_vector(np.ascontiguousarray(b, dtype=np.float32), lt.b)
    lt.is_trained = True
    # is_orthonormal は false のまま(reverse_transform は使わない。forward の
    # apply には A / b / is_trained だけあれば足りる)。

    nt = faiss.NormalizationTransform(d, 2.0)
    return lt, nt
