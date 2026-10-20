"""Fuzzy C-Means (FCM) kümeleme -- sıfırdan, kütüphanesiz.

Makale Bölüm 3.2: self (normal) örnekler FCM ile kümelenir. Aday dedektör her self
örnekle tek tek karşılaştırılmak yerine küme merkezi ve küme boyutuyla
karşılaştırılır; böylece tolerans aşamasındaki uzaklık hesabı azalır.

Notasyon (makale ile aynı):
  X   : n x d self örnekleri
  V   : c x d küme merkezleri
  U   : n x c üyelik matrisi, u_ij in [0, 1], her satırın toplamı 1
  m   : bulanıklık katsayısı (m > 1). m -> 1 iken FCM, K-means'e yaklaşır.

Amaç fonksiyonu (Denk. 15):
  J_m = sum_i sum_j u_ij^m * ||x_i - v_j||^2
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class FCMResult:
    centers: np.ndarray    # V, (c, d)
    U: np.ndarray          # üyelik matrisi, (n, c)
    J: float               # son amaç fonksiyonu değeri
    n_iter: int
    m: float

    @property
    def labels(self) -> np.ndarray:
        """Her örneğin en yüksek üyelik derecesine sahip olduğu küme (defuzzify)."""
        return np.argmax(self.U, axis=1)


def update_memberships(X: np.ndarray, V: np.ndarray, m: float) -> np.ndarray:
    """Denk. 16: u_ij = 1 / sum_k ( ||x_i - v_j|| / ||x_i - v_k|| )^(2/(m-1))."""
    D = np.fmax(cdist(X, V), 1e-12)                 # sıfıra bölmeyi önle
    ratio = (D[:, :, None] / D[:, None, :]) ** (2.0 / (m - 1.0))
    return 1.0 / ratio.sum(axis=2)


def update_centers(X: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
    """Denk. 17: v_j = sum_i u_ij^m x_i / sum_i u_ij^m."""
    Um = U ** m
    return (Um.T @ X) / Um.sum(axis=0)[:, None]


def objective(X: np.ndarray, V: np.ndarray, U: np.ndarray, m: float) -> float:
    """Denk. 15."""
    return float(np.sum((U ** m) * cdist(X, V) ** 2))


def fcm(
    X: np.ndarray,
    c: int,
    m: float = 2.0,
    eps: float = 1e-5,
    max_iter: int = 300,
    rng: np.random.Generator | None = None,
) -> FCMResult:
    """FCM algoritması (makale Bölüm 3.2, adım 1-4).

    1) U rastgele başlatılır (satır toplamları 1).
    2) Merkezler güncellenir (Denk. 17).
    3) Üyelikler güncellenir (Denk. 16).
    4) |J_{t+1} - J_t| < eps olunca durulur (Denk. 18).
    """
    if m <= 1.0:
        raise ValueError("m > 1 olmalı")
    rng = rng or np.random.default_rng()
    c = min(c, len(X))
    U = rng.random((len(X), c))
    U /= U.sum(axis=1, keepdims=True)

    J_prev = np.inf
    for it in range(1, max_iter + 1):
        V = update_centers(X, U, m)
        U = update_memberships(X, V, m)
        J = objective(X, V, U, m)
        if abs(J_prev - J) < eps:
            break
        J_prev = J
    return FCMResult(V, U, J, it, m)


def com_sep_index(X: np.ndarray, res: FCMResult) -> float:
    """Kümeleme kalite indeksi S = Com / Sep (Denk. 19). Küçük olan daha iyi.

    Com: her kümenin üyelikle ağırlıklandırılmış iç-küme sıkılığı, küme
         büyüklüğüyle (sum_i u_ij) normalize edilir.
    Sep: küme merkezleri arasındaki ayrıklık.
    Makale Denk. 19'u iki küme için yazar; burada c küme için genelleştirilir ve
    Sep, merkezler arasındaki en küçük uzaklık olarak alınır (varsayım A3).
    """
    V, U, m = res.centers, res.U, res.m
    if len(V) < 2:
        return np.inf
    D2 = cdist(X, V) ** 2
    com = float(np.sum(np.sum((U ** m) * D2, axis=0) / U.sum(axis=0)))
    dv = cdist(V, V)
    sep = float(np.min(dv[np.triu_indices(len(V), k=1)]))
    return com / max(sep, 1e-12)


def select_fcm(
    X: np.ndarray,
    c_values=range(2, 11),
    m_values=(1.5, 2.0, 2.3, 2.5, 3.0),
    rng: np.random.Generator | None = None,
) -> tuple[FCMResult, list[dict]]:
    """Com/Sep indeksini en küçükleyen (c, m) çiftini ızgara aramasıyla seçer."""
    rng = rng or np.random.default_rng()
    best, best_s, log = None, np.inf, []
    for c in c_values:
        if c >= len(X):
            break
        for m in m_values:
            res = fcm(X, c, m, rng=rng)
            s = com_sep_index(X, res)
            log.append({"c": c, "m": m, "S": s, "J": res.J})
            if s < best_s:
                best, best_s = res, s
    return best, log
