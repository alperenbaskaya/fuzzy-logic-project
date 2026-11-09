# Fuzzy C-Means (FCM) - makale Bolum 3.2, Denk. 15-19
# Self (normal) ornekleri kumelemek icin kullaniyorum. Kutuphane kullanmadim,
# formulleri makaleden direkt yazdim.
#
# X: n x d veri, V: c x d merkezler, U: n x c uyelik matrisi (satir toplami 1)
# m: bulaniklik katsayisi (m > 1). m 1'e yaklastikca K-means gibi davranir.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class FCMResult:
    centers: np.ndarray
    U: np.ndarray
    J: float
    n_iter: int
    m: float

    @property
    def labels(self):
        # her ornegi en yuksek uyelige sahip oldugu kumeye ata
        return np.argmax(self.U, axis=1)


def update_memberships(X, V, m):
    # Denk. 16
    D = np.fmax(cdist(X, V), 1e-12)   # sifira bolmesin
    ratio = (D[:, :, None] / D[:, None, :]) ** (2.0 / (m - 1.0))
    return 1.0 / ratio.sum(axis=2)


def update_centers(X, U, m):
    # Denk. 17
    Um = U ** m
    return (Um.T @ X) / Um.sum(axis=0)[:, None]


def objective(X, V, U, m):
    # Denk. 15: J = sum u^m * ||x - v||^2
    return float(np.sum((U ** m) * cdist(X, V) ** 2))


def fcm(X, c, m=2.0, eps=1e-5, max_iter=300, rng=None):
    if m <= 1.0:
        raise ValueError("m > 1 olmali")
    rng = rng or np.random.default_rng()
    c = min(c, len(X))

    # rastgele uyelik matrisiyle basla
    U = rng.random((len(X), c))
    U /= U.sum(axis=1, keepdims=True)

    J_prev = np.inf
    for it in range(1, max_iter + 1):
        V = update_centers(X, U, m)
        U = update_memberships(X, V, m)
        J = objective(X, V, U, m)
        if abs(J_prev - J) < eps:      # Denk. 18 durma kosulu
            break
        J_prev = J
    return FCMResult(V, U, J, it, m)


def com_sep_index(X, res):
    # Denk. 19: S = Com / Sep, kucuk olan daha iyi
    # makale iki kume icin yazmis, ben c kumeye genellestirdim (ASSUMPTIONS A3)
    V, U, m = res.centers, res.U, res.m
    if len(V) < 2:
        return np.inf
    D2 = cdist(X, V) ** 2
    com = float(np.sum(np.sum((U ** m) * D2, axis=0) / U.sum(axis=0)))
    dv = cdist(V, V)
    sep = float(np.min(dv[np.triu_indices(len(V), k=1)]))
    return com / max(sep, 1e-12)


def select_fcm(X, c_values=range(2, 11), m_values=(1.5, 2.0, 2.3, 2.5, 3.0), rng=None):
    # Com/Sep'i en kucuk yapan (c, m) ikilisini grid search ile bul
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
