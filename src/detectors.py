"""Self uzayı ve dedektör kümesi.

Gerçek değerli negatif seçimde (RNSA / V-Detector):
  - Her self örnek, yarıçapı r_self olan bir hiperküredir.
  - Her dedektör (c, r) merkezli bir hiperküredir.
  - Aday merkez c için geçerli en büyük yarıçap:  r = d_min(c, self) - r_self.
    r > 0 ise dedektör hiçbir self örneği kapsamaz ("tolerans" sağlanır).
  - Test örneği x herhangi bir dedektörün içindeyse (||x - c|| <= r) anomali sayılır.

Tolerans aşamasının maliyeti d_min(c, self) hesabıdır. Klasik yöntem bunun için
tüm self örneklerle uzaklık hesaplar. NaIM-IDS'te (makale Bölüm 3.2) self örnekler
FCM ile kümelenir ve hesap küme merkezleri + küme boyutlarıyla kısaltılır.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


class SelfSpace:
    """Bir noktanın en yakın self örneğe uzaklığını hesaplar.

    centers/labels verilmezse: kaba kuvvet (her self örnekle uzaklık).
    verilirse: FCM kümelerini kullanarak üçgen eşitsizliğiyle budama (varsayım A2).

    Küme k'nın yarıçapı R_k = max_{s in k} ||s - v_k|| ("küme boyutu").
    Herhangi bir s in k için  ||x - s|| >= ||x - v_k|| - R_k  (üçgen eşitsizliği).
    Bu alt sınır o ana kadar bulunan en iyi uzaklıktan büyükse kümenin üyelerine
    hiç bakılmaz. Sonuç kaba kuvvetle birebir aynıdır, yalnızca daha az uzaklık
    hesaplanır. `n_dist` sayacı bu tasarrufu ölçmek için tutulur.
    """

    def __init__(self, X_self: np.ndarray, centers: np.ndarray | None = None,
                 labels: np.ndarray | None = None):
        self.X = np.asarray(X_self, float)
        self.n_dist = 0
        self.centers = None
        if centers is not None:
            keep = [k for k in range(len(centers)) if np.any(labels == k)]   # boş kümeleri at
            self.centers = np.asarray(centers)[keep]
            self.members = [self.X[labels == k] for k in keep]
            self.radii = np.array([np.max(np.linalg.norm(M - v, axis=1))
                                   for M, v in zip(self.members, self.centers)])

    def nearest(self, x: np.ndarray) -> float:
        """d_min(x, self)."""
        if self.centers is None:
            self.n_dist += len(self.X)
            return float(np.min(np.linalg.norm(self.X - x, axis=1)))

        dc = np.linalg.norm(self.centers - x, axis=1)
        self.n_dist += len(self.centers)
        lower = dc - self.radii                       # her küme için alt sınır
        best = np.inf
        for k in np.argsort(lower):
            if lower[k] >= best:                      # bu ve sonraki kümeler daha iyi olamaz
                break
            M = self.members[k]
            self.n_dist += len(M)
            best = min(best, float(np.min(np.linalg.norm(M - x, axis=1))))
        return best

    def nearest_batch(self, Xq: np.ndarray) -> np.ndarray:
        return np.array([self.nearest(x) for x in Xq])


class DetectorSet:
    """Hiperküre dedektörlerin kümesi: merkezler C (k x d), yarıçaplar R (k,)."""

    def __init__(self, dim: int):
        self.C = np.empty((0, dim))
        self.R = np.empty(0)

    def __len__(self) -> int:
        return len(self.R)

    def add(self, c: np.ndarray, r: float) -> None:
        self.C = np.vstack([self.C, c])
        self.R = np.append(self.R, r)

    def covers_point(self, x: np.ndarray) -> bool:
        if len(self) == 0:
            return False
        return bool(np.any(np.linalg.norm(self.C - x, axis=1) <= self.R))

    def max_coverage_ratio(self, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """Her x için max_i ( r_i / ||x - c_i|| ). >= 1 ise x en az bir dedektörün içindedir.

        Bulanık tespit katmanında kullanılmak üzere sürekli bir 'kapsama' büyüklüğü verir.
        """
        out = np.zeros(len(X))
        if len(self) == 0:
            return out
        for s in range(0, len(X), chunk):
            D = np.fmax(cdist(X[s:s + chunk], self.C), 1e-12)
            out[s:s + chunk] = np.max(self.R[None, :] / D, axis=1)
        return out

    def covers(self, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """Her örnek için en az bir dedektörün içinde mi? (keskin karar)"""
        if len(self) == 0:
            return np.zeros(len(X), bool)
        out = np.empty(len(X), bool)
        for s in range(0, len(X), chunk):
            D = cdist(X[s:s + chunk], self.C)
            out[s:s + chunk] = np.any(D <= self.R[None, :], axis=1)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = anomali (non-self), 0 = normal (self)."""
        return self.covers(X).astype(int)
