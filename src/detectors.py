# Self uzayi ve dedektor kumesi.
#
# Mantik: her dedektor bir kure (merkez c, yaricap r).
#   r = (c'nin en yakin self ornege uzakligi) - r_self
# r > 0 ise dedektor hicbir normal ornegi icine almaz.
# Test ornegi herhangi bir kurenin icindeyse -> saldiri.
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


class SelfSpace:
    # Bir noktanin en yakin self ornege uzakligini bulur.
    # FCM merkezleri verilirse uzak kumeleri hic hesaplamadan atliyor (ASSUMPTIONS A2).
    # n_dist: kac tane uzaklik hesapladigimizi sayiyor, hizlanmayi olcmek icin.

    def __init__(self, X_self, centers=None, labels=None):
        self.X = np.asarray(X_self, float)
        self.n_dist = 0
        self.centers = None
        if centers is not None:
            keep = [k for k in range(len(centers)) if np.any(labels == k)]   # bos kume olmasin
            self.centers = np.asarray(centers)[keep]
            self.members = [self.X[labels == k] for k in keep]
            # kume yaricapi = merkeze en uzak uye
            self.radii = np.array([np.max(np.linalg.norm(M - v, axis=1))
                                   for M, v in zip(self.members, self.centers)])

    def nearest(self, x):
        if self.centers is None:
            # FCM yok, butun self orneklere bak
            self.n_dist += len(self.X)
            return float(np.min(np.linalg.norm(self.X - x, axis=1)))

        dc = np.linalg.norm(self.centers - x, axis=1)
        self.n_dist += len(self.centers)
        # ucgen esitsizligi: kumedeki her uye en az (dc - R) uzakta
        lower = dc - self.radii
        best = np.inf
        for k in np.argsort(lower):
            if lower[k] >= best:     # bundan sonraki kumeler daha yakin olamaz
                break
            M = self.members[k]
            self.n_dist += len(M)
            best = min(best, float(np.min(np.linalg.norm(M - x, axis=1))))
        return best

    def nearest_batch(self, Xq):
        return np.array([self.nearest(x) for x in Xq])


class DetectorSet:
    def __init__(self, dim):
        self.C = np.empty((0, dim))   # merkezler
        self.R = np.empty(0)          # yaricaplar

    def __len__(self):
        return len(self.R)

    def add(self, c, r):
        self.C = np.vstack([self.C, c])
        self.R = np.append(self.R, r)

    def covers_point(self, x):
        if len(self) == 0:
            return False
        return bool(np.any(np.linalg.norm(self.C - x, axis=1) <= self.R))

    def max_coverage_ratio(self, X, chunk=4096):
        # her x icin max(r_i / ||x - c_i||). 1 ve ustu = en az bir kurenin icinde.
        # bulanik katmanda girdi olarak kullaniyorum.
        out = np.zeros(len(X))
        if len(self) == 0:
            return out
        for s in range(0, len(X), chunk):
            D = np.fmax(cdist(X[s:s + chunk], self.C), 1e-12)
            out[s:s + chunk] = np.max(self.R[None, :] / D, axis=1)
        return out

    def covers(self, X, chunk=4096):
        if len(self) == 0:
            return np.zeros(len(X), bool)
        out = np.empty(len(X), bool)
        for s in range(0, len(X), chunk):
            D = cdist(X[s:s + chunk], self.C)
            out[s:s + chunk] = np.any(D <= self.R[None, :], axis=1)
        return out

    def predict(self, X):
        # keskin karar: 1 = saldiri, 0 = normal
        return self.covers(X).astype(int)
