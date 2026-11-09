# Ikili kodlu genetik algoritma - sinir disi dedektorler icin (makale Bolum 3.3.2)
# secim: rulet, caprazlama: uniform, mutasyon: fast mutation, + elitizm
# Her boyut `bits` bit ile [lb, ub] araligina kodlaniyor.
from __future__ import annotations

import numpy as np


def encode(x, lb, ub, bits):
    levels = 2 ** bits - 1
    q = np.round((x - lb) / np.fmax(ub - lb, 1e-12) * levels).astype(int)
    q = np.clip(q, 0, levels)
    return ((q[:, None] >> np.arange(bits - 1, -1, -1)) & 1).ravel().astype(np.uint8)


def decode(g, lb, ub, bits):
    levels = 2 ** bits - 1
    q = g.reshape(-1, bits) @ (1 << np.arange(bits - 1, -1, -1))
    return lb + q / levels * (ub - lb)


def _fast_mutation(g, rng, beta=1.5):
    # cevrilecek bit sayisi k ~ k^(-beta): genelde az bit, arada bir cok bit
    L = len(g)
    ks = np.arange(1, max(2, L // 2) + 1)
    p = ks ** (-beta)
    k = rng.choice(ks, p=p / p.sum())
    flip = rng.random(L) < k / L
    out = g.copy()
    out[flip] ^= 1
    return out


def ga_maximize(fitness, init_pop, lb, ub, n_gen, rng, bits=12, pc=0.9):
    G = np.array([encode(np.clip(x, lb, ub), lb, ub, bits) for x in init_pop])
    n = len(G)
    fit = np.array([fitness(decode(g, lb, ub, bits)) for g in G])

    for _ in range(n_gen):
        elite = G[np.argmax(fit)].copy()

        # rulet secimi (negatif fitness olmasin diye kaydiriyorum)
        w = fit - fit.min() + 1e-9
        parents = G[rng.choice(n, size=n, p=w / w.sum())]

        children = parents.copy()
        for i in range(0, n - 1, 2):
            if rng.random() < pc:
                mask = rng.random(G.shape[1]) < 0.5
                children[i, mask], children[i + 1, mask] = parents[i + 1, mask], parents[i, mask]

        children = np.array([_fast_mutation(c, rng) for c in children])
        children[0] = elite
        G = children
        fit = np.array([fitness(decode(g, lb, ub, bits)) for g in G])

    i = int(np.argmax(fit))
    return decode(G[i], lb, ub, bits), float(fit[i])
