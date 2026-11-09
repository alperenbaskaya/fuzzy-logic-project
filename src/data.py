# Veri yukleme ve on isleme (makale Bolum 4).
# - tekrar eden / eksik satirlari sil
# - kategorikleri LabelEncoder ile sayiya cevir
# - min-max ile [0, 1] araligina olcekle
# Etiket: 0 = self (normal), 1 = non-self (saldiri)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list
    name: str


def minmax(X, eps=1e-12):
    lo, hi = X.min(axis=0), X.max(axis=0)
    return (X - lo) / (hi - lo + eps)


def clean_frame(df):
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df.reset_index(drop=True)


def load_iris_self(self_class):
    # secilen sinif self, diger ikisi saldiri gibi (makale 4.3.1)
    iris = load_iris()
    X = minmax(iris.data.astype(float))
    y = (iris.target != self_class).astype(int)
    return Dataset(X, y, list(iris.feature_names), f"iris-{iris.target_names[self_class]}")


def load_skin(path=None):
    # UCI Skin Segmentation: B G R etiket. deri (1) = self, deri degil (2) = saldiri
    # makale 245.057 kayit diyor yani tekrarlari silmemis, ben de silmiyorum
    path = path or RAW_DIR / "Skin_NonSkin.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["B", "G", "R", "label"])
    X = minmax(df[["B", "G", "R"]].to_numpy(float))
    y = (df["label"].to_numpy() == 2).astype(int)
    return Dataset(X, y, ["B", "G", "R"], "skin")


# makale Tablo 3'teki BoT-IoT ozellikleri
BOTIOT_PAPER_FEATURES = ["stime", "flgs", "sport", "dport", "seq", "spkts"]
# zaman damgasi / id / port gibi ozellikler -> veri sizintisi riski
BOTIOT_LEAKY = ["pkSeqID", "stime", "ltime", "seq", "saddr", "daddr", "sport", "dport"]


def load_botiot(path=None, features=None, max_rows=20000, seed=0):
    # BoT-IoT cok dengesiz (normal trafik cok az), o yuzden dengeli alt orneklem aliyorum
    path = path or RAW_DIR / "botiot"
    files = sorted(Path(path).glob("*.csv")) if Path(path).is_dir() else [Path(path)]
    if not files:
        raise FileNotFoundError(f"BoT-IoT CSV bulunamadi: {path} (data/README.md)")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    features = features or BOTIOT_PAPER_FEATURES
    df = clean_frame(df[features + ["attack"]])

    rng = np.random.default_rng(seed)
    if max_rows is not None:
        normal = df[df["attack"] == 0]
        attack = df[df["attack"] == 1]
        k = min(len(normal), max_rows // 2)
        normal = normal.iloc[rng.permutation(len(normal))[:k]]
        attack = attack.iloc[rng.permutation(len(attack))[: max_rows - k]]
        df = pd.concat([normal, attack], ignore_index=True)

    X = minmax(df[features].to_numpy(float))
    y = df["attack"].to_numpy(int)
    return Dataset(X, y, features, "bot-iot")


def split_self_train(ds, train_frac, rng):
    # self ve saldiri orneklerinin train_frac kadarini egitim icin sec
    out = []
    for label in (0, 1):
        idx = np.flatnonzero(ds.y == label)
        k = max(1, int(round(train_frac * len(idx))))
        out.append(ds.X[rng.permutation(idx)[:k]])
    return out[0], out[1]


def split_train_test(ds, train_frac, rng):
    # bulanik deneyler icin: egitimde kullanilmayan orneklerde test edecegim
    tr_idx = []
    for label in (0, 1):
        idx = rng.permutation(np.flatnonzero(ds.y == label))
        tr_idx.append(idx[: max(1, int(round(train_frac * len(idx))))])
    tr = np.concatenate(tr_idx)
    te = np.setdiff1d(np.arange(len(ds.y)), tr)
    X_self = ds.X[tr][ds.y[tr] == 0]
    X_non = ds.X[tr][ds.y[tr] == 1]
    return X_self, X_non, ds.X[te], ds.y[te]
