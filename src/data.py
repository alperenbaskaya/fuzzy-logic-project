"""Veri yükleme ve ön işleme.

Makale Bölüm 4'teki ön işleme adımları uygulanır:
tekrar eden / eksik satırların silinmesi, kategorik özelliklerin LabelEncoder ile
kodlanması ve min-max normalizasyonu ([0, 1] aralığı).

Negatif seçim algoritmalarında örnekler iki gruba ayrılır:
  self     -> normal trafik (etiket 0)
  non-self -> saldırı / anomali (etiket 1)
"""
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
    X: np.ndarray          # (n, d), [0, 1] aralığında
    y: np.ndarray          # (n,), 0 = self (normal), 1 = non-self (anomali)
    feature_names: list[str]
    name: str


def minmax(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-max normalizasyonu. eps sıfıra bölmeyi önler (makale Bölüm 4)."""
    lo, hi = X.min(axis=0), X.max(axis=0)
    return (X - lo) / (hi - lo + eps)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tekrar eden satırları, inf ve eksik değerleri temizler; kategorikleri kodlar."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df.reset_index(drop=True)


def load_iris_self(self_class: int) -> Dataset:
    """Iris: seçilen sınıf self, diğer iki sınıf non-self (makale Bölüm 4.3.1)."""
    iris = load_iris()
    X = minmax(iris.data.astype(float))
    y = (iris.target != self_class).astype(int)
    name = f"iris-{iris.target_names[self_class]}"
    return Dataset(X, y, list(iris.feature_names), name)


def load_skin(path: Path | None = None) -> Dataset:
    """UCI Skin Segmentation: deri (sınıf 1) self, deri olmayan (sınıf 2) non-self.

    Dosya biçimi: B G R etiket (sekme ayraçlı, başlıksız).
    Not: makale yalnızca tekrarları silmekten bahseder; Skin setinde çok sayıda
    tekrar eden piksel vardır. Varsayılan olarak tekrarlar silinmez, çünkü makalenin
    raporladığı kayıt sayısı (245.057) ham dosyanın tamamıdır.
    """
    path = path or RAW_DIR / "Skin_NonSkin.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["B", "G", "R", "label"])
    X = minmax(df[["B", "G", "R"]].to_numpy(float))
    y = (df["label"].to_numpy() == 2).astype(int)
    return Dataset(X, y, ["B", "G", "R"], "skin")


# Makale Tablo 3'te BoT-IoT için seçilen 6 özellik.
BOTIOT_PAPER_FEATURES = ["stime", "flgs", "sport", "dport", "seq", "spkts"]
# Tanımlayıcı / zaman damgası niteliğindeki, veri sızıntısı riski taşıyan özellikler.
BOTIOT_LEAKY = ["pkSeqID", "stime", "ltime", "seq", "saddr", "daddr", "sport", "dport"]


def load_botiot(
    path: Path | None = None,
    features: list[str] | None = None,
    max_rows: int | None = 20000,
    seed: int = 0,
) -> Dataset:
    """BoT-IoT CSV dosyalarını yükler.

    `attack` sütunu 0/1 etikettir (0 = normal). BoT-IoT çok dengesizdir (normal
    trafik %0.01 civarı); bu yüzden dengeli bir alt örneklem alınır.
    features=None ise makalenin Tablo 3 listesi kullanılır.
    """
    path = path or RAW_DIR / "botiot"
    files = sorted(Path(path).glob("*.csv")) if Path(path).is_dir() else [Path(path)]
    if not files:
        raise FileNotFoundError(f"BoT-IoT CSV bulunamadı: {path}. data/README.md'ye bakın.")
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


def split_self_train(
    ds: Dataset, train_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Eğitim için self ve non-self örneklerin `train_frac` kadarını rastgele seçer.

    Döndürür: (self eğitim örnekleri, non-self eğitim örnekleri).
    Test her zaman veri setinin tamamı üzerinde yapılır (makale Tablo 7 düzeni:
    %100 eğitimde FAR = 0 olması, testin tüm veri üzerinde yapıldığını gösterir).
    """
    out = []
    for label in (0, 1):
        idx = np.flatnonzero(ds.y == label)
        k = max(1, int(round(train_frac * len(idx))))
        out.append(ds.X[rng.permutation(idx)[:k]])
    return out[0], out[1]
