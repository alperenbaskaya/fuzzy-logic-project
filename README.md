# Fuzzy-Immune IDS: NaIM-IDS Yeniden Gerçekleştirme ve Bulanık Tespit Katmanı

Bulanık Mantık dersi dönem projesi. Referans makaledeki yapay bağışıklık tabanlı IoT saldırı tespit çerçevesinin (NaIM-IDS) tespit çekirdeği yeniden gerçekleştirilir ve üzerine bulanık çıkarım tabanlı bir tespit katmanı eklenir.

> Bu README geliştirme süresince güncellenmektedir.

## Student Information

- Öğrenci: Alperen Başkaya
- Öğrenci no: _doldurulacak_
- Ders: Bulanık Mantık, 2026 Güz

## Original Paper

M. Ma, G. Yang, J. He, and W. Fang, "An adaptive detection framework based on artificial immune for IoT intrusion detection system," *Applied Soft Computing*, vol. 166, Art. no. 112152, 2024, doi: [10.1016/j.asoc.2024.112152](https://doi.org/10.1016/j.asoc.2024.112152).

Makalenin yazarları kod yayımlamamıştır. Bu repodaki tüm kod makaledeki tanımlardan sıfırdan yazılmıştır. Makalenin vermediği ayrıntılar için yapılan varsayımlar [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) dosyasında listelenir.

## Project Structure

```
src/
  data.py        veri yükleme ve ön işleme
  fcm.py         Fuzzy C-Means (makale Denk. 15-19), Com/Sep indeksi
  detectors.py   self uzayı (FCM budamalı en yakın-self) ve dedektör kümesi
  vdetector.py   V-Detector baseline [Ji & Dasgupta, 2004]
  rfo.py         Red Fox Optimization (sınır dedektörleri)
  ga.py          ikili kodlu genetik algoritma (sınır dışı dedektörler)
  naim.py        NaIM-IDS uyarlanabilir dedektör üretimi (makale Bölüm 3.2-3.3)
  metrics.py     DR, FAR, accuracy, precision, F1
experiments/baseline/   yeniden üretim deneyleri (E1-E3)
experiments/proposed/   özgün katkı deneyleri (E4-E8)
results/tables/, results/figures/
```

## Installation

```bash
git clone <repo-url>
cd fuzzy-logic-project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

```bash
# E1: Iris yeniden üretim deneyi (makale Tablo 7)
python -m experiments.baseline.run_iris
```

Sonuçlar `results/tables/` altına yazılır.
