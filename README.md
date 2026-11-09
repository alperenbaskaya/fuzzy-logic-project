# Fuzzy-Immune IDS: NaIM-IDS Yeniden Gerçekleştirme ve Bulanık Tespit Katmanı

Bulanık Mantık dersi dönem projesi. Referans makaledeki yapay bağışıklık tabanlı IoT saldırı tespit çerçevesinin (NaIM-IDS) tespit çekirdeği yeniden gerçekleştirilir ve üzerine bulanık çıkarım tabanlı bir tespit katmanı eklenir.

> Bu README geliştirme süresince güncellenmektedir.

## Student Information

- Öğrenci: Alperen Başkaya
- Öğrenci no: 1306220066
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

## Reproduction Experiments

### E1 — Iris (makale Tablo 7)

20 bağımsız tekrarın ortalaması. Tam tablo: [`results/tables/e1_iris_vs_paper.md`](results/tables/e1_iris_vs_paper.md).

| Self (eğitim) | Algoritma | DR % (bizim / makale) | FAR % (bizim / makale) |
|---|---|---|---|
| Versicolor (%50) | V-Detector | 94.5 / 96.5 | 10.8 / 8.6 |
| Versicolor (%50) | NaIM-IDS | 95.8 / 99.5 | 4.5 / 3.3 |
| Virginica (%50) | V-Detector | 97.8 / 96.4 | 17.9 / 10.0 |
| Virginica (%50) | NaIM-IDS | 96.7 / 99.6 | 4.5 / 7.8 |
| Versicolor (%100) | NaIM-IDS | 99.2 / 100.0 | 0.0 / 0.0 |

İlk gözlemler:

- Makaledeki ana örüntü yeniden üretildi: NaIM-IDS, V-Detector'a göre yanlış alarm oranını %50 eğitimde 2–4 kat düşürüyor.
- Tespit oranımız makaleden 1–4 puan düşük. Self küresinin (r_self) içinde kalan non-self örnekler hiçbir dedektörle yakalanamaz; makale bu parametreyi raporlamıyor.
- Dedektör sayıları makaleden iki mertebe az (ör. 5.7 / 260.3). Bizim dedektörlerimiz en büyük geçerli yarıçapla üretildiği için az sayıda büyük küre yetiyor. Makalenin dedektör yarıçapını nasıl sınırladığı belirtilmemiş.
- Eğitim süreleri farklı donanım ve uygulama dilleri nedeniyle doğrudan karşılaştırılamaz.

Varsayımlar: [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).
