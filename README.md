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
  fuzzy_detect.py  bulanık tespit katmanı (MF, Mamdani, Sugeno, durulaştırma)
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

# E4/E5/E7: keskin vs bulanık tespit + gürültü testi, sonra grafikler
python -m experiments.proposed.run_fuzzy_iris
python -m experiments.proposed.plot_figures
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

## Proposed Improvement: Bulanık Tespit Katmanı

Makalede karar keskindir: örnek bir dedektör küresinin içindeyse saldırı, değilse normal kabul edilir. Bu çalışmada eğitilmiş dedektörlerin üzerine elle yazılmış bir bulanık çıkarım sistemi (FIS) eklendi. Sistem iki girdiden 0–1 aralığında bir anomali derecesi üretir.

- **g1, kapsama oranı:** max r_i / ‖x − c_i‖. Değer 1 veya üzerindeyse örnek bir kürenin içindedir. Dilsel terimler: Dışarıda, Sınırda, İçeride.
- **g2, self uzaklığı:** en yakın self örneğe uzaklık / d_ref. Dilsel terimler: Yakın, Orta, Uzak.
- **Çıktı:** Normal, Şüpheli, Saldırı. 9 kural kullanılır, VE işlemi için min alınır. Mamdani (min kırpma, max birleştirme) ve 0. derece Sugeno birlikte uygulandı.
- **Karşılaştırılan varyantlar:** üçgen ve Gauss üyelik fonksiyonları; centroid, bisector ve MOM durulaştırma.

![Üyelik fonksiyonları](results/figures/fis_membership.png)
![Kontrol yüzeyi](results/figures/fis_surface.png)

## Comparative Results (Iris, ayrık test, 3 sınıf × 20 tekrar)

Tam tablolar: [`results/tables/e4_e7_fuzzy_iris.md`](results/tables/e4_e7_fuzzy_iris.md).

| Yöntem | DR % | FAR % | F1 % | AUC % |
|---|---|---|---|---|
| Keskin (makale) | 94.8 | 5.0 | 96.1 | 94.9 |
| Kapsama skoru, FIS yok | – | – | – | 98.5 |
| FIS üçgen Mamdani centroid | 83.1 | 0.9 | 89.6 | 96.7 |
| FIS Gauss Sugeno | 82.9 | 0.6 | 89.6 | 98.5 |

![Gürültü testi](results/figures/e7_noise.png)

**Bulgular**

1. Bulanık katman yanlış alarm oranını 5.0'dan 0.9'a indirir (Wilcoxon p < 0.001). Buna karşılık tespit oranı 94.8'den 83.1'e düşer ve F1 6.5 puan azalır. Kontrol yüzeyi bunun nedenini gösterir: kural tablosu, bir kürenin içinde olsa bile self'e çok yakın duran örnekleri Normal sayar. Sistem bu haliyle temkinli bir dedektördür.
2. FIS sürekli bir skor ürettiği için ROC/AUC hesaplanabilir; keskin karar ise tek bir ROC noktası verir. FIS'in AUC değeri keskin karardan yüksektir (96.7–98.5 ve 94.9). Ancak FIS'siz ham kapsama oranı da 98.5 AUC'ye ulaşır. Dolayısıyla sıralama kalitesindeki kazanç bulanık kurallardan değil, sürekli skordan gelir.
3. Eşik eğitim verisinin %30'luk bir doğrulama kısmında ayarlandığında da F1 keskin karara yetişmez. Iris'te doğrulama kümesi yalnızca 7–8 normal örnek içerdiğinden seçilen eşik oynaktır.
4. Gürültü altında keskin kararın F1 değeri 96.1'den 91.3'e düşerken bulanık katmanınki 89.6–90.1 aralığında sabit kalır. Yanlış alarm oranı bulanık katmanda her gürültü seviyesinde daha düşüktür.

Sonuç olarak Iris üzerinde bulanık katman genel doğruluğu artırmamaktadır. Katkısı, yanlış alarm ile tespit arasında ayarlanabilir bir denge, gürültüye karşı daha kararlı bir davranış ve kural tablosuyla açıklanabilir bir karar sunmasıdır. Daha büyük IoT verisinde (BoT-IoT) bu durumun değişip değişmediği test edilecektir.
