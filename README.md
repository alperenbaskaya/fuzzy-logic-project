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

# E2/E3/E6: Skin, BoT-IoT ve kümeleme ablasyonu (önce veri setleri, data/README.md)
python -m experiments.baseline.run_skin_botiot

# E4/E5/E7: keskin vs bulanık tespit + gürültü testi, sonra grafikler
python -m experiments.proposed.run_fuzzy_iris
python -m experiments.proposed.run_fuzzy
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

### E2 — Skin ve E3 — BoT-IoT (makale Tablo 8 ve 5)

10 tekrar. Tam tablo: [`results/tables/e2_e3_e6.md`](results/tables/e2_e3_e6.md).

| Veri | Algoritma | DR % (bizim / makale) | FAR % (bizim / makale) | Dedektör sayısı (bizim / makale) |
|---|---|---|---|---|
| Skin | V-Detector | 98.60 / 87.20 | 0.03 / 3.04 | 97 / 376 |
| Skin | NaIM-IDS | 98.62 / 98.55 | 0.01 / 3.01 | 21 / 248 |

| Veri | Algoritma | Accuracy % | Recall % | Precision % | F1 % |
|---|---|---|---|---|---|
| BoT-IoT (makale) | NaIM-IDS | 98.60 | 97.73 | 99.24 | 98.47 |
| BoT-IoT, makale özellikleri | NaIM-IDS (bizim) | 98.40 | 98.28 | 99.96 | 99.12 |
| BoT-IoT, makale özellikleri | V-Detector | 92.23 | 91.53 | 99.96 | 95.36 |
| BoT-IoT, temiz özellikler | NaIM-IDS (bizim) | 98.40 | 98.36 | 99.90 | 99.12 |
| BoT-IoT, temiz özellikler | V-Detector | 77.15 | 75.29 | 99.88 | 84.85 |

- NaIM-IDS'in Skin tespit oranı ve BoT-IoT doğruluğu makaleyle 0.2 puan içinde örtüşüyor.
- BoT-IoT'de NaIM, V-Detector'dan çok daha iyi; fark, sızıntı riski taşımayan özelliklerde büyüyor (77 → 98).
- Makalenin seçtiği özelliklerle (`stime`, `sport`, `seq` ...) temiz özellikler arasında NaIM için fark yok. Yani yüksek doğruluk sızıntıdan gelmiyor.
- Bizim V-Detector'ımız Skin'de makaledekinden çok daha iyi (98.6 / 87.2). Makalenin V-Detector parametreleri belirtilmemiş.

### E6 — Kümeleme ablasyonu

| Veri | Kümeleme | F1 % | Uzaklık hesabı sayısı |
|---|---|---|---|
| Skin | FCM (c = 5) | 99.24 | 7.85 M |
| Skin | K-means (c = 5) | 99.29 | 7.11 M |
| Skin | Yok | 99.33 | 17.03 M |
| BoT-IoT | FCM (c = 5) | 99.19 | 1.44 M |
| BoT-IoT | Yok | 99.17 | 2.12 M |

FCM, tolerans aşamasındaki uzaklık hesabını Skin'de %54, BoT-IoT'de %32 azaltıyor; doğruluk değişmiyor. Makale Tablo 9'da FCM çıkarılınca doğruluğun düştüğünü raporluyor (BoT-IoT 97.73 → 97.20). Bizim uygulamamızda budama tam (exact) olduğu için kümeleme yalnızca hesap maliyetini etkiliyor. FCM ile K-means arasında belirgin fark yok. Python döngü maliyeti nedeniyle hesap tasarrufu duvar saati süresine yansımıyor.

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

Sonuç olarak Iris üzerinde bulanık katman genel doğruluğu artırmamaktadır. Katkısı, yanlış alarm ile tespit arasında ayarlanabilir bir denge, gürültüye karşı daha kararlı bir davranış ve kural tablosuyla açıklanabilir bir karar sunmasıdır.

## Comparative Results (Skin ve BoT-IoT, 10 tekrar)

Eşik, eğitim verisinin %30'luk doğrulama kısmında seçildi (ASSUMPTIONS A13). Tam tablolar: [`results/tables/e4_e7_fuzzy.md`](results/tables/e4_e7_fuzzy.md).

| Veri | Yöntem | DR % | FAR % | F1 % | AUC % |
|---|---|---|---|---|---|
| Skin | Keskin (makale) | 98.49 | 0.01 | 99.24 | 99.24 |
| Skin | Kapsama skoru, FIS yok | 99.19 | 0.23 | 99.56 | 99.84 |
| Skin | **FIS Mamdani centroid** | **99.36** | 0.04 | **99.67** | 99.52 |
| Skin | FIS Gauss Sugeno | 99.58 | 0.78 | 99.69 | 99.99 |
| BoT-IoT | Keskin (makale) | 98.60 | 1.71 | 99.22 | 98.45 |
| BoT-IoT | Kapsama skoru, FIS yok | 99.00 | 2.61 | 99.39 | 99.26 |
| BoT-IoT | FIS Mamdani centroid | 98.56 | 5.26 | 99.05 | 99.08 |
| BoT-IoT | **FIS Gauss Sugeno** | **99.63** | 7.20 | **99.51** | **99.78** |
| BoT-IoT | FIS Mamdani, eşik 0.5 | 95.21 | **0.71** | 97.51 | 99.08 |

**Bulgular**

1. **Skin'de bulanık katman makaleyi geçiyor.** Mamdani FIS, F1'i 99.24'ten 99.67'ye çıkarıyor (Wilcoxon p = 0.002) ve yanlış alarm oranını anlamlı biçimde artırmıyor (p = 0.22). Ayrıca aynı şekilde eşiği ayarlanmış FIS'siz skoru da geçiyor (F1 +0.11, p = 0.027; FAR −0.19, p = 0.004). Yani ikinci girdi (self uzaklığı) ve kurallar bilgi katıyor.
2. **BoT-IoT'de tablo karışık.** Gauss Sugeno en yüksek AUC'yi veriyor (99.78; keskin 98.45, FIS'siz skor 99.26, p = 0.002) ve F1'i 0.29 puan artırıyor (p = 0.02). Ancak yanlış alarm oranı 1.7'den 7.2'ye çıkıyor. Test kümesinin %92'si saldırı olduğundan F1'e göre seçilen eşik tespit tarafına kayıyor.
3. **Sabit 0.5 eşiğinde FIS temkinli bir dedektör.** BoT-IoT'de yanlış alarm oranını 1.71'den 0.71'e indiriyor (p = 0.016) ama tespit oranı 3.4 puan düşüyor. Eşik, ağın yanlış alarm toleransına göre seçilebilen bir ayar düğmesi gibi çalışıyor.
4. **Gürültü dayanıklılığı genellenmiyor.** Iris'te gördüğümüz kararlılık Skin ve BoT-IoT'de tekrar etmiyor; σ ≥ 0.05'te bulanık katmanın yanlış alarm oranı keskin karardan daha hızlı artıyor.
5. **Üyelik fonksiyonu ve durulaştırma:** Gauss MF her veri setinde üçgenden daha yüksek AUC veriyor. Durulaştırma yöntemleri arasında MOM en zayıfı; centroid ile bisector arasındaki fark küçük.
