# Yeniden Gerçekleştirme Varsayımları

Referans makalenin kodu yayımlanmamıştır ve bazı ayrıntılar metinde verilmemiştir. Aşağıdaki her varsayım, makalenin ilgili bölümüyle birlikte listelenir. Rapordaki "Reproduction Methodology" ve "Limitations" bölümleri bu dosyaya dayanır.

| Kod | Konu | Makalede | Bizim kararımız | Gerekçe |
|---|---|---|---|---|
| A1 | Eğitim/test düzeni | Tablo 7'de "Training data: Setosa (50%)" | Self ve non-self örneklerin aynı oranı eğitimde kullanılır; test tüm veri üzerinde yapılır | %100 eğitimde FAR = 0 olması testin tüm veri üzerinde yapıldığını gösterir. NaIM sınır dışı dedektörlerde non-self örnek kullandığı için non-self eğitim verisi gerekir |
| A2 | FCM ile tolerans | "Küme merkezi ve boyutu bireysel tolerans yerine kullanılır" (Bölüm 3.2) | Küme yarıçapı R_k = üyelerin merkeze en büyük uzaklığı. Üçgen eşitsizliğiyle alt sınır d(x, v_k) − R_k hesaplanır; alt sınır mevcut en iyi uzaklıktan büyük olan kümeler atlanır | Sonuç kaba kuvvetle birebir aynıdır (doğruluk kaybı yok); tasarruf `n_dist` sayacıyla ölçülür |
| A3 | Kümeleme parametreleri | S = Com/Sep (Denk. 19, iki küme için); en iyi m = 2.3 (Şekil 4b) | Denk. 19 c kümeye genelleştirildi, Sep = merkezler arası en küçük uzaklık. c Com/Sep ile seçilir, m = 2.3 sabit | Com terimi u^m içerdiği için indeks m büyüdükçe küçülür ve her zaman en büyük m'yi seçer (deneyde doğrulandı) |
| A4 | Self yarıçapı | Verilmemiş | r_self = 0.05 ([0,1] ölçeğinde) | V-Detector literatüründe yaygın değer |
| A5 | V-Detector durma kuralı | Ji & Dasgupta'da hipotez testi | Ardışık 1/(1−c0) = 100 aday mevcut dedektörlerce kapsanınca dur (c0 = 0.99) | Basitleştirilmiş, yaygın kullanılan sürüm |
| A6 | Sınır bölgesi tanımı | Verilmemiş | Self'e en yakın %30'luk non-self dilimi sınır bölgesidir | Makalenin şeklindeki sınır/sınır dışı ayrımını sayısallaştıran en basit ölçüt |
| A7 | Dedektör uygunluğu | "Düşük örtüşme, yüksek kapsama, self ile eşleşmeme" (kaynak [45]) | F = kapsama + 0.5·r/√d − 0.3·örtüşme; r ≤ 0 ise geçersiz | Kaynak [45]'in ağırlıkları makalede yok |
| A8 | Alt popülasyon bölmesi | Denk. 20, Diff = abs(p − c); kategorik frekans adımı | Ortalama mutlak fark, eşik b = 0.15; kategorik frekans adımı uygulanmadı | Kullanılan veri setleri sayısal; kategorikler LabelEncoder ile kodlanıyor |
| A9 | RFO üreme adımı | Orijinal RFO'da yeni birey = κ·(en iyi iki tilkinin ortası) | Yeni birey, en iyi iki tilkinin arasındaki "yaşam alanı" içinde rastgele konum | Orijinal formül bireyleri sistematik olarak orijine doğru çekiyor |
| A10 | Özellik seçimi | RFO–HHO hibrit (Bölüm 3.1) | Kapsam dışı (MRR). IDS veri setlerinde makalenin Tablo 3 listesi veya sızıntısız alternatif kullanılır | Makalenin kendi ablasyonu (Tablo 9) FS'nin katkısını ~%0.7 doğruluk olarak gösteriyor |
| A11 | Skin eğitim oranı | Verilmemiş | Her sınıftan %2 eğitim (1.017 self, 3.884 non-self); test kalan ~240 bin örnekte | Makale oranı yazmıyor; %2 ile bile self uzayı yoğun örnekleniyor |
| A12 | BoT-IoT alt kümesi | Tablo 3'te 46 → 6 özellik; örnek sayısı yok | Resmî %5 alt küme: 477 normalin tamamı + kategoriye orantılı 5.050 saldırı (her kategoriden en az 50). İki özellik kümesi: makalenin listesi ve veri seti yazarlarının "10 best" listesinden `seq` çıkarılmış 9 özellik | BoT-IoT'de normal trafik %0.01; alt küme dışında self kümesi oluşturulamaz |
| A13 | Bulanık katman eşiği | — (bizim katkımız) | Eğitim kısmı 70/30 bölünür; NaIM 70'te eğitilir, eşik 30'da F1'i en büyük yapacak şekilde seçilir | Test verisine bakmadan eşik seçmek için |
| A14 | Parametre hassasiyeti verisi | Şekil 4: NSL-KDD üzerinde S, m, G, eşik | BoT-IoT (c, m, b, r_self) ve Skin (c, m); 5 tekrar | NSL-KDD bu projede kullanılmadı; aynı parametrelerin etkisi kullandığımız veri setlerinde ölçüldü |
