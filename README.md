# B2B Satış ve Pazarlama Karar Paneli

Bir kitap ve kırtasiye toptancısının 2020 yılına ait B2B satış verisiyle hazırlanmış karar destek paneli. Panel; satış, pazarlama ve tedarik zinciri yöneticilerinin geçmişi görmesini ve gelecek ay için tahminlere dayalı karar vermesini amaçlar.

**Canlı panel:** https://veri-madenciligi-odev-puggkd8tka3coki9fmxjez.streamlit.app/

## Dosyalar

| Dosya | Görev |
|---|---|
| `veri_on_isle.py` | Ön işleme: ham Excel'i temizler, sütun türetir ve `temiz_veri.parquet` + `temiz_veri_rapor.json` olarak kaydeder |
| `streamlit_app.py` | Panel: Parquet'i okur, analiz eder, tahmin eder ve görselleştirir |
| `temiz_veri.parquet` | Ön işlenmiş veri (654.089 satır, zstd, ~13 MB; Excel'in 40 MB'ına karşı ~1 sn'de açılır) |
| `temiz_veri_rapor.json` | Ön işleme adımları ve özet (panelde "Veri hazırlama adımları") |
| `model_karsilastirma.py` | Rapor için model ve tahmin yöntemi karşılaştırması; sonuçları `model_sonuclari.xlsx` dosyasına yazar |
| `B2B_Transaction_Data_.xlsx` | Ham veri (664.154 satır); bulutta zorunlu değil |
| `tr_iller.geojson` | Türkiye il sınırları (harita için) |
| `requirements.txt` | Gerekli Python paketleri |
| `.streamlit/config.toml` | Panel teması |

Veri güncellendiğinde `B2B_Transaction_Data_.xlsx` dosyasını değiştirip `python veri_on_isle.py` çalıştırmak yeterlidir. Parquet'i yeniden üretmeyi unutursanız panel Excel'in daha yeni olduğunu algılar, ön işlemeyi kendisi yapar ve Parquet'i yeniden yazar.

## Çalıştırma

```bash
pip install -r requirements.txt
python veri_on_isle.py              # Excel -> temiz_veri.parquet (veri değiştiğinde bir kez)
streamlit run streamlit_app.py      # paneli açar
python model_karsilastirma.py       # rapor tablolarını üretir
```

## Panel sayfaları

Her sayfanın üstünde, seçili filtrelere göre kendini güncelleyen **Satış & Tedarik Aksiyon Özeti** yer alır: (1) son 3 ayda alımı önceki 3 aya göre yarıdan fazla düşen değerli müşteriler ve risk altındaki ciro, (2) gelecek ay talebi en çok artması beklenen kategori grubu ve önerilen stok artışı, (3) ürün karmasına göre fazladan verilen iskonto ve en çok fazla iskonto yapılan il/müşteri. Sayfalar üst menüden seçilir; yalnızca açık sayfa hesaplandığı için geçişler hızlıdır.

| Sayfa | İçerik | İlgili departman |
|---|---|---|
| Çapraz satış | Ürün seçince aynı siparişte (InvoiceNo) en çok birlikte alınan 3 ürün (müşteri oranı, sipariş oranı, lift), tüm öneriler, ürünü alıp önerilen ürünü hiç almamış hedef müşteri listesi (CSV), kategori grupları arası birliktelik ısı haritası, en güçlü ürün çiftleri | Satış |
| İskonto & marj | Ağırlıklı iskonto, fazladan verilen iskonto, il ve müşteri bazında ürün karmasına göre düzeltilmiş fazla iskonto, yüksek ciro + fazla iskonto kadranı, iskonto dilimleri, aylık eğilim, marjı eriten büyük siparişler | Satış, finans |
| Genel bakış | Temel göstergeler, Türkiye il haritası, bölge ve il tabloları, aylık ciro ve gelecek ay beklentisi, günlük ciro, veri hazırlama adımları | Satış, yönetim |
| Ocak 2021 tahmini | Müşteri bazında büyük sipariş tahmini: tahmin ve gerçekleşen karşılaştırması, olasılık güvenilirliği, şehir/bölge/müşteri büyüklüğü/kategori kırılımları, kategori talebi, tahmini belirleyen etkenler, müşteri listesi ve müşteri analizi | Satış, pazarlama |
| Ürünler ve tedarik | Kategori grubu ve ürün bazında gelecek ay adet tahmini (alt–üst aralıkla), ciro yoğunlaşması (Pareto) | Tedarik zinciri |
| Kategoriler | 12 kategori grubunun konumu (pay ve büyüme), grup ayrıntısı ve alt kategoriler | Pazarlama |

Kenar çubuğundaki dönem, il ve kategori grubu filtreleri Genel bakış, Ürünler ve Kategoriler sayfalarını etkiler. Tahmin sayfası her zaman verinin tamamını kullanır.

## Yöntem

**1. Veri hazırlama** (`veri_hazirla`). Kitap adlarının tür düzeltmesi, boşluk kırpma, yer tutucu metinlerin (NULL, #YOK, "-") eksik sayılması, kritik alanı eksik satırların ve iş kuralına aykırı satırların (adet ≤ 0, net fiyat ≤ 0, net fiyat > liste fiyatı) silinmesi, tekrarlayan satırların adetleri toplanarak birleştirilmesi, aykırı değerlerin silinmeden işaretlenmesi. Türetilen sütunlar: ciro ve iskonto, ay/çeyrek/haftanın günü, COVID dönemi, 12 kategori grubu, il-plaka-bölge, ürün türü, ana müşteri. Her adım panelde ve `model_sonuclari.xlsx` dosyasında gerekçesiyle listelenir.

**2. Ciro ve adet tahmini.** 21 aday yöntem (naif, hareketli ortalama, eğilim, üstel düzleştirme, Holt) geriye dönük olarak sınanır: her ay yalnızca önceki aylar kullanılarak tahmin edilir. Son 3 ay test için ayrılır, yöntem öncesindeki ayların ortalama mutlak yüzde hatasına (MAPE) göre seçilir. Aynı yöntem seçimi kategori grubu ve ürün bazındaki adet serilerine de uygulanır. Olası aralık, seçilen yöntemin geçmiş hatalarının standart sapmasıdır.

**3. Büyük sipariş sınıflandırması.** Hedef: müşterinin gelecek ay tek siparişte büyük sipariş (sipariş adetlerinin %90'lık dilimi ve üzeri) verip vermeyeceği. Özellikler yalnızca geçmiş aylardan hesaplanır (geçen ay ve son 3 ay ciro/adet/sipariş, büyük sipariş sıklığı, alım yapılan ay sayısı, son alımdan beri geçen ay). Altı model (lojistik regresyon, Naive Bayes, karar ağacı, kNN, random forest, gradient boosting) zamana göre doğrulamayla karşılaştırılır; sınıf dengesizliği sınıf ağırlıklandırmasıyla ele alınır; kazanan önceden belirlenen kurala göre seçilir ve test verisinde bir kez değerlendirilir.

**4. Olasılık kalibrasyonu.** Sınıf ağırlıklandırması skorları yukarı çektiği için skorlar izotonik regresyonla, modelin görmediği aylardaki gerçekleşme oranlarına çevrilir. Panelde gösterilen olasılıklar bu kalibre edilmiş değerlerdir.

**5. Sepet (çapraz satış) analizi.** Fatura = sepet. 664 bin satırda tüm kalem kümelerini tarayan Apriori yerine yalnızca ikili birliktelikler hesaplanır: seçilen ürünü içeren faturalar gruplanarak her ürünün birlikte görülme sayısı, güven P(B|A), müşteri oranı ve lift = P(B|A)/P(B) bulunur. Ürün çiftleri tablosu fatura × ürün ikili seyrek matrisiyle C = XᵀX olarak tek çarpımda hesaplanır.

**6. İskonto ve marj.** İskonto oranı = 1 − NetPrice / UnitPrice. Veride maliyet olmadığından marj erimesi, ürün karmasıyla açıklanamayan iskonto üzerinden ölçülür: her ürünün tüm müşterilerdeki ağırlıklı iskonto oranı "olağan iskonto" kabul edilir; fazla iskonto = verilen iskonto − olağan iskonto. Bu, pahalı/kampanyalı ürün alan müşterilerin haksız yere işaretlenmesini önler.


## Temel sonuçlar

| | Sonuç |
|---|---|
| Veri | 664.154 ham satır → 654.089 temiz satır, ciro 29,58 milyon TL, 1.201 müşteri, 77 il |
| Ciro tahmin yöntemi | 3 aylık hareketli ortalama (doğrulama MAPE %30,3, test MAPE %19,1) |
| Kazanan sınıflandırıcı | Lojistik regresyon (doğrulama ROC-AUC 0,857; test ROC-AUC 0,882, recall 0,853, precision 0,361) |
| Listenin başı | Test aylarında skoru en yüksek 20 müşterinin %80–95'i büyük sipariş verdi |
| Kalibrasyon | Test Brier skoru 0,182 → 0,108 |

## Sınırlılıklar

- Veri yalnızca bir yılı kapsıyor; mevsimsellik ölçülemiyor. Nisan–Temmuz düşüşü büyük olasılıkla COVID ilk dalga kısıtlamalarından kaynaklanıyor ve mevsim etkisinden ayrılamıyor.
- Model müşterileri iyi sıralıyor, ancak toplam büyük sipariş sayısını test aylarında fazla tahmin ediyor; dönemler arasında gerçekleşme oranı değişiyor.
- Ciro, adet ve sipariş sayısı özellikleri birbiriyle güçlü ilişkili (çoklu doğrusal bağlantı); bu yüzden etkenler panelde gruplanarak gösterilir.
- Model ayarları sabit ve makul değerlerde tutuldu; ayrıntılı ayar araması yapılmadı.
- Panelde gelecek ay tahmini için model ve eşik bilinen tüm aylarla yeniden kurulur; model seçimi ve test yalnızca `model_karsilastirma.py` içinde, eşik eğitim döneminden hesaplanarak yapılır.
