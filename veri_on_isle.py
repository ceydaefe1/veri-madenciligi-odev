"""
Veri ön işleme · B2B Satış Paneli

Ham Excel'i (B2B_Transaction_Data_.xlsx) bir kez temizler ve sıkıştırılmış Parquet olarak kaydeder.
Panel (streamlit_app.py) doğrudan bu Parquet dosyasını okur: Excel'e göre çok daha küçük ve hızlıdır.

Çalıştırma:  python veri_on_isle.py
Çıktılar:    temiz_veri.parquet       (temiz satır verisi)
             temiz_veri_rapor.json    (ön işleme adımları ve özet; panelde gösterilir)
"""
import datetime as dt
import json
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
VERI_DOSYASI = "B2B_Transaction_Data_.xlsx"
VERI_YOLU = os.path.join(BASE, VERI_DOSYASI)
PARQUET_DOSYASI = "temiz_veri.parquet"
PARQUET_YOLU = os.path.join(BASE, PARQUET_DOSYASI)
RAPOR_YOLU = os.path.join(BASE, "temiz_veri_rapor.json")
BILINMIYOR = "Bilinmiyor"

# ===========================================================================
# A) VERİ ÖN İŞLEME
# ===========================================================================
METIN_SUTUNLARI = ["InvoiceNo", "CustomerID", "StockCode", "Description", "Category", "City"]
SAYI_SUTUNLARI = ["Quantity", "NetPrice", "UnitPrice"]
KRITIK_SUTUNLAR = ["InvoiceNo", "CustomerID", "InvoiceDate", "StockCode",
                   "Quantity", "NetPrice", "UnitPrice"]
DOLDURULACAK_SUTUNLAR = ["City", "Category", "Description"]

YER_TUTUCU = re.compile(
    r"^(#YOK|#N/A|#NA|#NULL|NULL|NA|N/A|NAN|NONE|YOK|BILINMIYOR|BİLİNMİYOR"
    r"|BILINMEYEN|BİLİNMEYEN|-+|\?+)$",
    re.IGNORECASE,
)
TOPLU_SIPARIS_ADET = 100      # tek satırda bu adetin üstü "toplu sipariş" olarak işaretlenir
YER_TUTUCU_FIYAT = 1.0        # 1 TL altı liste fiyatı gerçek fiyat değil, yer tutucu kabul edilir
GUNLER = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]

# ---------------------------------------------------------------------------
# Kategori grupları (306 alt kategori -> 12 grup + "Diğer" + "Bilinmiyor")
# ---------------------------------------------------------------------------
# Kurallar yukarıdan aşağıya denenir; bir kategori ilk eşleştiği gruba atanır, bu yüzden sıra önemlidir:
#   - Psikoloji, Yaşam'dan önce gelir ("Ruh Sağlığı" Yaşam & Sağlık'a düşmesin diye)
#   - Yaşam, Çocuk'tan önce gelir ("Aile - Çocuk", "Hamilelik ve Çocuk Sağlığı" ebeveynlik kitabıdır)
#   - Çocuk, Din ve Edebiyat'tan önce gelir ("İslami Çocuk Kitapları", "Çocuk Klasik" çocuk kitabıdır)
#   - Edebiyat, Bilim'den önce gelir ("Bilim Kurgu" bir edebiyat türüdür)
#   - Dil ve Sanat, Tarih'ten önce gelir ("Osmanlıca", "Sanat Tarihi" Tarih'e düşmesin diye)
# \b kelime sınırıdır: \bDin\b "Kadın"ı, \bDers "Readers"ı, \bPsikoloji "Parapsikoloji"yi yakalamaz.
KATEGORI_KURALLARI = [
    ("Hobi, Oyun & Kırtasiye", r"Puzzle|Oyuncak|Eğitici Oy|Kutu Oyun|Oyun Kitap|Bulmaca|Çıkartma|Defter|Ajanda|"
                               r"Organizer|Kalem|Hobi|Hediyelik|Kap Kağıt|Masa Set|Takvim|Kart Zarf"),
    ("Eğitim & Sınav",         r"YKS|TYT|AYT|LGS|KPSS|KPDS|YDS|DGS|ALES|TOEFL|Sınav|Exam|\bTest|\bDers|Okul Kitap|"
                               r"Kolej|Matematik|Fizik|Kimya|Biyoloji|Fen ve Teknoloji|Sosyal Bilgiler|^Türkçe$|"
                               r"^Eğitim$|Öğretmenlik|İstatistik"),
    ("Psikoloji & Felsefe",    r"Kişisel Gelişim|Self Help|\bPsikoloji|Ruh Sağlığı|Kadın Erkek İlişkisi|"
                               r"Mind and Spirit|Bilgelik|Felsefe"),
    ("Yaşam & Sağlık",         r"^Aile|Hamilelik|Sağlı|Beslenme|Diyet|Yemek|Food|Gastronomi|İçki|Spor|Fitness|Yoga|"
                               r"Moda|Hayvan Bakımı|Bahçe|Cinsellik|Uyuşturucu|Outdoor|Avcılık"),
    ("Çocuk & Genç",           r"Çocuk|Masal|Okul Öncesi|Preschool|Children|Storybook|Characters|Boyama|Etkinlik|"
                               r"Tübitak|\d\+ Yaş|İlk Yaş|Sesli"),
    ("Din & Maneviyat",        r"İslam|Tasavvuf|\bDin\b|İnanç|Hristiyan|Musevi|Budizm|Alevi|Religion|Spirituality|"
                               r"Astroloji|Kehanet|Rüya|Parapsikoloji"),
    ("Dil & Sözlük",           r"Sözlük|Dictionary|Grammar|Graded Readers|Phrase Book|Language|Dilbilim|\bDil\b|"
                               r"Russian|Spanish|Italian|German|Arabic|Kürtçe|Osmanlıca"),
    ("Edebiyat",               r"Roman|Öykü|Klasik|Classics|Literature|Şiir|Polisiye|Kurgu|Fantastik|Fantasy|Korku|"
                               r"Gerilim|Mystery|Thriller|Macera|Romantik|Edebiyat|Tiyatro|Oyunları|Senaryo|Mizah|"
                               r"Fıkra|Karikatür|Deneme|Yazın|Anlatı|Biyografi|Biography|Günlük Anı|Mektup|Röportaj|"
                               r"Söyleşi|Mitoloji|Destan|Halk Ozan|Antoloji|Erotik|Gezi"),
    ("Sanat",                  r"Sanat|Resim|Müzik|Sinema|Mimari|Fotoğraf|Fotograf"),
    ("İş & Ekonomi",           r"Yönetim|Pazarlama|Satış|Ekonomi|Economics|Business|İktisat|İşletme|Finans|Muhasebe|"
                               r"Maliye|Borsa|SPK|İnsan Kaynakları|Dış Ticaret|Turizm|İş Adamları"),
    ("Tarih, Toplum & Politika", r"Tarih|History|Atatürk|Osmanlı|Ermeni|Arkeoloji|Kaşif|Keşif|Politika|Politics|"
                                 r"Sosyoloji|Antropoloji|Etnoloji|Sosyal Bilim|Hukuk|Kamu|Sivil Toplum|Uluslararası|"
                                 r"Avrupa Birliği|Devlet Adamları|Medya|İletişim|Gazeteci|^Kadın$|Suç|İstanbul|Türkiye"),
    ("Bilim & Başvuru",        r"Bilim|Mühendislik|Bilgisayar|Bilişim|Tıp|Anatomi|Astronomi|Jeoloji|Botanik|Zooloji|"
                               r"Doğa|Coğrafya|Atlas|Harita|Ülkeler|Ansiklopedi|Almanak|Rehber|Reference"),
]

# ---------------------------------------------------------------------------
# Türkiye 81 il: ASCII büyük harfli anahtar -> (plaka, resmi ad, bölge)
# ---------------------------------------------------------------------------
ILLER = {
    "ADANA": (1, "Adana", "Akdeniz"), "ADIYAMAN": (2, "Adıyaman", "Güneydoğu Anadolu"),
    "AFYONKARAHISAR": (3, "Afyonkarahisar", "Ege"), "AFYON": (3, "Afyonkarahisar", "Ege"),
    "AGRI": (4, "Ağrı", "Doğu Anadolu"), "AMASYA": (5, "Amasya", "Karadeniz"), "ANKARA": (6, "Ankara", "İç Anadolu"),
    "ANTALYA": (7, "Antalya", "Akdeniz"), "ARTVIN": (8, "Artvin", "Karadeniz"), "AYDIN": (9, "Aydın", "Ege"),
    "BALIKESIR": (10, "Balıkesir", "Marmara"), "BILECIK": (11, "Bilecik", "Marmara"),
    "BINGOL": (12, "Bingöl", "Doğu Anadolu"), "BITLIS": (13, "Bitlis", "Doğu Anadolu"),
    "BOLU": (14, "Bolu", "Karadeniz"), "BURDUR": (15, "Burdur", "Akdeniz"), "BURSA": (16, "Bursa", "Marmara"),
    "CANAKKALE": (17, "Çanakkale", "Marmara"), "CANKIRI": (18, "Çankırı", "İç Anadolu"),
    "CORUM": (19, "Çorum", "Karadeniz"), "DENIZLI": (20, "Denizli", "Ege"),
    "DIYARBAKIR": (21, "Diyarbakır", "Güneydoğu Anadolu"), "EDIRNE": (22, "Edirne", "Marmara"),
    "ELAZIG": (23, "Elazığ", "Doğu Anadolu"), "ERZINCAN": (24, "Erzincan", "Doğu Anadolu"),
    "ERZURUM": (25, "Erzurum", "Doğu Anadolu"), "ESKISEHIR": (26, "Eskişehir", "İç Anadolu"),
    "GAZIANTEP": (27, "Gaziantep", "Güneydoğu Anadolu"), "GIRESUN": (28, "Giresun", "Karadeniz"),
    "GUMUSHANE": (29, "Gümüşhane", "Karadeniz"), "HAKKARI": (30, "Hakkari", "Doğu Anadolu"),
    "HATAY": (31, "Hatay", "Akdeniz"), "ISPARTA": (32, "Isparta", "Akdeniz"), "MERSIN": (33, "Mersin", "Akdeniz"),
    "ICEL": (33, "Mersin", "Akdeniz"), "ISTANBUL": (34, "İstanbul", "Marmara"), "IZMIR": (35, "İzmir", "Ege"),
    "KARS": (36, "Kars", "Doğu Anadolu"), "KASTAMONU": (37, "Kastamonu", "Karadeniz"),
    "KAYSERI": (38, "Kayseri", "İç Anadolu"), "KIRKLARELI": (39, "Kırklareli", "Marmara"),
    "KIRSEHIR": (40, "Kırşehir", "İç Anadolu"), "KOCAELI": (41, "Kocaeli", "Marmara"),
    "KONYA": (42, "Konya", "İç Anadolu"), "KUTAHYA": (43, "Kütahya", "Ege"),
    "MALATYA": (44, "Malatya", "Doğu Anadolu"), "MANISA": (45, "Manisa", "Ege"),
    "KAHRAMANMARAS": (46, "Kahramanmaraş", "Akdeniz"), "KMARAS": (46, "Kahramanmaraş", "Akdeniz"),
    "MARDIN": (47, "Mardin", "Güneydoğu Anadolu"), "MUGLA": (48, "Muğla", "Ege"), "MUS": (49, "Muş", "Doğu Anadolu"),
    "NEVSEHIR": (50, "Nevşehir", "İç Anadolu"), "NIGDE": (51, "Niğde", "İç Anadolu"),
    "ORDU": (52, "Ordu", "Karadeniz"), "RIZE": (53, "Rize", "Karadeniz"), "SAKARYA": (54, "Sakarya", "Marmara"),
    "SAMSUN": (55, "Samsun", "Karadeniz"), "SIIRT": (56, "Siirt", "Güneydoğu Anadolu"),
    "SINOP": (57, "Sinop", "Karadeniz"), "SIVAS": (58, "Sivas", "İç Anadolu"),
    "TEKIRDAG": (59, "Tekirdağ", "Marmara"), "TOKAT": (60, "Tokat", "Karadeniz"),
    "TRABZON": (61, "Trabzon", "Karadeniz"), "TUNCELI": (62, "Tunceli", "Doğu Anadolu"),
    "SANLIURFA": (63, "Şanlıurfa", "Güneydoğu Anadolu"), "USAK": (64, "Uşak", "Ege"),
    "VAN": (65, "Van", "Doğu Anadolu"), "YOZGAT": (66, "Yozgat", "İç Anadolu"),
    "ZONGULDAK": (67, "Zonguldak", "Karadeniz"), "AKSARAY": (68, "Aksaray", "İç Anadolu"),
    "BAYBURT": (69, "Bayburt", "Karadeniz"), "KARAMAN": (70, "Karaman", "İç Anadolu"),
    "KIRIKKALE": (71, "Kırıkkale", "İç Anadolu"), "BATMAN": (72, "Batman", "Güneydoğu Anadolu"),
    "SIRNAK": (73, "Şırnak", "Güneydoğu Anadolu"), "BARTIN": (74, "Bartın", "Karadeniz"),
    "ARDAHAN": (75, "Ardahan", "Doğu Anadolu"), "IGDIR": (76, "Iğdır", "Doğu Anadolu"),
    "YALOVA": (77, "Yalova", "Marmara"), "KARABUK": (78, "Karabük", "Karadeniz"),
    "KILIS": (79, "Kilis", "Güneydoğu Anadolu"), "OSMANIYE": (80, "Osmaniye", "Akdeniz"),
    "DUZCE": (81, "Düzce", "Karadeniz"),
}
TR_ASCII = str.maketrans("İIıŞşĞğÜüÖöÇç", "IIISSGGUUOOCC")

# ---------------------------------------------------------------------------
# COVID dönemleri (Türkiye, 2020)
# ---------------------------------------------------------------------------
#   11 Mart 2020   : ilk vaka açıklandı
#   11 Mart-31 May : okulların kapanması, sokağa çıkma kısıtlamaları
#   1 Haziran 2020 : "yeni normalleşme" başladı
#   20 Kasım 2020  : hafta sonu sokağa çıkma kısıtlamaları yeniden başladı
COVID_SINIRLARI = ["2020-03-11", "2020-06-01", "2020-11-20"]
COVID_ETIKETLERI = ["COVID öncesi", "İlk dalga kısıtlamaları", "Normalleşme", "İkinci dalga kısıtlamaları"]


# ===========================================================================
# YARDIMCILAR
# ===========================================================================
class _Kayit:
    """Ön işleme adımlarını sırayla kaydeder."""

    def __init__(self):
        self.satirlar = []

    def ekle(self, asama, adim, once, sonra, etkilenen, islem, gerekce):
        self.satirlar.append({
            "Aşama": asama, "Adım": adim, "Önce (satır)": once, "Sonra (satır)": sonra,
            "Etkilenen": etkilenen, "İşlem": islem, "Gerekçe": gerekce,
        })

    def tablo(self):
        t = pd.DataFrame(self.satirlar)
        t.insert(0, "No", range(1, len(t) + 1))
        return t


def _bos_mu(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def _metne_cevir(v):
    """Excel'in sayıya/tarihe/saate çevirdiği kitap adlarını metne geri çevirir."""
    if _bos_mu(v):
        return np.nan
    if isinstance(v, dt.datetime):            # "22/11/63" -> 1963-11-22 olarak okunmuş
        return v.strftime("%d/%m/%y")
    if isinstance(v, dt.time):                # "3:01" -> 03:01:00 olarak okunmuş
        return f"{v.hour}:{v.minute:02d}"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _ham_oku(yol):
    """Hiçbir değeri kendiliğinden eksik saymadan okur; yer tutucuları biz yakalarız."""
    return pd.read_excel(
        yol, engine="calamine", keep_default_na=False, na_values=[],
        dtype={"InvoiceNo": str, "CustomerID": str, "StockCode": str},
    )


def kategori_grubu(kategori):
    """Tek bir alt kategoriyi gruba atar."""
    if pd.isna(kategori) or kategori == BILINMIYOR:
        return BILINMIYOR
    for grup, kalip in KATEGORI_KURALLARI:
        if re.search(kalip, str(kategori), flags=re.IGNORECASE):
            return grup
    return "Diğer"


def il_bilgisi(sehir):
    """Şehir yazımını (İZMİR, Izmir, K.MARAS) resmi il adı, plaka ve bölgeye çevirir."""
    if pd.isna(sehir) or sehir == BILINMIYOR:
        return 0, BILINMIYOR, BILINMIYOR
    anahtar = str(sehir).translate(TR_ASCII).upper().replace(".", "").replace(" ", "")
    return ILLER.get(anahtar, (0, str(sehir).title(), "Diğer"))


# ===========================================================================
# ANA FONKSİYON
# ===========================================================================
def veri_hazirla(yol):
    """Ham Excel'i okur, temizler, yeni sütunlar türetir ve (temiz_veri, rapor) döndürür."""
    k = _Kayit()
    df = _ham_oku(yol)
    ham_satir = len(df)
    k.ekle("Temizleme", "Ham veri okundu", ham_satir, ham_satir, 0, "—",
           "Excel dosyası değiştirilmeden, hiçbir değer kendiliğinden eksik sayılmadan okundu.")

    # -----------------------------------------------------------------------
    # A) TEMİZLEME
    # -----------------------------------------------------------------------
    # A1) Metin sütunları: tür düzeltme, boşluk kırpma, yer tutucuları eksik yapma
    ad_duzeltme = int(df["Description"].map(
        lambda v: isinstance(v, (dt.datetime, dt.time, int, float)) and not _bos_mu(v)).sum())
    df["Description"] = df["Description"].astype(object).map(_metne_cevir)
    bosluk, yer_tutucu = 0, {}
    for c in METIN_SUTUNLARI:
        s = df[c].astype(object).where(df[c].notna(), None).astype(str).where(df[c].notna())
        kirpik = s.str.strip()
        bosluk += int((kirpik != s).fillna(False).sum())
        maske = (kirpik.eq("") | kirpik.str.fullmatch(YER_TUTUCU)).fillna(False).astype(bool)
        if maske.any():
            yer_tutucu[c] = int(maske.sum())
        df[c] = kirpik.mask(maske)
    k.ekle("Temizleme", "Kitap adı tür düzeltmesi", len(df), len(df), ad_duzeltme, "Metne çevrildi",
           "Excel bazı kitap adlarını sayı, tarih ya da saat sanmış (örn. '1984', '22/11/63', '3:01').")
    k.ekle("Temizleme", "Boşluk kırpma", len(df), len(df), bosluk, "Baş/son boşluklar silindi",
           "'Roman' ile 'Roman ' aynı kategori iken ayrı sayılıyordu.")
    k.ekle("Temizleme", "Yer tutucular", len(df), len(df), sum(yer_tutucu.values()), "Eksik değer yapıldı",
           "NULL, #YOK, Bilinmeyen, '-' gibi metinler gerçek değer değil: "
           + ", ".join(f"{c} {n:,}" for c, n in yer_tutucu.items()) + ".")

    # A2) Tür dönüşümleri
    cevrilemeyen = {}
    for c in SAYI_SUTUNLARI + ["InvoiceDate"]:
        once_bos = df[c].isna()
        if c == "InvoiceDate":
            df[c] = pd.to_datetime(df[c], errors="coerce")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        n = int((df[c].isna() & ~once_bos).sum())
        if n:
            cevrilemeyen[c] = n
    k.ekle("Temizleme", "Tür dönüşümü", len(df), len(df), sum(cevrilemeyen.values()),
           "Sayı ve tarih türüne çevrildi",
           "Hesaplama için gerekli. Sayıya/tarihe çevrilemeyen değerler (örn. 'NULL') eksik sayıldı"
           + (": " + ", ".join(f"{c} {n:,}" for c, n in cevrilemeyen.items()) if cevrilemeyen else "")
           + ".")

    eksik_once = df.isna().sum()

    # A3) Kritik alanı eksik satırlar
    once = len(df)
    eksik_kritik = df[KRITIK_SUTUNLAR].isna().any(axis=1)
    df = df[~eksik_kritik].copy()
    k.ekle("Temizleme", "Kritik alanı eksik satırlar", once, len(df), once - len(df), "Silindi",
           "Tarih, adet, fiyat, fatura, müşteri veya ürün kodu olmadan ciro ve trend hesaplanamaz.")

    # A4) İş kuralına aykırı satırlar
    kurallar = {
        "Adet ≤ 0": df["Quantity"] <= 0,
        "Net fiyat ≤ 0": df["NetPrice"] <= 0,
        f"Liste fiyatı < {YER_TUTUCU_FIYAT:g} TL (yer tutucu)": df["UnitPrice"] < YER_TUTUCU_FIYAT,
        "Net fiyat > liste fiyatı": df["NetPrice"] > df["UnitPrice"],
    }
    gecersiz = pd.concat(kurallar, axis=1).any(axis=1)
    once = len(df)
    df = df[~gecersiz].copy()
    k.ekle("Temizleme", "Geçersiz değerli satırlar", once, len(df), once - len(df), "Silindi",
           "; ".join(f"{ad}: {int(m.sum())}" for ad, m in kurallar.items())
           + ". Satış verisinde anlamsız değerler (kesişenler bir kez sayıldı).")

    # A5) Eksik metinleri doldurma
    doldurulan = {c: int(df[c].isna().sum()) for c in DOLDURULACAK_SUTUNLAR}
    df[DOLDURULACAK_SUTUNLAR] = df[DOLDURULACAK_SUTUNLAR].fillna(BILINMIYOR)
    k.ekle("Temizleme", "Eksik şehir/kategori/ürün adı", len(df), len(df), sum(doldurulan.values()),
           f"'{BILINMIYOR}' yazıldı",
           "Satış gerçek; silinirse ciro eksik kalır. Bu müşterilerin başka satırlarında da şehir yok. "
           + ", ".join(f"{c} {n:,}" for c, n in doldurulan.items()) + ".")

    # A6) Tekrarlayan satırlar: gerçek sipariş kabul edilip birleştirilir
    # Quantity dışında her şeyi aynı olan satırlar iki türdür:
    #   A) adetleri farklı  -> aynı faturada aynı ürünün ayrı kalemleri
    #   B) adetleri de aynı -> sisteme iki kez girilmiş kayıt OLABİLİR; ancak B2B'de aynı gün verilen
    #      siparişler tek faturada toplanabildiğinden ikinci sipariş olması da mümkündür.
    # Veride kalem numarası olmadığı için iki ihtimal ayırt edilemez. B'lerin tamamı kopya sayılsa bile
    # ciro yalnızca %0,55 değişiyor. Bu yüzden ikisi de gerçek sipariş kabul edilip adetleri toplanır.
    anahtar = [c for c in df.columns if c != "Quantity"]
    tam_kopya = int(df.duplicated().sum())
    once, ciro_once = len(df), float((df["Quantity"] * df["NetPrice"]).sum())
    df = df.groupby(anahtar, as_index=False, dropna=False)["Quantity"].sum()
    ciro_sonra = float((df["Quantity"] * df["NetPrice"]).sum())
    birlestirilen = once - len(df)
    k.ekle("Temizleme", "Tekrarlayan satırlar", once, len(df), birlestirilen, "Adetleri toplanarak birleştirildi",
           f"Aynı fatura-ürün-fiyatla tekrarlayan satırlar (bunların {tam_kopya:,} tanesi adet dahil birebir aynı). "
           "Kalem numarası olmadığı için kopya kayıt olduğu kanıtlanamıyor; B2B'de aynı gün verilen "
           "siparişler tek faturada toplanabildiğinden gerçek sipariş kabul edildi. "
           f"Ciro değişmedi ({ciro_once:,.0f} → {ciro_sonra:,.0f} TL).")

    # A7) Aykırı değerler: silinmez, işaretlenir
    df["Toplu_siparis"] = df["Quantity"] > TOPLU_SIPARIS_ADET
    fiyat_ust = df["UnitPrice"].quantile(0.999)
    df["Yuksek_fiyat"] = df["UnitPrice"] > fiyat_ust
    k.ekle("Temizleme", "Aykırı değerler", len(df), len(df),
           int(df["Toplu_siparis"].sum() + df["Yuksek_fiyat"].sum()), "İşaretlendi (silinmedi)",
           f"Adet > {TOPLU_SIPARIS_ADET}: {int(df['Toplu_siparis'].sum()):,} satır; "
           f"liste fiyatı > {fiyat_ust:,.0f} TL (%99,9 dilim): {int(df['Yuksek_fiyat'].sum()):,} satır. "
           "B2B'de toplu alım ve pahalı setler gerçek satıştır; silmek ciroyu bozar.")

    # -----------------------------------------------------------------------
    # B) TÜRETME
    # -----------------------------------------------------------------------
    n = len(df)

    # B1) Ciro ve iskonto
    df["Tutar"] = df["Quantity"] * df["NetPrice"]
    df["ListeTutar"] = df["Quantity"] * df["UnitPrice"]
    df["IskontoTutari"] = df["ListeTutar"] - df["Tutar"]
    df["IskontoOrani"] = (1 - df["NetPrice"] / df["UnitPrice"]).clip(0, 1)
    k.ekle("Türetme", "Ciro ve iskonto", n, n, n, "Tutar, ListeTutar, IskontoTutari, IskontoOrani",
           "Tutar = adet × net fiyat; ListeTutar = adet × liste fiyatı; "
           "İskonto oranı = 1 − net fiyat / liste fiyatı.")

    # B2) Zaman boyutları
    df["Ay"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Ceyrek"] = "Ç" + df["InvoiceDate"].dt.quarter.astype(str)
    df["HaftaninGunu"] = pd.Categorical(df["InvoiceDate"].dt.day_of_week.map(dict(enumerate(GUNLER))),
                                        categories=GUNLER, ordered=True)
    df["CovidDonemi"] = pd.cut(df["InvoiceDate"],
                               bins=[pd.Timestamp.min, *map(pd.Timestamp, COVID_SINIRLARI), pd.Timestamp.max],
                               labels=COVID_ETIKETLERI, right=False)
    k.ekle("Türetme", "Zaman boyutları", n, n, n, "Ay, Ceyrek, HaftaninGunu, CovidDonemi",
           "Aylık trend, sezonsallık ve haftalık talep analizi için. COVID dönemleri: ilk vaka (11 Mart), "
           "normalleşme (1 Haziran), hafta sonu kısıtlamaları (20 Kasım).")

    # B3) Kategori grupları
    eslesme = {kat: kategori_grubu(kat) for kat in df["Category"].unique()}
    df["KategoriGrubu"] = df["Category"].map(eslesme)
    diger = (df["KategoriGrubu"] == "Diğer").mean() * 100
    k.ekle("Türetme", "Kategori grupları", n, n, df["Category"].nunique(), "KategoriGrubu",
           f"{df['Category'].nunique()} alt kategori, kural tabanlı eşlemeyle "
           f"{df['KategoriGrubu'].nunique()} gruba toplandı; 'Diğer'e düşen satır payı %{diger:.1f}.")

    # B4) İl ve bölge
    il = {sehir: il_bilgisi(sehir) for sehir in df["City"].unique()}
    df["PlakaKodu"] = df["City"].map(lambda s: il[s][0])
    df["Il"] = df["City"].map(lambda s: il[s][1])
    df["Bolge"] = df["City"].map(lambda s: il[s][2])
    eslesmeyen = sorted(s for s, b in il.items() if b[2] == "Diğer")
    k.ekle("Türetme", "İl ve bölge", n, n, df["City"].nunique(), "PlakaKodu, Il, Bolge",
           "Farklı şehir yazımları (İZMİR, Izmir, K.MARAS) resmi il adı, plaka ve coğrafi bölgeye eşlendi. "
           + (f"Eşleşmeyen: {', '.join(eslesmeyen)}." if eslesmeyen else "Eşleşmeyen şehir yok."))

    # B5) Ürün türü ve kampanya
    df["UrunTuru"] = np.where(df["StockCode"].str[:3].isin(["978", "979"]), "Kitap", "Kitap dışı")
    df["Kampanya"] = df["Description"].str.contains(r"Kampanya|İADESİZ|İadesiz", regex=True, na=False)
    k.ekle("Türetme", "Ürün türü ve kampanya", n, n, int((df["UrunTuru"] == "Kitap dışı").sum()),
           "UrunTuru, Kampanya",
           "ISBN 978/979 ile başlayan kodlar kitap, diğerleri kitap dışı (puzzle, kırtasiye vb.). "
           f"Adında 'Kampanya' ya da 'İadesiz' geçen {int(df['Kampanya'].sum()):,} satır kampanyalı işaretlendi.")

    # B6) Ana müşteri
    # Bazı müşteri kodları "A01212-02" biçiminde. "-" sonrasındaki ekin aynı müşterinin şube ya da teslimat
    # noktasını gösterdiği varsayıldı. Bu varsayım veride doğrulanamaz; ek bulunmayan kodlar aynen kalır.
    df["AnaMusteriID"] = df["CustomerID"].str.split("-").str[0]
    alt = int(df.loc[df["CustomerID"].str.contains("-"), "CustomerID"].nunique())
    k.ekle("Türetme", "Ana müşteri", n, n, alt, "AnaMusteriID",
           f"{alt} şube kodu ('A01212-02' gibi) ana müşteri altında toplandı: "
           f"{df['CustomerID'].nunique():,} müşteri kodu → {df['AnaMusteriID'].nunique():,} ana müşteri. "
           "Ekin şube/teslimat noktası olduğu varsayıldı.")

    # -----------------------------------------------------------------------
    # RAPOR
    # -----------------------------------------------------------------------
    eksik = pd.DataFrame({
        "Eksik (temizlik öncesi)": eksik_once,
        "Eksik (temizlik sonrası)": df.reindex(columns=eksik_once.index).isna().sum(),
    }).astype(int)
    eksik["Oran (%)"] = (eksik["Eksik (temizlik öncesi)"] / ham_satir * 100).round(3)
    eksik = eksik.rename_axis("Sütun").reset_index()

    rapor = {
        "adimlar": k.tablo(),
        "eksik": eksik,
        "ozet": {
            "ham_satir": ham_satir,
            "temiz_satir": len(df),
            "silinen": int(eksik_kritik.sum() + gecersiz.sum()),
            "birlestirilen": birlestirilen,
            "duzeltilen_hucre": ad_duzeltme + bosluk + sum(yer_tutucu.values()),
            "doldurulan": sum(doldurulan.values()),
            "ciro": float(df["Tutar"].sum()),
        },
    }
    return df.reset_index(drop=True), rapor


# ===========================================================================
# PARQUET KAYDET / OKU
# ===========================================================================
def parquet_kaydet(df, rapor, yol=PARQUET_YOLU, rapor_yolu=RAPOR_YOLU):
    """Temiz veriyi zstd ile sıkıştırılmış Parquet'e, raporu JSON'a yazar.
    Parquet metin sütunlarını zaten sözlükle (dictionary) kodlar; ayrıca kategoriye çevirmek okumayı yavaşlatır."""
    df.to_parquet(yol, engine="pyarrow", compression="zstd", index=False)
    with open(rapor_yolu, "w", encoding="utf-8") as f:
        json.dump({
            "adimlar": rapor["adimlar"].to_dict(orient="records"),
            "eksik": rapor["eksik"].to_dict(orient="records"),
            "ozet": rapor["ozet"],
            "kaynak": VERI_DOSYASI,
            "olusturma": pd.Timestamp.now(tz="Europe/Istanbul").isoformat(),
        }, f, ensure_ascii=False, indent=1, default=float)


def parquet_oku(yol=PARQUET_YOLU, rapor_yolu=RAPOR_YOLU, kolonlar=None):
    """kolonlar verilirse yalnızca bu sütunlar okunur (bellek tasarrufu)."""
    df = pd.read_parquet(yol, engine="pyarrow", columns=kolonlar)
    with open(rapor_yolu, encoding="utf-8") as f:
        r = json.load(f)
    rapor = {"adimlar": pd.DataFrame(r["adimlar"]), "eksik": pd.DataFrame(r["eksik"]), "ozet": r["ozet"],
             "olusturma": r.get("olusturma")}
    return df, rapor


def guncel_mi():
    """Parquet ve rapor var mı, ham Excel'den daha yeni mi?"""
    if not (os.path.exists(PARQUET_YOLU) and os.path.exists(RAPOR_YOLU)):
        return False
    if not os.path.exists(VERI_YOLU):
        return True                          # bulutta yalnızca Parquet olabilir
    # GitHub'dan indirilen dosyaların değişme zamanı rastgeledir (aynı saniyede yazılır); 5 dk tolerans tanınır.
    return os.path.getmtime(PARQUET_YOLU) + 300 >= os.path.getmtime(VERI_YOLU)


if __name__ == "__main__":
    yol = sys.argv[1] if len(sys.argv) > 1 else VERI_YOLU
    import time
    t0 = time.time()
    print(f"Okunuyor: {yol}")
    veri, rapor = veri_hazirla(yol)
    parquet_kaydet(veri, rapor)
    oz = rapor["ozet"]
    print(f"Ham {oz['ham_satir']:,} satır -> temiz {oz['temiz_satir']:,} satır, ciro {oz['ciro']:,.0f} TL")
    print(f"Yazıldı: {PARQUET_YOLU} ({os.path.getsize(PARQUET_YOLU) / 1e6:.1f} MB) · {time.time() - t0:.0f} sn")
