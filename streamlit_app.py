"""
B2B Satış ve Pazarlama Karar Paneli
İTÜ ISL 533 Veri Madenciliği · Ödev I

Ön işlenmiş veriyi (temiz_veri.parquet, bkz. veri_on_isle.py) okur, analiz eder ve Streamlit panelini çizer.
Parquet yoksa ya da ham Excel daha yeniyse ön işleme burada otomatik çalışır ve Parquet yeniden yazılır.

Çalıştırma:  python veri_on_isle.py        (veri değiştiğinde bir kez)
             streamlit run streamlit_app.py
"""
import html
import json
import os
import urllib.request

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.isotonic import IsotonicRegression

from streamlit import runtime

# Dosya ▶ düğmesiyle ya da "python streamlit_app2.py" ile çalıştırılırsa
# kendini "streamlit run" ile yeniden başlatır.
if __name__ == "__main__" and not runtime.exists():
    import sys
    from streamlit.web import cli
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(cli.main())

st.set_page_config(page_title="Satış Paneli", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

BASE = os.path.dirname(os.path.abspath(__file__))

TR_AYLAR = {1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
            7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"}
BASLANGIC = 4   # tahmin yönteminin kullandığı başlangıç geçmişi (ay)
TEST_AY = 3     # son 3 ay kontrol için ayrılır

# Doğrulama karşılaştırmasından sonra en başarılı bulunan modeli buraya yazın.
# Seçenekler: "Lojistik Regresyon", "Karar Ağacı", "Naive Bayes", "kNN", "Random Forest", "Gradient Boosting"
SECILEN_MODEL = "Lojistik Regresyon"


from veri_on_isle import (VERI_DOSYASI, VERI_YOLU, PARQUET_DOSYASI, PARQUET_YOLU, BILINMIYOR,  # noqa: E402
                          COVID_SINIRLARI, COVID_ETIKETLERI, veri_hazirla, parquet_kaydet, parquet_oku, guncel_mi)

# ===========================================================================
# B) ANALİZ
# ===========================================================================
def model_kur(ad):
    if ad == "Lojistik Regresyon":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    if ad == "Naive Bayes":
        return GaussianNB()
    if ad == "kNN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))
    if ad == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                      class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    if ad == "Gradient Boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                              class_weight="balanced", random_state=42)
    return DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, class_weight="balanced", random_state=42)


def tr_sayi(x, ondalik=0):
    """Sayıyı Türkçe biçimde yazar: 3.287.245 ya da 44,7"""
    metin = f"{x:,.{ondalik}f}"
    return metin.replace(",", "X").replace(".", ",").replace("X", ".")


def tr_ay(ts):
    return f"{TR_AYLAR[ts.month]} {ts.year}"


@st.cache_data(show_spinner="Veri yükleniyor…")
def veri_yukle(degisme_zamani):
    """Önce sıkıştırılmış Parquet'i okur (hızlı yol). Parquet yoksa ya da ham Excel daha yeniyse
    Excel'i ön işler ve Parquet'i yeniden yazar. degisme_zamani, dosya değişince önbelleği yeniler."""
    if guncel_mi():
        df, rapor = parquet_oku()
        kaynak = PARQUET_DOSYASI
    else:
        df, rapor = veri_hazirla(VERI_YOLU)
        try:
            parquet_kaydet(df, rapor)
        except OSError:
            pass
        kaynak = VERI_DOSYASI
    df["City"] = df["Il"]          # panel resmi il adlarıyla çalışır
    # Ürün karmasına göre beklenen iskonto: her ürünün tüm satışlardaki ağırlıklı iskonto oranı.
    # Fazla iskonto = verilen iskonto − bu ürünlerin olağan iskontosu (müşteriye özel ek indirim).
    g = df.groupby("StockCode")[["IskontoTutari", "ListeTutar"]].sum()
    urun_orani = (g["IskontoTutari"] / g["ListeTutar"]).fillna(0)
    df["BeklenenIskonto"] = df["ListeTutar"] * df["StockCode"].map(urun_orani)
    df["FazlaIskonto"] = df["IskontoTutari"] - df["BeklenenIskonto"]
    return df, rapor, kaynak


def filtrele(df, bas, son, sehirler, kategoriler):
    m = (df["InvoiceDate"].dt.date >= bas) & (df["InvoiceDate"].dt.date <= son)
    m &= df["City"].isin(sehirler)
    m &= df["KategoriGrubu"].isin(kategoriler)
    return df[m]


def kpi_hesapla(df):
    fatura_tutarlari = df.groupby("InvoiceNo")["Tutar"].sum()
    return {
        "ciro": df["Tutar"].sum(),
        "adet": df["Quantity"].sum(),
        "fatura": df["InvoiceNo"].nunique(),
        "musteri": df["CustomerID"].nunique(),
        "ort_fatura": fatura_tutarlari.mean() if len(fatura_tutarlari) else 0,
    }


def pareto_bilgisi(df):
    urun = df.groupby("Description")["Tutar"].sum().sort_values(ascending=False)
    if urun.sum() <= 0 or len(urun) == 0:
        return urun, None, None
    kumulatif = urun.cumsum() / urun.sum()
    n80 = int((kumulatif <= 0.8).sum()) + 1
    yuzde = n80 / len(urun) * 100
    return urun, n80, yuzde


def son_alimdan_ay(s):
    """Her ay için son alımdan bu yana geçen ay sayısı (o ayın alımı hariç)."""
    ciktilar, son_idx = [], -1
    for i, v in enumerate(s.values):
        ciktilar.append(i - son_idx if son_idx >= 0 else 99)
        if v > 0:
            son_idx = i
    return pd.Series(ciktilar, index=s.index)


def ses_fc(hist, h, alfa):
    y = np.asarray(hist, dtype=float)
    seviye = y[0]
    for v in y[1:]:
        seviye = alfa * v + (1 - alfa) * seviye
    return [seviye] * h


def holt_fc(hist, h, alfa, beta):
    y = np.asarray(hist, dtype=float)
    seviye = y[0]
    egim = (y[1] - y[0]) if len(y) > 1 else 0.0
    for gercek in y:
        yeni_seviye = alfa * gercek + (1 - alfa) * (seviye + egim)
        egim = beta * (yeni_seviye - seviye) + (1 - beta) * egim
        seviye = yeni_seviye
    return [seviye + (k + 1) * egim for k in range(h)]


def yontem_adaylari():
    adaylar = {
        "Naif (son ay)": lambda y, h: [float(y[-1])] * h,
        "Hareketli ortalama (3 ay)": lambda y, h: [float(np.mean(y[-3:]))] * h,
        "Hareketli ortalama (6 ay)": lambda y, h: [float(np.mean(y[-6:]))] * h,
        "Eğilim (drift)": lambda y, h: [
            float(y[-1] + (y[-1] - y[0]) / (len(y) - 1) * (k + 1)) for k in range(h)
        ],
    }
    for al in [0.1, 0.3, 0.5, 0.7, 0.9]:
        adaylar[f"Üstel düzleştirme (α={al})"] = (
            lambda al: lambda y, h: ses_fc(y, h, al)
        )(al)
    for al in [0.1, 0.3, 0.5, 0.8]:
        for be in [0.05, 0.1, 0.3]:
            adaylar[f"Holt (α={al}, β={be})"] = (
                lambda al, be: lambda y, h: holt_fc(y, h, al, be)
            )(al, be)
    return adaylar


def geri_test(y, f):
    tahminler, gercekler = [], []
    for t in range(BASLANGIC, len(y)):
        tahminler.append(f(y[:t], 1)[0])
        gercekler.append(y[t])
    return np.array(tahminler), np.array(gercekler)


def mape(gercek, tahmin):
    return float(np.mean(np.abs(gercek - tahmin) / np.maximum(np.abs(gercek), 1)) * 100)


def tahmin_secimi(seri_degerleri, ileri=3):
    """Yöntemleri doğrulama hatasına göre seçer. Tahmin ve hata payını döndürür."""
    y = np.asarray(seri_degerleri, dtype=float)
    if len(y) < BASLANGIC + TEST_AY + 1:
        return None
    sonuclar, geri = {}, {}
    for ad, f in yontem_adaylari().items():
        tah, ger = geri_test(y, f)
        sonuclar[ad] = mape(ger[:-TEST_AY], tah[:-TEST_AY])
        geri[ad] = tah
    en_iyi = min(sonuclar, key=sonuclar.get)
    f_en_iyi = yontem_adaylari()[en_iyi]
    hatalar = y[BASLANGIC:] - geri[en_iyi]
    return {
        "tahmin": f_en_iyi(y, ileri),
        "hata_std": float(np.std(hatalar)),
    }


def aylik_seri(df):
    return df.groupby("Ay")["Tutar"].sum().sort_index().asfreq("MS", fill_value=0)


OZELLIK_AD = {
    "ciro_onceki_ay": "Geçen ayın cirosu",
    "adet_onceki_ay": "Geçen ayın adedi",
    "fatura_onceki_ay": "Geçen ayki sipariş sayısı",
    "buyuk_var_onceki_ay": "Geçen ay büyük sipariş",
    "ciro_3ay_ort": "Son 3 ay ortalama ciro",
    "adet_3ay_ort": "Son 3 ay ortalama adet",
    "fatura_3ay_ort": "Son 3 ay ortalama sipariş",
    "buyuk_var_3ay_ort": "Son 3 ayda büyük sipariş sıklığı",
    "aktif_ay_gecmis": "Alım yapılan ay sayısı",
    "son_alimdan_ay": "Son alımdan beri geçen ay",
}


ETKEN_GRUP = {
    "Son 3 ayın satış hacmi": ["ciro_3ay_ort", "adet_3ay_ort", "fatura_3ay_ort"],
    "Geçen ayın satış hacmi": ["ciro_onceki_ay", "adet_onceki_ay", "fatura_onceki_ay"],
    "Büyük sipariş geçmişi": ["buyuk_var_onceki_ay", "buyuk_var_3ay_ort"],
    "Alım yapılan ay sayısı": ["aktif_ay_gecmis"],
    "Son alımdan beri geçen süre": ["son_alimdan_ay"],
}


def etken_grupla(etken):
    """Birbirine bağlı özelliklerin katkılarını anlamlı gruplarda toplar."""
    return pd.DataFrame({g: etken[k].sum(axis=1) for g, k in ETKEN_GRUP.items()})


@st.cache_data
def musteri_tahminleri(_veri, esik_yuzdelik, secilen_model, veri_kimligi):
    """Seçilen modelle müşteri bazında gelecek ay tahmini, geçmiş aylarda tahmin-gerçekleşen karşılaştırması,
    kalibre edilmiş olasılıklar ve tahmini etkileyen etkenler."""
    d = _veri.assign(Ay=_veri["InvoiceDate"].dt.to_period("M"))
    son_gercek = d["Ay"].max()

    fatura = d.groupby(["InvoiceNo", "CustomerID", "Ay"]).agg(
        adet=("Quantity", "sum"), tutar=("Tutar", "sum")
    ).reset_index()
    esik = fatura["adet"].quantile(esik_yuzdelik)
    fatura["buyuk"] = fatura["adet"] >= esik

    aylik = fatura.groupby(["CustomerID", "Ay"]).agg(
        ciro=("tutar", "sum"), adet=("adet", "sum"),
        fatura=("InvoiceNo", "count"), buyuk_fatura=("buyuk", "sum"),
    ).reset_index()
    aylik["buyuk_var"] = (aylik["buyuk_fatura"] > 0).astype(int)

    aylar = pd.period_range(d["Ay"].min(), son_gercek + 1, freq="M")
    izgara = pd.MultiIndex.from_product(
        [aylik["CustomerID"].unique(), aylar], names=["CustomerID", "Ay"]
    ).to_frame(index=False)
    tum = izgara.merge(aylik, on=["CustomerID", "Ay"], how="left").fillna(
        {"ciro": 0, "adet": 0, "fatura": 0, "buyuk_fatura": 0, "buyuk_var": 0}
    )
    tum = tum.sort_values(["CustomerID", "Ay"]).reset_index(drop=True)
    g = tum.groupby("CustomerID")
    for kol in ["ciro", "adet", "fatura", "buyuk_var"]:
        tum[f"{kol}_onceki_ay"] = g[kol].shift(1)
        tum[f"{kol}_3ay_ort"] = g[kol].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    tum["aktif_ay_gecmis"] = g["fatura"].transform(lambda s: (s.shift(1) > 0).cumsum())
    tum["son_alimdan_ay"] = g["fatura"].transform(son_alimdan_ay)
    ozellikler = list(OZELLIK_AD)
    tum = tum[tum["Ay"] > tum["Ay"].min()]
    tum = tum[tum["aktif_ay_gecmis"] > 0].copy()
    tablo = tum[tum["Ay"] <= son_gercek]
    aylar_sirali = sorted(tablo["Ay"].unique())

    # 1) Geriye dönük tahmin: her ay yalnızca önceki aylarla eğitilen modelle tahmin edilir
    parcalar = []
    for ay in aylar_sirali[2:]:
        gecmis = tablo[tablo["Ay"] < ay]
        if gecmis["buyuk_var"].nunique() < 2:
            continue
        m = model_kur(secilen_model).fit(gecmis[ozellikler], gecmis["buyuk_var"])
        o = tablo[tablo["Ay"] == ay]
        parcalar.append(pd.DataFrame({"CustomerID": o["CustomerID"].values, "Ay": ay,
                                      "gercek": o["buyuk_var"].values,
                                      "ham": m.predict_proba(o[ozellikler])[:, 1]}))
    bt = pd.concat(parcalar, ignore_index=True)

    # 2) Kalibrasyon: ham skoru, geçmişte gerçekleşme oranına çevir (yalnızca önceki aylarla)
    bt["p"] = np.nan
    bt_aylar = sorted(bt["Ay"].unique())
    for ay in bt_aylar[2:]:
        once = bt[bt["Ay"] < ay]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(once["ham"], once["gercek"])
        maske = bt["Ay"] == ay
        bt.loc[maske, "p"] = iso.predict(bt.loc[maske, "ham"])
    gecmis_tahmin = bt.dropna(subset=["p"]).copy()
    kalibre = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(bt["ham"], bt["gercek"])

    # 3) Seçim bittiği için gerçek kullanımda model tüm geçmişle yeniden eğitilir
    model = model_kur(secilen_model).fit(tablo[ozellikler], tablo["buyuk_var"])
    tum["p"] = kalibre.predict(model.predict_proba(tum[ozellikler])[:, 1])
    gelecek = son_gercek + 1
    tum.loc[tum["Ay"] == gelecek, "buyuk_var"] = np.nan
    gecmis_olasilik = tum[["CustomerID", "Ay", "p", "buyuk_var", "ciro"]].copy()

    # 4) Etkenler (lojistik regresyonda katsayı × standartlaştırılmış değer)
    simdi = tum[tum["Ay"] == gelecek].set_index("CustomerID")
    etken, genel_etki = None, None
    son_adim = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(son_adim, "coef_") and hasattr(model, "steps"):
        olcek = model.steps[0][1]
        z = (simdi[ozellikler] - olcek.mean_) / olcek.scale_
        etken = z * son_adim.coef_[0]
        genel_etki = pd.Series(son_adim.coef_[0], index=ozellikler)

    # 5) Müşteri profili
    son3 = d[d["Ay"] > son_gercek - 3]
    onc3 = d[(d["Ay"] > son_gercek - 6) & (d["Ay"] <= son_gercek - 3)]
    kat = (son3.groupby(["CustomerID", "Category"])["Tutar"].sum().reset_index()
           .sort_values(["CustomerID", "Tutar"], ascending=[True, False]))
    grp = (son3.groupby(["CustomerID", "KategoriGrubu"])["Tutar"].sum().reset_index()
           .sort_values(["CustomerID", "Tutar"], ascending=[True, False]))
    buyuk_aylik = aylik[aylik["buyuk_var"] == 1]
    profil = pd.DataFrame({
        "sehir": d.groupby("CustomerID")["City"].first(),
        "son_siparis": d.groupby("CustomerID")["InvoiceDate"].max(),
        "yillik_ciro": d.groupby("CustomerID")["Tutar"].sum(),
        "son3_ciro": son3.groupby("CustomerID")["Tutar"].sum(),
        "onceki3_ciro": onc3.groupby("CustomerID")["Tutar"].sum(),
        "ana_kategori": grp.groupby("CustomerID")["KategoriGrubu"].first(),
        "kategoriler": kat.groupby("CustomerID")["Category"].apply(lambda s: ", ".join(s.head(3))),
        "tipik_buyuk_ciro": buyuk_aylik.groupby("CustomerID")["ciro"].median(),
        "buyuk_ay_6": aylik[aylik["Ay"] > son_gercek - 6].groupby("CustomerID")["buyuk_var"].sum(),
    })
    profil[["son3_ciro", "onceki3_ciro", "buyuk_ay_6"]] = profil[["son3_ciro", "onceki3_ciro", "buyuk_ay_6"]].fillna(0)
    profil[["ana_kategori", "kategoriler"]] = profil[["ana_kategori", "kategoriler"]].fillna("—")

    onceki_p = tum[tum["Ay"] == son_gercek].set_index("CustomerID")["p"]
    sonuc = pd.DataFrame({"p": simdi["p"], "p_onceki": onceki_p.reindex(simdi.index),
                          "ciro_3ay_ort": simdi["ciro_3ay_ort"]}).join(profil)
    sonuc["beklenen_ciro"] = sonuc["p"] * sonuc["tipik_buyuk_ciro"].fillna(sonuc["ciro_3ay_ort"])
    sonuc["degerli"] = sonuc["yillik_ciro"] >= sonuc["yillik_ciro"].quantile(0.75)
    sonuc["buyukluk"] = pd.qcut(sonuc["yillik_ciro"].rank(method="first"), 4,
                                labels=["Küçük (alt %25)", "Orta-alt", "Orta-üst", "Büyük (üst %25)"]).astype(str)
    sonuc["bolge"] = sonuc["sehir"].map(lambda c: IL_BOLGE_ANAHTAR.get(il_anahtari(c), "Diğer"))
    # Son 12 ayın aylık cirosu ve büyük sipariş ayları (tablodaki küçük grafikler için)
    ay12 = pd.period_range(son_gercek - 11, son_gercek, freq="M")
    piv = aylik.pivot_table(index="CustomerID", columns="Ay", values="ciro", aggfunc="sum").reindex(columns=ay12).fillna(0)
    pivb = aylik.pivot_table(index="CustomerID", columns="Ay", values="buyuk_var", aggfunc="max").reindex(columns=ay12).fillna(0)
    sonuc["ciro_12"] = piv.reindex(sonuc.index).fillna(0).values.round(0).tolist()
    sonuc["buyuk_12"] = pivb.reindex(sonuc.index).fillna(0).values.astype(int).tolist()
    sonuc = sonuc.sort_values(["p", "beklenen_ciro"], ascending=False)
    sonuc["sira"] = range(1, len(sonuc) + 1)

    # 6) Kategori talebi: beklenen büyük sipariş cirosu müşterilerin büyük sipariş kategori payına dağıtılır
    satir = d.merge(fatura[["InvoiceNo", "CustomerID", "buyuk"]], on=["InvoiceNo", "CustomerID"], how="left")
    son6_buyuk = satir[(satir["buyuk"]) & (satir["Ay"] > son_gercek - 6)]
    pay = son6_buyuk.groupby(["CustomerID", "KategoriGrubu"])["Tutar"].sum()
    pay = (pay / pay.groupby(level=0).transform("sum")).rename("pay").reset_index()
    kat_talep = pay.merge(sonuc["beklenen_ciro"], left_on="CustomerID", right_index=True)
    kat_talep = (kat_talep["pay"] * kat_talep["beklenen_ciro"]).groupby(kat_talep["KategoriGrubu"]).sum()
    kat_son = satir[(satir["buyuk"]) & (satir["Ay"] == son_gercek)].groupby("KategoriGrubu")["Tutar"].sum()
    kat_onc3 = satir[(satir["buyuk"]) & (satir["Ay"] > son_gercek - 3)].groupby("KategoriGrubu")["Tutar"].sum() / 3
    kategori_talebi = pd.DataFrame({"beklenen": kat_talep, "son_ay": kat_son, "ort3": kat_onc3}).fillna(0)

    return {
        "musteri": sonuc.reset_index(), "gecmis_tahmin": gecmis_tahmin, "gecmis_olasilik": gecmis_olasilik,
        "etken": etken, "genel_etki": genel_etki, "kategori_talebi": kategori_talebi,
        "tahmin_ay": gelecek.to_timestamp(), "son_ay": son_gercek.to_timestamp(), "esik": float(esik),
    }


def kategori_tablosu(_df, kolon="KategoriGrubu", min_pay=0.005):
    """Kategorilerin son 3 ay payı, önceki 3 aya göre değişimi ve alıcı sayısı."""
    bitis = _df["InvoiceDate"].max()
    s3 = _df[_df["InvoiceDate"] > bitis - pd.DateOffset(months=3)]
    o3 = _df[(_df["InvoiceDate"] > bitis - pd.DateOffset(months=6))
             & (_df["InvoiceDate"] <= bitis - pd.DateOffset(months=3))]
    t = pd.DataFrame({
        "son3": s3.groupby(kolon)["Tutar"].sum(),
        "onceki3": o3.groupby(kolon)["Tutar"].sum(),
        "alici_son3": s3.groupby(kolon)["CustomerID"].nunique(),
        "alici_onceki3": o3.groupby(kolon)["CustomerID"].nunique(),
    }).fillna(0)
    t = t[t["son3"] + t["onceki3"] > 0]
    t["pay"] = t["son3"] / t["son3"].sum() * 100
    # Önceki dönemde çok küçük olan kategorilerde yüzde değişim anlamsız olur
    anlamli = (t["onceki3"] >= t["onceki3"].sum() * min_pay) & (t["onceki3"] > 0)
    t["degisim"] = np.where(anlamli, (t["son3"] - t["onceki3"]) / t["onceki3"].where(anlamli, 1) * 100, np.nan)
    medyan_pay = t.loc[anlamli, "pay"].median()

    def durum(r):
        if np.isnan(r["degisim"]):
            return "Yeni ya da çok küçük"
        buyuk = r["pay"] >= medyan_pay
        if buyuk and r["degisim"] >= 0:
            return "Güçlü ve büyüyor"
        if buyuk:
            return "Güçlü ama geriliyor"
        if r["degisim"] >= 0:
            return "Küçük ama büyüyor"
        return "Küçük ve geriliyor"

    t["durum"] = t.apply(durum, axis=1)
    return t.sort_values("son3", ascending=False), medyan_pay


# ---------- İl haritası yardımcıları ----------
GEOJSON_DOSYA = "tr_iller.geojson"
GEOJSON_URL = "https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json"
# Verideki eski ya da kısaltılmış il adları
IL_DUZELT = {"ICEL": "MERSIN", "K.MARAS": "KAHRAMANMARAS", "KMARAS": "KAHRAMANMARAS", "AFYONKARAHISAR": "AFYON"}
BOLGELER = {
    "Marmara": ["Balıkesir", "Bilecik", "Bursa", "Çanakkale", "Edirne", "İstanbul", "Kırklareli", "Kocaeli",
                "Sakarya", "Tekirdağ", "Yalova"],
    "Ege": ["Afyon", "Aydın", "Denizli", "İzmir", "Kütahya", "Manisa", "Muğla", "Uşak"],
    "Akdeniz": ["Adana", "Antalya", "Burdur", "Hatay", "Isparta", "Kahramanmaraş", "Mersin", "Osmaniye"],
    "İç Anadolu": ["Aksaray", "Ankara", "Çankırı", "Eskişehir", "Karaman", "Kayseri", "Kırıkkale", "Kırşehir",
                   "Konya", "Nevşehir", "Niğde", "Sivas", "Yozgat"],
    "Karadeniz": ["Amasya", "Artvin", "Bartın", "Bayburt", "Bolu", "Çorum", "Düzce", "Giresun", "Gümüşhane",
                  "Karabük", "Kastamonu", "Ordu", "Rize", "Samsun", "Sinop", "Tokat", "Trabzon", "Zonguldak"],
    "Doğu Anadolu": ["Ağrı", "Ardahan", "Bingöl", "Bitlis", "Elazığ", "Erzincan", "Erzurum", "Hakkari", "Iğdır",
                     "Kars", "Malatya", "Muş", "Tunceli", "Van"],
    "Güneydoğu Anadolu": ["Adıyaman", "Batman", "Diyarbakır", "Gaziantep", "Kilis", "Mardin", "Siirt",
                          "Şanlıurfa", "Şırnak"],
}
IL_BOLGE = {il: b for b, iller in BOLGELER.items() for il in iller}
IL_BOLGE_ANAHTAR = {}


def il_anahtari(ad):
    """Farklı yazımları (İZMİR, Izmir, IZMIR, K.MARAS) tek bir anahtara indirger."""
    t = str(ad).strip()
    for a, b in {"ı": "i", "İ": "I", "ğ": "g", "Ğ": "G", "ş": "s", "Ş": "S", "ü": "u", "Ü": "U",
                 "ö": "o", "Ö": "O", "ç": "c", "Ç": "C", " ": ""}.items():
        t = t.replace(a, b)
    t = t.upper()
    return IL_DUZELT.get(t, t)


IL_BOLGE_ANAHTAR.update({il_anahtari(il): b for il, b in IL_BOLGE.items()})


@st.cache_data
def il_haritasi_yukle():
    """İl sınırlarını önce klasördeki dosyadan, yoksa internetten okur (ve klasöre kaydeder)."""
    yol = os.path.join(BASE, GEOJSON_DOSYA)
    try:
        if os.path.exists(yol):
            with open(yol, encoding="utf-8") as f:
                return json.load(f)
        with urllib.request.urlopen(GEOJSON_URL, timeout=20) as yanit:
            geo = json.loads(yanit.read().decode("utf-8"))
        try:
            with open(yol, "w", encoding="utf-8") as f:
                json.dump(geo, f, ensure_ascii=False)
        except OSError:
            pass
        return geo
    except Exception:
        return None


def il_tablosu(_df, geo):
    """İl bazında ciro, adet, müşteri, sipariş ve son 3 ay büyümesi; 81 ilin tamamı (satış olmayanlar 0)."""
    d = _df.assign(anahtar=_df["City"].map(il_anahtari))
    bitis = d["InvoiceDate"].max()
    s3 = d["InvoiceDate"] > bitis - pd.DateOffset(months=3)
    o3 = (d["InvoiceDate"] > bitis - pd.DateOffset(months=6)) & ~s3
    t = d.groupby("anahtar").agg(
        ciro=("Tutar", "sum"), adet=("Quantity", "sum"),
        musteri=("CustomerID", "nunique"), siparis=("InvoiceNo", "nunique"),
    )
    t["son3"] = d[s3].groupby("anahtar")["Tutar"].sum()
    t["onceki3"] = d[o3].groupby("anahtar")["Tutar"].sum()
    iller = [f["properties"]["name"] for f in geo["features"]] if geo else sorted(IL_BOLGE)
    tum = pd.DataFrame({"İl": iller})
    tum["anahtar"] = tum["İl"].map(il_anahtari)
    tum = tum.merge(t, left_on="anahtar", right_index=True, how="left")
    tum[["ciro", "adet", "musteri", "siparis", "son3", "onceki3"]] = tum[
        ["ciro", "adet", "musteri", "siparis", "son3", "onceki3"]].fillna(0)
    tum["Bölge"] = tum["İl"].map(IL_BOLGE).fillna("Diğer")
    tum["pay"] = tum["ciro"] / tum["ciro"].sum() * 100 if tum["ciro"].sum() else 0
    tum["musteri_basi"] = np.where(tum["musteri"] > 0, tum["ciro"] / tum["musteri"].where(tum["musteri"] > 0, 1), 0)
    # Önceki dönemde çok küçük olan illerde yüzde değişim yanıltıcı olur
    anlamli = tum["onceki3"] >= max(tum["onceki3"].sum() * 0.002, 1)
    tum["buyume"] = np.where(anlamli, (tum["son3"] - tum["onceki3"]) / tum["onceki3"].where(anlamli, 1) * 100, np.nan)
    eslesmeyen = sorted(set(t.index) - set(tum["anahtar"]) - {il_anahtari(BILINMIYOR)})
    return tum.sort_values("ciro", ascending=False).reset_index(drop=True), eslesmeyen


def siniflandir(deger, sinif_sayisi=5):
    """Çarpık dağılımlı değerleri eşit sayıda il içeren sınıflara böler; 0 olanlar ayrı sınıf."""
    pozitif = deger[deger > 0]
    if pozitif.nunique() < 2:
        return pd.Series(np.where(deger > 0, "Satış var", "Satış yok"), index=deger.index), ["Satış var"]
    sinirlar = np.unique(np.quantile(pozitif, np.linspace(0, 1, sinif_sayisi + 1)))
    etiketler = []
    for i in range(len(sinirlar) - 1):
        alt, ust = sinirlar[i], sinirlar[i + 1]
        etiketler.append(f"{kisa_sayi(alt)} – {kisa_sayi(ust)}")
    sinif = pd.cut(deger, bins=sinirlar, labels=etiketler, include_lowest=True).astype(str)
    sinif[deger <= 0] = "Satış yok"
    return sinif, etiketler


@st.cache_data
def il_cizgileri(_geo):
    """Her il için sınır koordinatlarını (None ile ayrılmış parçalar) ve merkez noktasını hazırlar."""
    sonuc = {}
    for f in _geo["features"]:
        g = f["geometry"]
        parcalar = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        xs, ys, en_buyuk = [], [], None
        for cokgen in parcalar:
            dis = cokgen[0]
            xs += [p[0] for p in dis] + [None]
            ys += [p[1] for p in dis] + [None]
            if en_buyuk is None or len(dis) > len(en_buyuk):
                en_buyuk = dis
        # Merkez: en büyük parçanın köşelerinin ortalaması (etiket ve üzerine gelme noktası için yeterli)
        mx = float(np.mean([p[0] for p in en_buyuk]))
        my = float(np.mean([p[1] for p in en_buyuk]))
        sonuc[f["properties"]["name"]] = (xs, ys, mx, my)
    return sonuc


def turkiye_haritasi(harita_df, cizgiler, sira, renk_haritasi, baslik):
    """İlleri sınıf rengine göre boyar. Dış kaynak (CDN) gerektirmez; internet olmadan da çalışır."""
    fig = go.Figure()
    for sinif in sira:
        alt = harita_df[harita_df["Sınıf"] == sinif]
        xs, ys = [], []
        for il in alt["İl"]:
            if il in cizgiler:
                xs += cizgiler[il][0]
                ys += cizgiler[il][1]
        fig.add_trace(go.Scatter(
            x=xs if xs else [None], y=ys if ys else [None], mode="lines", fill="toself",
            fillcolor=renk_haritasi[sinif], line=dict(color="white", width=0.8),
            name=sinif, legendgroup=sinif, hoverinfo="skip", showlegend=bool(len(alt)),
        ))
    # Üzerine gelince bilgi: her ilin merkezinde görünmez bir nokta
    merkez = harita_df[harita_df["İl"].isin(cizgiler)].copy()
    merkez["mx"] = merkez["İl"].map(lambda i: cizgiler[i][2])
    merkez["my"] = merkez["İl"].map(lambda i: cizgiler[i][3])
    fig.add_trace(go.Scatter(
        x=merkez["mx"], y=merkez["my"], mode="markers", marker=dict(size=22, opacity=0),
        showlegend=False,
        customdata=merkez[["İl", "Bölge", "ciro", "pay", "musteri", "siparis", "adet", "buyume_metin"]].values,
        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                      "Ciro: %{customdata[2]:,.0f} (%%{customdata[3]:.1f})<br>"
                      "Müşteri: %{customdata[4]:,.0f} · Sipariş: %{customdata[5]:,.0f}<br>"
                      "Adet: %{customdata[6]:,.0f}<br>Son 3 ay değişim: %{customdata[7]}<extra></extra>",
    ))
    # En büyük 8 ilin adı haritada yazsın
    for _, rr in merkez.nlargest(8, "ciro").iterrows():
        fig.add_annotation(x=rr["mx"], y=rr["my"], text=rr["İl"], showarrow=False,
                           font=dict(size=10, color="#0f172a"), bgcolor="rgba(255,255,255,0.7)")
    fig.update_xaxes(visible=False, range=[25.6, 45.0])
    # Enlem ~39° civarında 1 derece boylam, 1 derece enlemden kısadır: oranı koruyarak çiz
    fig.update_yaxes(visible=False, range=[35.7, 42.3], scaleanchor="x", scaleratio=1 / np.cos(np.radians(39)))
    fig.update_layout(
        template=TEMA, height=480, margin=dict(l=0, r=0, t=10, b=0), dragmode=False,
        plot_bgcolor="rgba(0,0,0,0)", hovermode="closest",
        legend=dict(title=baslik, orientation="h", yanchor="top", y=0.0, x=0.5, xanchor="center"),
    )
    return fig


def kisa_sayi(x):
    if abs(x) >= 1e6:
        return f"{tr_sayi(x / 1e6, 1)} mn"
    if abs(x) >= 1e3:
        return f"{tr_sayi(x / 1e3, 0)} bin"
    return tr_sayi(x, 0)


# ---------- Veri ----------
if not (os.path.exists(PARQUET_YOLU) or os.path.exists(VERI_YOLU)):
    st.error(f"{PARQUET_DOSYASI} ya da {VERI_DOSYASI} bulunamadı. Veri dosyasını panelle aynı klasöre koyun.")
    st.stop()

veri_kimligi = tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (PARQUET_YOLU, VERI_YOLU))
veri, hazirlik, veri_kaynagi = veri_yukle(veri_kimligi)
if veri.empty:
    st.error("Veri dosyası boş ya da okunamadı.")
    st.stop()

# ---------- Tasarım dili (SaaS) ----------
MAVI, TURUNCU, GRI = "#4f46e5", "#f97316", "#94a3b8"
YESIL, KIRMIZI, MOR = "#10b981", "#ef4444", "#7c3aed"
ACIK_MAVI = "#c7d2fe"
TR_KISA = {1: "Oca", 2: "Şub", 3: "Mar", 4: "Nis", 5: "May", 6: "Haz",
           7: "Tem", 8: "Ağu", 9: "Eyl", 10: "Eki", 11: "Kas", 12: "Ara"}
# Grafikler: kart içinde beyaz zemin, ince ızgara, Inter yazı tipi ve Türkçe sayı ayırıcıları (1.234.567 / 12,5)
TEMA = go.layout.Template(pio.templates["plotly_white"])
TEMA.layout.update(
    separators=",.",
    font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=12, color="#334155"),
    colorway=[MAVI, TURUNCU, YESIL, "#06b6d4", MOR, "#f43f5e", "#eab308", "#64748b"],
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor="#0f172a", bordercolor="#0f172a", font=dict(color="#ffffff", size=12)),
    title=dict(font=dict(size=14, color="#0f172a")),
    legend=dict(font=dict(size=11, color="#475569")),
    margin=dict(l=8, r=8, t=36, b=8),
)
TEMA.layout.xaxis.update(gridcolor="#eef0f5", linecolor="#e2e8f0", zerolinecolor="#e2e8f0")
TEMA.layout.yaxis.update(gridcolor="#eef0f5", linecolor="#e2e8f0", zerolinecolor="#e2e8f0")


def tr_ay_ekseni(fig, aylar):
    """Zaman eksenindeki ay adlarını Türkçe yazar."""
    aylar = pd.to_datetime(pd.Series(aylar).drop_duplicates().sort_values())
    fig.update_xaxes(tickvals=aylar, ticktext=[f"{TR_KISA[a.month]} {a.year}" for a in aylar])
    return fig


def covid_golge(fig):
    """Grafiklerdeki COVID gölgeleri kaldırıldı; dönemler 'COVID etkisi' sayfasında ayrı incelenir."""
    return fig


def sayi_kolonu(baslik, ondalik=0):
    """Tablo sütununu sayı olarak tutar (sıralanabilir), tarayıcının diline göre biçimlendirir."""
    return st.column_config.NumberColumn(baslik, format="localized")


def yuvarla(tablo):
    """Tablodaki ondalıklı sayıları tam sayıya yuvarlar (yüzde sütunları hariç)."""
    tablo = tablo.copy()
    for c in tablo.select_dtypes("number").columns:
        ondalikli = "%" in c or "katı" in c
        tablo[c] = tablo[c].round(1 if ondalikli else 0)
    return tablo


def tablo(veri, **kw):
    st.dataframe(yuvarla(veri), **kw)


# ---------- Kenar çubuğu ----------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --bg: #f5f6fa; --kart: #ffffff; --kenar: #e6e8f0; --metin: #0f172a; --soluk: #64748b;
  --vurgu: #4f46e5; --vurgu2: #7c3aed; --golge: 0 1px 2px rgba(16,24,40,.04), 0 4px 16px rgba(16,24,40,.05);
}
html, body, [class*="css"], .stApp, button, input, textarea, select { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: var(--bg); }
.stMainBlockContainer, .block-container { padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1480px; }
header[data-testid="stHeader"] { background: rgba(255,255,255,.85); backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--kenar); }

/* --- Kenar çubuğu --- */
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--kenar); }
section[data-testid="stSidebar"] h4 {
  font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
  color: #94a3b8; margin: 1.2rem 0 .3rem 0; padding-top: .9rem; border-top: 1px solid #eef0f5;
}
.marka { display:flex; align-items:center; gap:.7rem; padding:.2rem 0 .6rem 0; }
.marka-logo { width:38px; height:38px; border-radius:11px; display:grid; place-items:center; color:#fff;
  font-weight:800; font-size:1rem; background: linear-gradient(135deg, var(--vurgu), var(--vurgu2));
  box-shadow: 0 6px 16px rgba(79,70,229,.35); }
.marka-ad { font-weight:750; font-size:1.02rem; color:var(--metin); line-height:1.1; }
.marka-alt { font-size:.72rem; color:var(--soluk); }

/* --- Üst başlık (hero) --- */
.hero { position:relative; overflow:hidden; border-radius:20px; padding:1.5rem 1.7rem; margin-bottom:1rem;
  background: radial-gradient(1200px 300px at 100% 0%, rgba(124,58,237,.55), transparent 60%),
              linear-gradient(120deg, #312e81 0%, #4f46e5 55%, #6d28d9 100%);
  color:#fff; box-shadow: 0 12px 32px rgba(79,70,229,.25); }
.hero-ust { font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; opacity:.75; font-weight:600; }
.hero h1 { color:#fff !important; font-size:1.75rem !important; font-weight:800 !important; margin:.25rem 0 .2rem 0 !important;
  padding:0 !important; letter-spacing:-.02em; }
.hero p { margin:0; opacity:.85; font-size:.92rem; }
.cipler { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.95rem; }
.cip { background: rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.22); border-radius:999px;
  padding:.28rem .75rem; font-size:.78rem; font-weight:500; backdrop-filter: blur(6px); }

/* --- Yönetici özeti --- */
.ozet { background:var(--kart); border:1px solid var(--kenar); border-radius:18px; padding:1.05rem 1.15rem 1.15rem;
  box-shadow: var(--golge); margin-bottom:1.2rem; }
.ozet-bas { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-bottom:.8rem; }
.ozet-baslik { font-weight:750; font-size:1rem; color:var(--metin); display:flex; align-items:center; gap:.5rem; }
.ozet-canli { font-size:.72rem; color:#059669; background:#ecfdf5; border:1px solid #a7f3d0; border-radius:999px;
  padding:.15rem .6rem; font-weight:600; }
.ozet-not { font-size:.76rem; color:var(--soluk); }
.ozet-grid { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:.8rem; }
@media (max-width: 1000px) { .ozet-grid { grid-template-columns: 1fr; } }
.ozet-oge { border-radius:14px; padding:.85rem .95rem; border:1px solid var(--kenar); background:#fbfbfe; position:relative; }
.ozet-oge::before { content:""; position:absolute; left:0; top:12px; bottom:12px; width:3px; border-radius:3px; background:var(--renk); }
.ozet-etiket { font-size:.66rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--renk); }
.ozet-sayi { font-size:1.22rem; font-weight:800; color:var(--metin); margin:.15rem 0 .25rem 0; letter-spacing:-.01em; }
.ozet-metin { font-size:.83rem; color:#475569; line-height:1.45; }
.ozet-aksiyon { margin-top:.45rem; font-size:.78rem; font-weight:600; color:var(--renk); }

/* --- KPI kartları --- */
div[data-testid="stMetric"] { background:var(--kart); border:1px solid var(--kenar); border-radius:16px;
  padding:.95rem 1.1rem; box-shadow: var(--golge); height:100%; }
div[data-testid="stMetricLabel"] p { font-size:.78rem !important; font-weight:600; color:var(--soluk); }
div[data-testid="stMetricValue"] { font-size:1.55rem !important; font-weight:750; color:var(--metin); letter-spacing:-.02em; }
div[data-testid="stMetricDelta"] { font-weight:600; }

/* --- Kartlar: grafik, tablo, çerçeveli kap --- */
div[data-testid="stPlotlyChart"] { background:var(--kart); border:1px solid var(--kenar); border-radius:16px;
  padding:.6rem .7rem .3rem .7rem; box-shadow: var(--golge); }
div[class*="st-key-kart"] { background:var(--kart); border-radius:18px !important; box-shadow: var(--golge);
  border:1px solid var(--kenar) !important; padding:1.1rem 1.2rem !important; }
div[class*="st-key-kart"] div[data-testid="stPlotlyChart"],
div[class*="st-key-kart"] div[data-testid="stMetric"] { box-shadow:none; }
div[class*="st-key-kart"] div[data-testid="stPlotlyChart"] { border:none; padding:0; }
.kart-baslik { font-weight:700; font-size:1.02rem; color:var(--metin); letter-spacing:-.01em; }
.kart-aciklama { font-size:.82rem; color:var(--soluk); margin-top:.1rem; margin-bottom:.35rem; }
div[data-testid="stExpander"] { background:var(--kart); border-radius:14px; border:1px solid var(--kenar); box-shadow: var(--golge); }
div[data-testid="stExpander"] details { border:none; }

/* --- Bölüm başlıkları --- */
.stMainBlockContainer h4 { font-size:1.08rem; font-weight:750; color:var(--metin); letter-spacing:-.01em;
  margin:1.5rem 0 .4rem 0; padding:0 0 0 .7rem; border-left:4px solid var(--vurgu); }
.stMainBlockContainer h5 { font-size:.95rem; font-weight:700; color:#1e293b; margin-top:.8rem; }

/* --- Öneri kartları (çapraz satış) --- */
.oneri { background:linear-gradient(180deg,#ffffff, #fafaff); border:1px solid var(--kenar); border-radius:16px;
  padding:1rem 1.05rem; box-shadow: var(--golge); height:100%; }
.oneri-sira { display:inline-grid; place-items:center; width:26px; height:26px; border-radius:8px; font-weight:800;
  font-size:.8rem; color:#fff; background:linear-gradient(135deg,var(--vurgu),var(--vurgu2)); }
.oneri-ad { font-weight:700; color:var(--metin); font-size:.95rem; margin:.55rem 0 .15rem 0; line-height:1.3;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; min-height:2.5em; }
.oneri-grup { font-size:.74rem; color:var(--soluk); }
.oneri-oran { font-size:1.9rem; font-weight:800; color:var(--vurgu); letter-spacing:-.02em; margin-top:.5rem; }
.oneri-oran span { font-size:.8rem; font-weight:600; color:var(--soluk); margin-left:.3rem; }
.oneri-alt { display:flex; gap:.4rem; flex-wrap:wrap; margin-top:.5rem; }
.rozet { font-size:.72rem; font-weight:600; border-radius:999px; padding:.15rem .55rem; background:#eef2ff; color:#4338ca; }
.rozet.yesil { background:#ecfdf5; color:#047857; }
.cumle { font-size:.95rem; color:#334155; margin:.3rem 0 .8rem 0; }
.cumle b { color:var(--metin); }
.bilgi-not { font-size:.78rem; color:var(--soluk); background:#f8fafc; border:1px dashed #cbd5e1; border-radius:12px;
  padding:.55rem .8rem; margin-top:.4rem; }

/* --- Düğmeler / seçim kontrolleri --- */
.stDownloadButton button, .stButton button { border-radius:10px; font-weight:600; }
div[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def kart(baslik=None, aciklama=None):
    """Beyaz, gölgeli SaaS kartı. Anahtar 'kart_' ile başladığı için CSS tarafından biçimlenir."""
    st.session_state["_kart_sayac"] = st.session_state.get("_kart_sayac", 0) + 1
    kap = st.container(border=True, key=f"kart_{st.session_state['_kart_sayac']}")
    if baslik:
        kap.markdown(f"<div class='kart-baslik'>{baslik}</div>"
                     + (f"<div class='kart-aciklama'>{aciklama}</div>" if aciklama else ""),
                     unsafe_allow_html=True)
    return kap


st.session_state["_kart_sayac"] = 0
FILTRE_ANAHTARLARI = ["donem", "tarih_ozel", "sehir_mod", "sehir_sec", "kat_mod", "kat_sec"]


def filtreleri_sifirla():
    for k in FILTRE_ANAHTARLARI:
        st.session_state.pop(k, None)


def tr_tarih(t):
    return f"{t.day} {TR_KISA[t.month]} {t.year}"


tarih_min = veri["InvoiceDate"].min().date()
tarih_max = veri["InvoiceDate"].max().date()
son_ay = pd.Timestamp(tarih_max).to_period("M")


def ay_geri(n):
    return max((son_ay - (n - 1)).to_timestamp().date(), tarih_min)


DONEMLER = {
    "Tüm dönem": (tarih_min, tarih_max),
    "Son 3 ay": (ay_geri(3), tarih_max),
    "Son 6 ay": (ay_geri(6), tarih_max),
}
for yil in sorted({tarih_min.year, tarih_max.year}, reverse=True):
    for q in range(4, 0, -1):
        q_bas = pd.Timestamp(year=yil, month=3 * q - 2, day=1)
        q_son = q_bas + pd.offsets.QuarterEnd(0)
        if q_son.date() >= tarih_min and q_bas.date() <= tarih_max:
            DONEMLER[f"{q}. çeyrek {yil}"] = (max(q_bas.date(), tarih_min), min(q_son.date(), tarih_max))
DONEMLER["Özel aralık"] = None

st.sidebar.markdown(
    "<div class='marka'><div class='marka-logo'>B2B</div><div><div class='marka-ad'>Satış Zekâsı</div>"
    "<div class='marka-alt'>Karar destek paneli</div></div></div>", unsafe_allow_html=True)

# --- Dönem ---
st.sidebar.markdown("#### 📅 Dönem")
donem = st.sidebar.selectbox("Hazır dönem", list(DONEMLER), key="donem", label_visibility="collapsed")
if DONEMLER[donem] is None:
    secim = st.sidebar.date_input(
        "Başlangıç ve bitiş", value=(tarih_min, tarih_max), min_value=tarih_min, max_value=tarih_max,
        format="DD.MM.YYYY", key="tarih_ozel",
    )
    # Kullanıcı yalnızca ilk tarihi seçtiğinde demet tek elemanlı gelir
    bas, son = (secim[0], secim[1]) if len(secim) == 2 else (secim[0], secim[0])
else:
    bas, son = DONEMLER[donem]
st.sidebar.caption(f"{tr_tarih(bas)} – {tr_tarih(son)} · {(son - bas).days + 1} gün")

# --- Şehir ---
sehir_ciro = veri.groupby("City")["Tutar"].sum().sort_values(ascending=False)
sehir_pay = sehir_ciro / sehir_ciro.sum() * 100
st.sidebar.markdown("#### 📍 Şehir")
sehir_mod = st.sidebar.segmented_control(
    "Şehir seçimi", ["Tümü", "En büyük 5", "Seç"], default="Tümü", key="sehir_mod",
    label_visibility="collapsed",
) or "Tümü"
if sehir_mod == "En büyük 5":
    sehirler = sehir_ciro.index[:5].tolist()
    st.sidebar.caption(", ".join(sehirler) + f" · cironun %{tr_sayi(sehir_pay.iloc[:5].sum(), 0)}'i")
elif sehir_mod == "Seç":
    sehirler = st.sidebar.multiselect(
        "Şehirler", sehir_ciro.index.tolist(), key="sehir_sec", label_visibility="collapsed",
        placeholder="Şehir arayın ya da seçin…",
        format_func=lambda c: f"{c} · %{tr_sayi(sehir_pay[c], 1)}",
    ) or sehir_ciro.index.tolist()
    if not st.session_state.get("sehir_sec"):
        st.sidebar.caption("Seçim yapılmadı; tüm şehirler gösteriliyor.")
else:
    sehirler = sehir_ciro.index.tolist()
    st.sidebar.caption(f"{len(sehirler)} şehrin tamamı")

# --- Kategori ---
kat_ciro = veri.groupby("KategoriGrubu")["Tutar"].sum().sort_values(ascending=False)
kat_pay = kat_ciro / kat_ciro.sum() * 100
st.sidebar.markdown("#### 🏷️ Kategori grubu")
kat_mod = st.sidebar.segmented_control(
    "Kategori seçimi", ["Tümü", "En büyük 3", "Seç"], default="Tümü", key="kat_mod",
    label_visibility="collapsed",
) or "Tümü"
if kat_mod == "En büyük 3":
    kategoriler = kat_ciro.index[:3].tolist()
    st.sidebar.caption(", ".join(kategoriler) + f" · cironun %{tr_sayi(kat_pay.iloc[:3].sum(), 0)}'i")
elif kat_mod == "Seç":
    kategoriler = st.sidebar.multiselect(
        "Kategoriler", kat_ciro.index.tolist(), key="kat_sec", label_visibility="collapsed",
        placeholder="Kategori grubu seçin…",
        format_func=lambda c: f"{c} · %{tr_sayi(kat_pay[c], 1)}",
    ) or kat_ciro.index.tolist()
    if not st.session_state.get("kat_sec"):
        st.sidebar.caption("Seçim yapılmadı; tüm gruplar gösteriliyor.")
else:
    kategoriler = kat_ciro.index.tolist()
    st.sidebar.caption(f"{len(kategoriler)} kategori grubunun tamamı")

st.sidebar.button("↺ Filtreleri sıfırla", on_click=filtreleri_sifirla, width="stretch")
st.sidebar.caption("Filtreler aksiyon özetini ve tahmin dışındaki tüm sayfaları etkiler. "
                   "Müşteri tahminleri her zaman tüm veriyle hazırlanır.")

# --- Veri bilgisi ---
with st.sidebar.expander("ℹ️ Veri hakkında"):
    guncelleme = pd.Timestamp(max(veri_kimligi), unit="s", tz="UTC").tz_convert("Europe/Istanbul")
    oz = hazirlik["ozet"]
    st.caption(
        f"Kaynak: {veri_kaynagi}"
        + (" (ön işlenmiş, hızlı yükleme)" if veri_kaynagi == PARQUET_DOSYASI else " (ham veri işlendi)") + "  \n"
        f"Son güncelleme: {guncelleme:%d.%m.%Y %H:%M}  \n"
        f"Kapsam: {tr_tarih(tarih_min)} – {tr_tarih(tarih_max)}  \n"
        f"Ham: {tr_sayi(oz['ham_satir'])} satır → temiz: {tr_sayi(oz['temiz_satir'])} satır  \n"
        f"Silinen: {tr_sayi(oz['silinen'])} · birleştirilen: {tr_sayi(oz['birlestirilen'])}  \n"
        f"{tr_sayi(veri['CustomerID'].nunique())} müşteri · {tr_sayi(veri['Category'].nunique())} alt kategori"
    )

df = filtrele(veri, bas, son, sehirler, kategoriler)

sehir_etiket = "Tüm şehirler" if len(sehirler) == len(sehir_ciro) else (
    sehirler[0] if len(sehirler) == 1 else f"{len(sehirler)} şehir")
kat_etiket = "Tüm kategori grupları" if len(kategoriler) == len(kat_ciro) else (
    kategoriler[0] if len(kategoriler) == 1 else f"{len(kategoriler)} kategori grubu")
# Filtrelenmiş veriye bağlı ağır hesapların önbellek anahtarı (DataFrame'i her seferinde özetlemek yerine)
FILTRE_ANAHTARI = (str(bas), str(son), tuple(sorted(sehirler)), tuple(sorted(kategoriler)), veri_kimligi)


def hero():
    cip = [f"📅 {tr_tarih(bas)} – {tr_tarih(son)}", f"📍 {sehir_etiket}", f"🏷️ {kat_etiket}",
           f"🧾 {tr_sayi(len(df))} kayıt"]
    st.markdown(
        "<div class='hero'><div class='hero-ust'>B2B Satış Zekâsı · 2020 işlem verisi</div>"
        "<h1>Satış ve Pazarlama Karar Paneli</h1>"
        "<p>Ciro, müşteri riski, çapraz satış, iskonto ve tedarik kararları tek ekranda.</p>"
        "<div class='cipler'>" + "".join(f"<span class='cip'>{html.escape(c)}</span>" for c in cip) + "</div></div>",
        unsafe_allow_html=True)


if df.empty:
    hero()
    st.warning("Seçilen filtrelerde veri yok. Filtreleri genişletin.")
    st.stop()

# ---------- Hesaplamalar ----------
seri = aylik_seri(df)
sec = tahmin_secimi(seri.values, ileri=3) if len(seri) >= BASLANGIC + TEST_AY + 1 else None
tah = musteri_tahminleri(veri, 0.90, SECILEN_MODEL, veri_kimligi)
tahmin_ay = tah["tahmin_ay"]
kat_tablo, medyan_pay = kategori_tablosu(df, "KategoriGrubu", min_pay=0)
bitis = df["InvoiceDate"].max()


def onceki_donem(bas, son):
    """Seçili dönemle aynı uzunluktaki bir önceki dönem (aynı şehir ve kategori filtreleriyle)."""
    gun = (son - bas).days + 1
    o_son = bas - pd.Timedelta(days=1)
    o_bas = o_son - pd.Timedelta(days=gun - 1)
    if o_bas < tarih_min:
        return None
    return filtrele(veri, o_bas, o_son, sehirler, kategoriler)


def degisim_metni(simdi, once):
    if once is None or not once:
        return None
    deg = (simdi / once - 1) * 100
    return f"{'-' if deg < 0 else '+'}%{tr_sayi(abs(deg), 1)}"


# =====================================================================
# GENEL BAKIŞ
# =====================================================================
def sayfa_genel():
    yonetici_ozeti()   # aksiyon özeti yalnızca bu sayfada
    k = kpi_hesapla(df)
    od = onceki_donem(bas, son)
    ko = kpi_hesapla(od) if od is not None and len(od) else None
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ciro", tr_sayi(k["ciro"]), degisim_metni(k["ciro"], ko and ko["ciro"]))
    c2.metric("Satılan adet", tr_sayi(k["adet"]), degisim_metni(k["adet"], ko and ko["adet"]))
    c3.metric("Sipariş", tr_sayi(k["fatura"]), degisim_metni(k["fatura"], ko and ko["fatura"]))
    c4.metric("Aktif müşteri", tr_sayi(k["musteri"]), degisim_metni(k["musteri"], ko and ko["musteri"]))
    c5.metric("Ortalama sipariş tutarı", tr_sayi(k["ort_fatura"]),
              degisim_metni(k["ort_fatura"], ko and ko["ort_fatura"]))
    if ko is not None:
        st.caption("Değişimler, seçili dönemle aynı uzunluktaki bir önceki döneme göredir.")

    # ---- İller ----
    st.markdown("#### İllere göre satış")
    geo = il_haritasi_yukle()
    iller, eslesmeyen = il_tablosu(df, geo)
    satisli = iller[iller["ciro"] > 0]
    bolge = iller.groupby("Bölge").agg(
        Ciro=("ciro", "sum"), Müşteri=("musteri", "sum"), il_sayisi=("İl", "size"),
        satisli_il=("ciro", lambda s_: int((s_ > 0).sum())), son3=("son3", "sum"), onceki3=("onceki3", "sum"),
    ).sort_values("Ciro", ascending=False)
    bolge["Pay (%)"] = bolge["Ciro"] / bolge["Ciro"].sum() * 100 if bolge["Ciro"].sum() else 0
    bolge["Değişim (%)"] = np.where(
        bolge["onceki3"] > 0, (bolge["son3"] / bolge["onceki3"].where(bolge["onceki3"] > 0, 1) - 1) * 100, np.nan)
    bolge["Kapsama"] = bolge["satisli_il"].astype(str) + " / " + bolge["il_sayisi"].astype(str)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Satış yapılan il", f"{len(satisli)} / {len(iller)}")
    if len(satisli):
        g2.metric(f"En büyük il: {satisli['İl'].iloc[0]}", f"%{tr_sayi(satisli['pay'].iloc[0], 1)}",
                  help="Toplam ciro içindeki payı.")
        g3.metric("İlk 5 ilin payı", f"%{tr_sayi(satisli['pay'].head(5).sum(), 1)}")
        g4.metric(f"En büyük bölge: {bolge.index[0]}", f"%{tr_sayi(bolge['Pay (%)'].iloc[0], 1)}")

    OLCULER = {
        "Ciro": ("ciro", "Ciro"),
        "Satış adedi": ("adet", "Adet"),
        "Müşteri sayısı": ("musteri", "Müşteri"),
        "Müşteri başına ciro": ("musteri_basi", "Müşteri başına ciro"),
        "Değişim (son 3 ay)": ("buyume", "Değişim"),
    }
    olcu = st.segmented_control("Harita", list(OLCULER), default="Ciro", key="harita_olcu",
                                label_visibility="collapsed") or "Ciro"
    kolon, olcu_ad = OLCULER[olcu]
    h1, h2 = st.columns([3, 2])
    with h1:
        if geo is None:
            st.warning(f"İl sınırları yüklenemedi. {GEOJSON_DOSYA} dosyasını panelle aynı klasöre koyun.")
        else:
            harita_df = iller.copy()
            if kolon == "buyume":
                sinirlar = [-np.inf, -25, -5, 5, 25, np.inf]
                etiketler = ["%25+ düşüş", "%5–25 düşüş", "±%5", "%5–25 artış", "%25+ artış"]
                renkler = ["#ef4444", "#fca5a5", "#e2e8f0", "#a5b4fc", "#4f46e5"]
                harita_df["Sınıf"] = pd.cut(harita_df["buyume"], bins=sinirlar, labels=etiketler).astype(str)
                harita_df.loc[harita_df["buyume"].isna(), "Sınıf"] = "Karşılaştırma yok"
                sira = etiketler + ["Karşılaştırma yok"]
                renk_haritasi = dict(zip(sira, renkler + ["#f1f5f9"]))
            else:
                harita_df["Sınıf"], etiketler = siniflandir(harita_df[kolon])
                mavi_tonlar = ["#e0e7ff", "#a5b4fc", "#6366f1", "#4338ca", "#312e81"][-len(etiketler):]
                sira = etiketler[::-1] + ["Satış yok"]
                renk_haritasi = dict(zip(etiketler, mavi_tonlar)) | {"Satış yok": "#f1f5f9"}
            harita_df["buyume_metin"] = harita_df["buyume"].map(
                lambda x: "—" if np.isnan(x) else f"%{tr_sayi(x, 1)}")
            fig = turkiye_haritasi(harita_df, il_cizgileri(geo), sira, renk_haritasi, olcu_ad)
            st.plotly_chart(fig, width="stretch", config={"scrollZoom": False, "displayModeBar": False})
            st.caption("Renk sınıfları eşit sayıda il içerir. Değişim: son 3 ay, önceki 3 aya göre.")
    with h2:
        tablo(bolge.reset_index()[["Bölge", "Ciro", "Pay (%)", "Değişim (%)", "Kapsama"]],
              hide_index=True, width="stretch",
              column_config={"Ciro": sayi_kolonu("Ciro"), "Pay (%)": sayi_kolonu("Pay (%)"),
                             "Değişim (%)": sayi_kolonu("Değişim (%)"),
                             "Kapsama": st.column_config.TextColumn("İl", help="Satış yapılan il / toplam il")})
        ilk10 = satisli.head(10).rename(columns={"ciro": "Ciro", "pay": "Pay (%)", "buyume": "Değişim (%)"})
        tablo(ilk10[["İl", "Ciro", "Pay (%)", "Değişim (%)"]], hide_index=True, width="stretch",
              column_config={"Ciro": sayi_kolonu("Ciro"), "Değişim (%)": sayi_kolonu("Değişim (%)"),
                             "Pay (%)": st.column_config.ProgressColumn(
                                 "Pay (%)", format="%.1f", min_value=0,
                                 max_value=float(max(ilk10["Pay (%)"].max(), 1)) if len(ilk10) else 1.0)})
    satissiz = iller.loc[iller["ciro"] <= 0, "İl"].tolist()
    if satissiz:
        st.caption("Seçili dönemde satış olmayan iller: " + ", ".join(satissiz))
    if eslesmeyen:
        st.caption("Haritada eşleşmeyen şehir adları: " + ", ".join(eslesmeyen))

    with st.expander("Tüm iller tablosu"):
        il_goster = satisli.rename(columns={
            "ciro": "Ciro", "pay": "Pay (%)", "adet": "Adet", "musteri": "Müşteri", "siparis": "Sipariş",
            "musteri_basi": "Müşteri başına ciro", "buyume": "Değişim (%)",
        })[["İl", "Bölge", "Ciro", "Pay (%)", "Değişim (%)", "Müşteri", "Sipariş", "Adet", "Müşteri başına ciro"]]
        tablo(il_goster, hide_index=True, width="stretch", height=380,
              column_config={c: sayi_kolonu(c) for c in il_goster.columns if c not in ("İl", "Bölge")})
        st.download_button("CSV olarak indir",
                           data=yuvarla(il_goster).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name="il_bazinda_satis.csv", mime="text/csv")

    with st.expander("İl ayrıntısı"):
        if len(satisli):
            il_sec = st.selectbox("İl", satisli["İl"].tolist(), key="il_sec")
            ir = satisli.set_index("İl").loc[il_sec]
            idf = df[df["City"].map(il_anahtari) == ir["anahtar"]]
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Ciro", tr_sayi(ir["ciro"]),
                      None if np.isnan(ir["buyume"]) else f"%{tr_sayi(ir['buyume'], 1)} son 3 ay")
            i2.metric("Cirodaki payı", f"%{tr_sayi(ir['pay'], 1)}")
            i3.metric("Müşteri", tr_sayi(ir["musteri"]))
            i4.metric("Müşteri başına ciro", tr_sayi(ir["musteri_basi"]))
            j1, j2 = st.columns(2)
            with j1:
                ia = idf.groupby("Ay", as_index=False)["Tutar"].sum()
                fig = px.bar(ia, x="Ay", y="Tutar", color_discrete_sequence=[MAVI],
                             labels={"Ay": "", "Tutar": "Ciro"}, title="Aylık ciro")
                fig.update_layout(template=TEMA, height=280, bargap=0.3)
                fig.update_yaxes(tickformat=",.0f")
                tr_ay_ekseni(fig, ia["Ay"])
                st.plotly_chart(fig, width="stretch")
            with j2:
                im = idf.groupby("CustomerID").agg(Ciro=("Tutar", "sum"), Sipariş=("InvoiceNo", "nunique")) \
                    .nlargest(8, "Ciro").reset_index()
                im["İldeki payı (%)"] = im["Ciro"] / ir["ciro"] * 100
                st.markdown("**En büyük müşteriler**")
                tablo(im.rename(columns={"CustomerID": "Müşteri"}), hide_index=True, width="stretch",
                      column_config={"Ciro": sayi_kolonu("Ciro"), "İldeki payı (%)": sayi_kolonu("İldeki payı (%)")})


    st.markdown("#### Aylık ciro ve gelecek ay beklentisi")
    a1, a2 = st.columns([3, 1])
    with a1:
        aylik_bar = pd.DataFrame({"Ay": seri.index, "Ciro": seri.values, "Seri": "Gerçekleşen"})
        if sec is not None:
            gelecek = seri.index[-1] + pd.DateOffset(months=1)
            aylik_bar = pd.concat([aylik_bar, pd.DataFrame(
                {"Ay": [gelecek], "Ciro": [sec["tahmin"][0]], "Seri": "Beklenen"})])
        fig = px.bar(aylik_bar, x="Ay", y="Ciro", color="Seri",
                     color_discrete_map={"Gerçekleşen": MAVI, "Beklenen": TURUNCU}, labels={"Ay": ""})
        if sec is not None:
            fig.update_traces(selector=dict(name="Beklenen"),
                              error_y=dict(type="data", array=[sec["hata_std"]], color=GRI, thickness=1.5))
        fig.update_layout(template=TEMA, height=340, bargap=0.3, legend_title="")
        fig.update_yaxes(tickformat=",.0f")
        covid_golge(fig)
        tr_ay_ekseni(fig, aylik_bar["Ay"])
        st.plotly_chart(fig, width="stretch")
    with a2:
        if sec is not None:
            tahmin, hata = sec["tahmin"][0], sec["hata_std"]
            st.metric(f"{tr_ay(gelecek)} beklenen ciro", tr_sayi(tahmin))
            st.metric("Olası aralık (milyon)",
                      f"{tr_sayi(max(tahmin - hata, 0) / 1e6, 2)} – {tr_sayi((tahmin + hata) / 1e6, 2)}",
                      help="Yöntemin geçmiş aylardaki tahmin hatalarının standart sapması kadar alt ve üst sınır.")
            st.metric(f"{tr_ay(seri.index[-1])} gerçekleşen", tr_sayi(seri.iloc[-1]))
        else:
            st.metric("Gelecek ay beklentisi", "—", help="Tahmin için en az 8 aylık veri gerekir.")

    st.markdown("#### Günlük ciro")
    gunluk = df.groupby(df["InvoiceDate"].dt.normalize())["Tutar"].sum().asfreq("D", fill_value=0)
    trend = pd.DataFrame({
        "Günlük ciro": gunluk,
        "7 günlük ortalama": gunluk.rolling(7, min_periods=1).mean(),
    }).reset_index().rename(columns={"InvoiceDate": "Tarih"}).melt(
        id_vars="Tarih", var_name="Seri", value_name="Ciro")
    fig = px.line(trend, x="Tarih", y="Ciro", color="Seri", labels={"Tarih": ""},
                  color_discrete_map={"Günlük ciro": "#c7d2fe", "7 günlük ortalama": TURUNCU})
    fig.update_layout(template=TEMA, height=300, legend_title="")
    fig.update_yaxes(tickformat=",.0f")
    covid_golge(fig)
    tr_ay_ekseni(fig, gunluk.index.to_period("M").to_timestamp())
    st.plotly_chart(fig, width="stretch")

    with st.expander("Veri hazırlama adımları"):
        tablo(hazirlik["adimlar"], hide_index=True, width="stretch",
              column_config={c: sayi_kolonu(c) for c in ["Önce (satır)", "Sonra (satır)", "Etkilenen"]})

# =====================================================================
# TAHMİN
# =====================================================================
def sayfa_tahmin():
    m = tah["musteri"].copy()
    gt = tah["gecmis_tahmin"]
    go_ = tah["gecmis_olasilik"]
    son_ay_p = pd.Period(tah["son_ay"], "M")
    son_ay_ad = tr_ay(tah["son_ay"])
    tahmin_ad = tr_ay(tahmin_ay)

    # Gruplar
    n_beklenen = int(round(m["p"].sum()))
    m["grup"] = "Diğer"
    m.loc[m.index[:n_beklenen], "grup"] = "Büyük sipariş beklenen"
    risk_maske = m["degerli"] & (m["buyuk_ay_6"] >= 1) & ((m["p"] - m["p_onceki"] <= -0.10) | (m["p"] < 0.15))
    m.loc[risk_maske & (m["grup"] == "Diğer"), "grup"] = "Risk altında"

    son_ay_satir = go_[go_["Ay"] == son_ay_p]
    son_gercek_sayi = int(son_ay_satir["buyuk_var"].sum())
    son_gercek_ciro = son_ay_satir.loc[son_ay_satir["buyuk_var"] == 1, "ciro"].sum()
    aylik_kars = gt.groupby("Ay").agg(Tahmin=("p", "sum"), Gerçekleşen=("gercek", "sum"))
    sapma = (aylik_kars["Tahmin"] - aylik_kars["Gerçekleşen"]).abs() / aylik_kars["Gerçekleşen"].clip(lower=1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Büyük sipariş veren müşteri", f"≈ {tr_sayi(m['p'].sum())}",
              f"{m['p'].sum() - son_gercek_sayi:+.0f} ({son_ay_ad}: {son_gercek_sayi})",
              help=f"Tek siparişte {tr_sayi(tah['esik'])} adet ve üzeri. Müşterilerin olasılıklarının toplamı.")
    k2.metric("Büyük sipariş cirosu", tr_sayi(m["beklenen_ciro"].sum()),
              degisim_metni(m["beklenen_ciro"].sum(), son_gercek_ciro),
              help="Her müşterinin olasılığı × büyük sipariş verdiği aylardaki tipik cirosu.")
    k3.metric("Aylık tahmin sapması", f"%{tr_sayi(sapma.mean() * 100, 0)}",
              help="Geçmiş aylarda, model o ayı önceden tahmin ettiğinde beklenen ve gerçekleşen müşteri "
                   "sayısı arasındaki ortalama fark.")
    k4.metric("Risk altındaki değerli müşteri", tr_sayi(int((m["grup"] == "Risk altında").sum())),
              f"yıllık ciro {kisa_sayi(m.loc[m['grup'] == 'Risk altında', 'yillik_ciro'].sum())}",
              delta_color="off",
              help="Yıllık cirosu üst %25'te, son 6 ayda büyük sipariş vermiş; olasılığı 10 puandan fazla düşmüş "
                   "ya da %15'in altında olan müşteriler.")

    # --- Tahmin ve gerçekleşen / olasılık güvenilirliği ---
    t1, t2 = st.columns([3, 2])
    with t1:
        st.markdown("##### Tahmin ve gerçekleşen")
        ak = aylik_kars.reset_index()
        ak["Ay"] = ak["Ay"].dt.to_timestamp()
        ak = ak.melt(id_vars="Ay", var_name="Seri", value_name="Müşteri")
        ileri = pd.DataFrame({"Ay": [tahmin_ay], "Seri": ["Tahmin"], "Müşteri": [m["p"].sum()]})
        fig = px.line(pd.concat([ak, ileri]), x="Ay", y="Müşteri", color="Seri", markers=True,
                      color_discrete_map={"Gerçekleşen": MAVI, "Tahmin": TURUNCU},
                      labels={"Ay": "", "Müşteri": "Büyük sipariş veren müşteri"})
        fig.update_traces(line_width=2, marker_size=8)
        fig.update_traces(selector=dict(name="Tahmin"), line_dash="dot")
        fig.update_layout(template=TEMA, height=320, legend_title="", legend=dict(orientation="h", y=1.12, x=0))
        covid_golge(fig)
        tr_ay_ekseni(fig, pd.concat([ak, ileri])["Ay"])
        st.plotly_chart(fig, width="stretch")
    with t2:
        st.markdown("##### Olasılık ve gerçekleşme")
        dilim = pd.cut(gt["p"], [0, 0.1, 0.2, 0.35, 0.5, 1.0],
                       labels=["%0–10", "%10–20", "%20–35", "%35–50", "%50+"], include_lowest=True)
        kal = gt.groupby(dilim, observed=True).agg(Gerçekleşen=("gercek", "mean"), Tahmin=("p", "mean"),
                                                   n=("gercek", "size")).reset_index()
        kal = kal.rename(columns={"p": "Dilim"})
        kal.columns = ["Dilim", "Gerçekleşen", "Tahmin", "n"]
        fig = go.Figure()
        fig.add_bar(x=kal["Dilim"], y=kal["Gerçekleşen"] * 100, name="Gerçekleşen oran", marker_color=MAVI,
                    customdata=kal["n"], hovertemplate="%{x}: %%{y:.0f} gerçekleşti (%{customdata} gözlem)<extra></extra>")
        fig.add_scatter(x=kal["Dilim"], y=kal["Tahmin"] * 100, name="Ortalama tahmin", mode="markers",
                        marker=dict(color=TURUNCU, size=11, line=dict(color="white", width=2)),
                        hovertemplate="%{x}: tahmin %%{y:.0f}<extra></extra>")
        fig.update_layout(template=TEMA, height=320, bargap=0.35, legend=dict(orientation="h", y=1.12, x=0),
                          yaxis_title="%", xaxis_title="Tahmin edilen olasılık")
        st.plotly_chart(fig, width="stretch")

    # --- Segmentlere göre ---
    st.markdown("##### Segmentlere göre tahmin")
    SEGMENT = {"Şehir": "sehir", "Bölge": "bolge", "Müşteri büyüklüğü": "buyukluk", "Kategori grubu": "ana_kategori"}
    seg_ad = st.segmented_control("Segment", list(SEGMENT), default="Şehir", key="seg",
                                  label_visibility="collapsed") or "Şehir"
    seg = SEGMENT[seg_ad]
    seg_map = m.set_index("CustomerID")[seg]
    son_ay_satir = son_ay_satir.assign(seg=son_ay_satir["CustomerID"].map(seg_map))
    gts = gt.assign(seg=gt["CustomerID"].map(seg_map))
    st_tab = pd.DataFrame({
        "Müşteri": m.groupby(seg).size().astype(float),
        "Son ay": son_ay_satir.groupby("seg")["buyuk_var"].sum(),
        "Tahmin": m.groupby(seg)["p"].sum(),
        "Beklenen ciro": m.groupby(seg)["beklenen_ciro"].sum(),
        "gecmis_tahmin": gts.groupby("seg")["p"].sum(),
        "gecmis_gercek": gts.groupby("seg")["gercek"].sum(),
    }).fillna(0)
    st_tab["Geçmiş isabet (%)"] = np.where(
        st_tab["gecmis_gercek"] > 0,
        st_tab["gecmis_tahmin"] / st_tab["gecmis_gercek"].where(st_tab["gecmis_gercek"] > 0, 1) * 100, np.nan)
    st_tab = st_tab.sort_values("Tahmin", ascending=False)
    s1, s2 = st.columns([3, 2])
    with s1:
        st_tab["Fark"] = st_tab["Tahmin"] - st_tab["Son ay"]
        ilk = st_tab.head(12).sort_values("Fark")
        fig = go.Figure(go.Bar(
            x=ilk["Fark"], y=ilk.index.astype(str), orientation="h",
            marker_color=[MAVI if v >= 0 else "#ef4444" for v in ilk["Fark"]],
            customdata=ilk[["Son ay", "Tahmin"]].values,
            hovertemplate="%{y}<br>" + son_ay_ad + ": %{customdata[0]:.0f}<br>"
                          + tahmin_ad + ": %{customdata[1]:.1f}<br>Fark: %{x:+.1f}<extra></extra>"))
        fig.add_vline(x=0, line_color=GRI)
        fig.update_layout(template=TEMA, height=max(300, 30 * len(ilk) + 80), bargap=0.3,
                          xaxis_title=f"Büyük sipariş veren müşteri: {tahmin_ad} tahmini − {son_ay_ad}")
        st.plotly_chart(fig, width="stretch")
    with s2:
        tablo(st_tab.reset_index().rename(columns={seg: seg_ad, "index": seg_ad})[
                  [seg_ad, "Müşteri", "Son ay", "Tahmin", "Beklenen ciro", "Geçmiş isabet (%)"]],
              hide_index=True, width="stretch", height=max(300, 30 * min(len(st_tab), 12) + 80),
              column_config={
                  "Son ay": st.column_config.NumberColumn(son_ay_ad, format="localized",
                                                          help="Büyük sipariş veren müşteri sayısı"),
                  "Tahmin": st.column_config.NumberColumn(tahmin_ad, format="localized"),
                  "Beklenen ciro": sayi_kolonu("Beklenen ciro"),
                  "Geçmiş isabet (%)": st.column_config.NumberColumn(
                      "Tahmin / gerçek (%)", format="localized",
                      help="Geçmiş aylarda tahmin edilen toplamın gerçekleşene oranı. 100 = tam isabet, "
                           "100 üstü = fazla tahmin, 100 altı = eksik tahmin."),
              })

    # --- Kategori talebi ve etkenler ---
    c1_, c2_ = st.columns([3, 2])
    with c1_:
        st.markdown("##### Kategori grubuna göre beklenen büyük sipariş cirosu")
        kt_ = tah["kategori_talebi"].sort_values("beklenen", ascending=False).head(12).iloc[::-1]
        fig = go.Figure()
        fig.add_bar(y=kt_.index, x=kt_["ort3"], orientation="h", name="Son 3 ay ortalaması", marker_color="#c7d2fe",
                    hovertemplate="%{y}: %{x:,.0f}<extra>Son 3 ay ort.</extra>")
        fig.add_bar(y=kt_.index, x=kt_["beklenen"], orientation="h", name=f"{tahmin_ad} beklenen",
                    marker_color=TURUNCU, hovertemplate="%{y}: %{x:,.0f}<extra>Beklenen</extra>")
        fig.update_layout(template=TEMA, height=430, barmode="group", bargap=0.25, bargroupgap=0.05,
                          yaxis=dict(automargin=True),
                          legend=dict(orientation="h", y=1.08, x=0))
        fig.update_xaxes(tickformat=",.0f")
        st.plotly_chart(fig, width="stretch")
    with c2_:
        st.markdown("##### Tahmini belirleyen etkenler")
        if tah["etken"] is not None:
            onem = etken_grupla(tah["etken"]).abs().mean().sort_values()
            onem = onem / onem.sum() * 100
            fig = px.bar(x=onem.values, y=onem.index, orientation="h",
                         color_discrete_sequence=[MAVI], labels={"x": "Tahmindeki ağırlık (%)", "y": ""})
            fig.update_traces(hovertemplate="%{y}: %%{x:.1f}<extra></extra>")
            fig.update_layout(template=TEMA, height=430, bargap=0.35)
            st.plotly_chart(fig, width="stretch")

    # --- Müşteri listesi ---
    st.markdown("##### Müşteriler")
    l1, l2, l3 = st.columns([2, 1, 1])
    gorunum = l1.segmented_control("Liste", ["Büyük sipariş beklenen", "Risk altında", "Tüm müşteriler"],
                                   default="Büyük sipariş beklenen", key="musteri_gorunum",
                                   label_visibility="collapsed") or "Büyük sipariş beklenen"
    liste = m if gorunum == "Tüm müşteriler" else m[m["grup"] == gorunum]
    if gorunum == "Risk altında":
        liste = liste.sort_values("yillik_ciro", ascending=False)
    sehir_f = l2.multiselect("Şehir", sorted(liste["sehir"].dropna().unique()), placeholder="Tüm şehirler",
                             key="m_sehir", label_visibility="collapsed")
    kat_f = l3.multiselect("Kategori", sorted(liste["ana_kategori"].unique()), placeholder="Tüm kategoriler",
                           key="m_kat", label_visibility="collapsed")
    if sehir_f:
        liste = liste[liste["sehir"].isin(sehir_f)]
    if kat_f:
        liste = liste[liste["ana_kategori"].isin(kat_f)]
    goster = pd.DataFrame({
        "Sıra": liste["sira"], "Müşteri": liste["CustomerID"], "Şehir": liste["sehir"],
        "Olasılık": (liste["p"] * 100).round(0),
        "Değişim": ((liste["p"] - liste["p_onceki"]).fillna(0) * 100).round(0),
        "Beklenen ciro": liste["beklenen_ciro"],
        "Aylık ciro (12 ay)": liste["ciro_12"],
        "Büyük sipariş ayları": liste["buyuk_12"],
        "Yıllık ciro": liste["yillik_ciro"],
        "Son sipariş": liste["son_siparis"].dt.strftime("%d.%m.%Y"),
        "Kategoriler": liste["kategoriler"],
    })
    st.dataframe(yuvarla(goster.drop(columns=["Aylık ciro (12 ay)", "Büyük sipariş ayları"]))
                 .assign(**{"Aylık ciro (12 ay)": goster["Aylık ciro (12 ay)"],
                            "Büyük sipariş ayları": goster["Büyük sipariş ayları"]})[goster.columns],
                 hide_index=True, width="stretch", height=420, column_config={
        "Olasılık": st.column_config.ProgressColumn(
            f"{tahmin_ad} olasılığı", min_value=0, max_value=100, format="%%%d",
            help="Büyük sipariş verme olasılığı. Geçmişte bu olasılık düzeyindeki müşterilerin gerçekleşme oranına göre kalibre edildi."),
        "Değişim": st.column_config.NumberColumn("Değişim", format="%+d puan",
                                                 help=f"{son_ay_ad} için hesaplanan olasılığa göre fark."),
        "Beklenen ciro": sayi_kolonu("Beklenen ciro"),
        "Aylık ciro (12 ay)": st.column_config.LineChartColumn("Aylık ciro (12 ay)", width="small"),
        "Büyük sipariş ayları": st.column_config.BarChartColumn("Büyük sipariş (12 ay)", y_min=0, y_max=1,
                                                                width="small"),
        "Yıllık ciro": sayi_kolonu("Yıllık ciro"),
    })
    st.download_button("CSV olarak indir",
                       data=yuvarla(goster.drop(columns=["Aylık ciro (12 ay)", "Büyük sipariş ayları"]))
                       .to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name=f"tahmin_{tahmin_ay:%Y_%m}.csv", mime="text/csv")

    # --- Müşteri analizi ---
    st.markdown("##### Müşteri analizi")
    if len(liste):
        kart = st.selectbox("Müşteri", liste["CustomerID"].tolist(), key="kart_musteri",
                            format_func=lambda c: f"{c} · {liste.loc[liste['CustomerID'] == c, 'sehir'].iloc[0]}",
                            label_visibility="collapsed")
        kr = m.set_index("CustomerID").loc[kart]
        n1, n2, n3, n4 = st.columns(4)
        n1.metric(f"{tahmin_ad} olasılığı", f"%{tr_sayi(kr['p'] * 100)}",
                  None if np.isnan(kr["p_onceki"]) else f"{int(round((kr['p'] - kr['p_onceki']) * 100)):+d} puan")
        n2.metric("Beklenen ciro", tr_sayi(kr["beklenen_ciro"]))
        n3.metric("Son 3 ay ciro", tr_sayi(kr["son3_ciro"]), degisim_metni(kr["son3_ciro"], kr["onceki3_ciro"]))
        n4.metric("Yıllık ciro", tr_sayi(kr["yillik_ciro"]))
        q1, q2 = st.columns([3, 2])
        with q1:
            gm = go_[go_["CustomerID"] == kart].copy()
            gm["Ay"] = gm["Ay"].dt.to_timestamp()
            fig = go.Figure()
            fig.add_scatter(x=gm["Ay"], y=gm["p"] * 100, mode="lines", line=dict(color=GRI, width=2),
                            name="Olasılık", hovertemplate="%{x|%m.%Y}: %%{y:.0f}<extra></extra>")
            oldu = gm[gm["buyuk_var"] == 1]
            olmadi = gm[gm["buyuk_var"] == 0]
            fig.add_scatter(x=oldu["Ay"], y=oldu["p"] * 100, mode="markers", name="Büyük sipariş verdi",
                            marker=dict(color=MAVI, size=11, line=dict(color="white", width=2)),
                            hovertemplate="%{x|%m.%Y}: verdi<extra></extra>")
            fig.add_scatter(x=olmadi["Ay"], y=olmadi["p"] * 100, mode="markers", name="Vermedi",
                            marker=dict(color="white", size=10, line=dict(color=MAVI, width=2)),
                            hovertemplate="%{x|%m.%Y}: vermedi<extra></extra>")
            ileri_m = gm[gm["buyuk_var"].isna()]
            fig.add_scatter(x=ileri_m["Ay"], y=ileri_m["p"] * 100, mode="markers", name="Tahmin",
                            marker=dict(color=TURUNCU, size=13, symbol="diamond", line=dict(color="white", width=2)),
                            hovertemplate="%{x|%m.%Y}: %%{y:.0f}<extra>Tahmin</extra>")
            fig.update_layout(template=TEMA, height=320, yaxis=dict(title="Olasılık (%)", range=[0, 100]),
                              legend=dict(orientation="h", y=1.12, x=0), title=None)
            tr_ay_ekseni(fig, gm["Ay"])
            st.plotly_chart(fig, width="stretch")
        with q2:
            if tah["etken"] is not None and kart in tah["etken"].index:
                e = etken_grupla(tah["etken"].loc[[kart]]).iloc[0].sort_values()
                fig = go.Figure(go.Bar(
                    x=e.values, y=e.index, orientation="h",
                    marker_color=[MAVI if v > 0 else "#ef4444" for v in e.values],
                    hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
                fig.add_vline(x=0, line_color=GRI)
                fig.update_layout(template=TEMA, height=320, bargap=0.3,
                                  title=dict(text="Bu müşterinin olasılığını belirleyenler", font=dict(size=14)),
                                  xaxis_title="← düşürüyor · artırıyor →")
                st.plotly_chart(fig, width="stretch")

# =====================================================================
# KATEGORİLER
# =====================================================================
def sayfa_kategori():
    st.markdown("#### Kategori gruplarının konumu")
    harita = kat_tablo.dropna(subset=["degisim"]).reset_index().rename(
        columns={"index": "Grup", "KategoriGrubu": "Grup"})
    harita = harita[~harita["Grup"].isin([BILINMIYOR, "Diğer"])]
    k1_, k2_ = st.columns([3, 2])
    with k1_:
        if harita.empty:
            st.caption("Karşılaştırma için en az 6 aylık veri gerekir.")
        else:
            harita["degisim_goster"] = harita["degisim"].clip(-100, 200)
            fig = px.scatter(
                harita, x="pay", y="degisim_goster", size="alici_son3", size_max=34, log_x=True,
                text="Grup", color_discrete_sequence=[MAVI],
                custom_data=["son3", "degisim", "alici_son3"],
                labels={"pay": "Son 3 ayın ciro payı", "degisim_goster": "Önceki 3 aya göre değişim (%)"},
            )
            # Etiketler üst üste binmesin: değişime göre sıralayıp üst/alt dönüşümlü yerleştir
            sira_ = harita["degisim_goster"].rank(method="first").astype(int)
            fig.update_traces(
                textposition=["top center" if i % 2 else "bottom center" for i in sira_],
                textfont=dict(size=11, color="#0f172a"),
                marker=dict(opacity=0.75, line=dict(width=2, color="white")),
                hovertemplate="<b>%{text}</b><br>Son 3 ay ciro: %{customdata[0]:,.0f}<br>"
                              "Değişim: %%{customdata[1]:.1f}<br>Alıcı: %{customdata[2]}<extra></extra>",
            )
            fig.add_vline(x=medyan_pay, line_dash="dash", line_color=GRI)
            fig.add_hline(y=0, line_dash="dash", line_color=GRI)
            fig.update_layout(template=TEMA, height=460)
            fig.update_xaxes(tickvals=[0.3, 0.5, 1, 2, 5, 10, 20, 50],
                             ticktext=["%0,3", "%0,5", "%1", "%2", "%5", "%10", "%20", "%50"])
            st.plotly_chart(fig, width="stretch")
    with k2_:
        kt = kat_tablo.reset_index().rename(columns={
            "index": "Grup", "KategoriGrubu": "Grup", "son3": "Son 3 ay", "onceki3": "Önceki 3 ay",
            "pay": "Pay (%)", "degisim": "Değişim (%)", "alici_son3": "Alıcı",
        })[["Grup", "Son 3 ay", "Önceki 3 ay", "Değişim (%)", "Pay (%)", "Alıcı"]]
        tablo(kt, hide_index=True, width="stretch", height=460,
              column_config={c: sayi_kolonu(c) for c in kt.columns if c != "Grup"})

    st.markdown("#### Grup ayrıntısı")
    grup_sec = st.selectbox("Grup", kat_tablo.index.tolist(), key="kat_detay", label_visibility="collapsed")
    kr = kat_tablo.loc[grup_sec]
    kd = df[df["KategoriGrubu"] == grup_sec]
    kd_s3 = kd[kd["InvoiceDate"] > bitis - pd.DateOffset(months=3)]
    kd_o3 = kd[(kd["InvoiceDate"] > bitis - pd.DateOffset(months=6))
               & (kd["InvoiceDate"] <= bitis - pd.DateOffset(months=3))]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Son 3 ay ciro", tr_sayi(kr["son3"]), degisim_metni(kr["son3"], kr["onceki3"]))
    d2.metric("Cirodaki payı", f"%{tr_sayi(kr['pay'], 1)}")
    d3.metric("Alıcı müşteri (son 3 ay)", tr_sayi(kr["alici_son3"]),
              f"{int(kr['alici_son3'] - kr['alici_onceki3']):+d}")
    d4.metric("Ortalama iskonto", f"%{tr_sayi(kd_s3['IskontoOrani'].mean() * 100 if len(kd_s3) else 0, 1)}",
              help="Son 3 aydaki satırların ortalama iskonto oranı (1 − net fiyat / liste fiyatı).")

    kat_seri = aylik_seri(kd)
    kat_sonuc = tahmin_secimi(kat_seri.values, ileri=3)
    cizim = pd.DataFrame({"Ay": kat_seri.index, "Ciro": kat_seri.values, "Seri": "Gerçekleşen"})
    if kat_sonuc is not None:
        cizim = pd.concat([cizim, pd.DataFrame({"Ay": [kat_seri.index[-1] + pd.DateOffset(months=1)],
                                                "Ciro": [kat_sonuc["tahmin"][0]], "Seri": "Beklenen"})])
    g1_, g2_ = st.columns([3, 2])
    with g1_:
        fig = px.bar(cizim, x="Ay", y="Ciro", color="Seri", labels={"Ay": ""},
                     color_discrete_map={"Gerçekleşen": MAVI, "Beklenen": TURUNCU})
        fig.update_layout(template=TEMA, height=340, legend_title="", bargap=0.3,
                          legend=dict(orientation="h", y=1.2, x=0))
        fig.update_yaxes(tickformat=",.0f")
        covid_golge(fig)
        tr_ay_ekseni(fig, cizim["Ay"])
        st.plotly_chart(fig, width="stretch")
    with g2_:
        alt, _ = kategori_tablosu(kd, "Category")
        at_ = alt.reset_index().rename(columns={"Category": "Alt kategori", "index": "Alt kategori",
                                                "son3": "Son 3 ay", "degisim": "Değişim (%)", "pay": "Grup içi pay (%)"})
        tablo(at_[["Alt kategori", "Son 3 ay", "Değişim (%)", "Grup içi pay (%)"]], hide_index=True,
              width="stretch", height=320,
              column_config={c: sayi_kolonu(c) for c in ["Son 3 ay", "Değişim (%)", "Grup içi pay (%)"]})

    e1, e2, e3 = st.columns(3)
    with e1:
        km = pd.DataFrame({
            "Son 3 ay": kd_s3.groupby("CustomerID")["Tutar"].sum(),
            "Önceki 3 ay": kd_o3.groupby("CustomerID")["Tutar"].sum(),
        }).fillna(0).sort_values("Son 3 ay", ascending=False).head(10)
        st.markdown("**En çok alan müşteriler**")
        tablo(km.reset_index().rename(columns={"CustomerID": "Müşteri", "index": "Müşteri"}),
              hide_index=True, width="stretch",
              column_config={c: sayi_kolonu(c) for c in ["Son 3 ay", "Önceki 3 ay"]})
    with e2:
        birakan = kd_o3.groupby("CustomerID")["Tutar"].sum()
        birakan = birakan[~birakan.index.isin(kd_s3["CustomerID"].unique())].sort_values(ascending=False).head(10)
        st.markdown("**Son 3 ayda almayan müşteriler**")
        tablo(birakan.reset_index().rename(columns={"CustomerID": "Müşteri", "Tutar": "Önceki 3 ay"}),
              hide_index=True, width="stretch", column_config={"Önceki 3 ay": sayi_kolonu("Önceki 3 ay")})
    with e3:
        ku = kd_s3.groupby("Description").agg(Ciro=("Tutar", "sum"), Adet=("Quantity", "sum")).nlargest(10, "Ciro")
        st.markdown("**En çok satan ürünler**")
        tablo(ku.reset_index().rename(columns={"Description": "Ürün"}), hide_index=True, width="stretch",
              column_config={"Ciro": sayi_kolonu("Ciro"), "Adet": sayi_kolonu("Adet")})

# =====================================================================
# ÜRÜNLER VE TEDARİK
# =====================================================================
def adet_tahmin_tablosu(_df, kolon, anahtarlar, aylar_):
    """Her grup/ürün için aylık adet serisi kurar ve ciro tahminiyle aynı yöntem seçimini uygular."""
    pv = (_df[_df[kolon].isin(anahtarlar)].groupby([kolon, "Ay"])["Quantity"].sum()
          .unstack(fill_value=0).reindex(columns=aylar_, fill_value=0))
    satirlar = []
    for anahtar, seri_ in pv.iterrows():
        sonuc = tahmin_secimi(seri_.values, ileri=1)
        satirlar.append({
            kolon: anahtar,
            "son_ay": seri_.iloc[-1],
            "ort3": seri_.iloc[-3:].mean(),
            "tahmin": max(sonuc["tahmin"][0], 0) if sonuc else np.nan,
            "alt": max(sonuc["tahmin"][0] - sonuc["hata_std"], 0) if sonuc else np.nan,
            "ust": sonuc["tahmin"][0] + sonuc["hata_std"] if sonuc else np.nan,
            "seri": seri_.values.round(0).tolist(),
        })
    return pd.DataFrame(satirlar)


def sayfa_urun():
    aylar_ = seri.index
    adet_seri = df.groupby("Ay")["Quantity"].sum().reindex(aylar_, fill_value=0)
    adet_sonuc = tahmin_secimi(adet_seri.values, ileri=1)
    urun, n80, yuzde = pareto_bilgisi(df)

    u1, u2, u3, u4 = st.columns(4)
    if adet_sonuc is not None:
        u1.metric(f"{tr_ay(aylar_[-1] + pd.DateOffset(months=1))} adet tahmini", tr_sayi(adet_sonuc["tahmin"][0]),
                  degisim_metni(adet_sonuc["tahmin"][0], adet_seri.iloc[-1]),
                  help=f"{tr_ay(aylar_[-1])} gerçekleşen adede göre değişim.")
        u2.metric("Olası aralık (bin adet)",
                  f"{tr_sayi(max(adet_sonuc['tahmin'][0] - adet_sonuc['hata_std'], 0) / 1e3)} – "
                  f"{tr_sayi((adet_sonuc['tahmin'][0] + adet_sonuc['hata_std']) / 1e3)}")
    u3.metric("Satılan ürün çeşidi", tr_sayi(len(urun)))
    if n80 is not None:
        u4.metric("Cironun %80'ini getiren ürün", tr_sayi(n80), f"çeşidin %{tr_sayi(yuzde, 1)}'i",
                  delta_color="off")

    st.markdown("#### Kategori grubuna göre gelecek ay talebi (adet)")
    gruplar = df.groupby("KategoriGrubu")["Quantity"].sum().sort_values(ascending=False).index.tolist()
    gt_ = adet_tahmin_tablosu(df, "KategoriGrubu", gruplar, aylar_)
    if len(gt_) and gt_["tahmin"].notna().any():
        p1, p2 = st.columns([3, 2])
        with p1:
            gc = gt_.sort_values("tahmin")
            fig = go.Figure()
            fig.add_bar(y=gc["KategoriGrubu"], x=gc["ort3"], orientation="h", name="Son 3 ay ortalaması",
                        marker_color="#c7d2fe", hovertemplate="%{y}: %{x:,.0f}<extra>Son 3 ay ort.</extra>")
            fig.add_bar(y=gc["KategoriGrubu"], x=gc["tahmin"], orientation="h", name="Gelecek ay tahmini",
                        marker_color=TURUNCU,
                        error_x=dict(type="data", symmetric=False, array=(gc["ust"] - gc["tahmin"]).values,
                                     arrayminus=(gc["tahmin"] - gc["alt"]).values, color=GRI, thickness=1),
                        hovertemplate="%{y}: %{x:,.0f}<extra>Tahmin</extra>")
            fig.update_layout(template=TEMA, height=440, barmode="group", bargap=0.25, bargroupgap=0.05,
                              legend=dict(orientation="h", y=1.08, x=0), yaxis=dict(automargin=True))
            fig.update_xaxes(tickformat=",.0f")
            st.plotly_chart(fig, width="stretch")
        with p2:
            gtab = gt_.rename(columns={"KategoriGrubu": "Grup", "son_ay": tr_ay(aylar_[-1]),
                                       "tahmin": "Tahmin", "alt": "Alt", "ust": "Üst", "seri": "12 ay"})
            gtab["Değişim (%)"] = (gtab["Tahmin"] / gtab[tr_ay(aylar_[-1])].where(gtab[tr_ay(aylar_[-1])] > 0) - 1) * 100
            gtab = gtab[["Grup", tr_ay(aylar_[-1]), "Tahmin", "Alt", "Üst", "Değişim (%)", "12 ay"]].sort_values(
                "Tahmin", ascending=False)
            st.dataframe(yuvarla(gtab.drop(columns="12 ay")).assign(**{"12 ay": gtab["12 ay"]}),
                         hide_index=True, width="stretch", height=440,
                         column_config={**{c: sayi_kolonu(c) for c in [tr_ay(aylar_[-1]), "Tahmin", "Alt", "Üst",
                                                                       "Değişim (%)"]},
                                        "12 ay": st.column_config.LineChartColumn("Aylık adet", width="small")})

    st.markdown("#### Ürün bazında gelecek ay talebi")
    s3_maske = df["InvoiceDate"] > bitis - pd.DateOffset(months=3)
    n_urun = st.select_slider("Ürün sayısı", options=[20, 50, 100], value=50, key="urun_sayisi",
                              help="Son 3 ayda en çok satılan ürünler.")
    en_cok = df[s3_maske].groupby("Description")["Quantity"].sum().nlargest(n_urun).index.tolist()
    ut_ = adet_tahmin_tablosu(df, "Description", en_cok, aylar_)
    if len(ut_):
        bilgi = df[df["Description"].isin(en_cok)].groupby("Description").agg(
            Grup=("KategoriGrubu", "first"), Kategori=("Category", "first"),
            Ciro=("Tutar", "sum"), Müşteri=("CustomerID", "nunique"))
        ut_ = ut_.merge(bilgi, left_on="Description", right_index=True)
        ut_["Değişim (%)"] = (ut_["tahmin"] / ut_["son_ay"].where(ut_["son_ay"] > 0) - 1) * 100
        ut_ = ut_.sort_values("ort3", ascending=False)
        utab = pd.DataFrame({
            "Ürün": ut_["Description"], "Grup": ut_["Grup"], "Kategori": ut_["Kategori"],
            tr_ay(aylar_[-1]): ut_["son_ay"], "Son 3 ay ort.": ut_["ort3"], "Tahmin": ut_["tahmin"],
            "Alt": ut_["alt"], "Üst": ut_["ust"], "Değişim (%)": ut_["Değişim (%)"],
            "Müşteri": ut_["Müşteri"], "12 ay": ut_["seri"],
        })
        st.dataframe(yuvarla(utab.drop(columns="12 ay")).assign(**{"12 ay": utab["12 ay"]}),
                     hide_index=True, width="stretch", height=460,
                     column_config={**{c: sayi_kolonu(c) for c in [tr_ay(aylar_[-1]), "Son 3 ay ort.", "Tahmin",
                                                                   "Alt", "Üst", "Değişim (%)", "Müşteri"]},
                                    "Tahmin": st.column_config.NumberColumn(
                                        "Tahmin", format="localized",
                                        help="Her ürünün kendi aylık adet serisine, ciro tahminindeki yöntem "
                                             "seçimi uygulanarak bulunan gelecek ay adedi."),
                                    "12 ay": st.column_config.LineChartColumn("Aylık adet", width="small")})
        st.download_button("CSV olarak indir",
                           data=yuvarla(utab.drop(columns="12 ay")).to_csv(index=False, sep=";", decimal=",")
                           .encode("utf-8-sig"),
                           file_name="urun_talep_tahmini.csv", mime="text/csv")

    if n80 is not None:
        st.markdown("#### Ciro yoğunlaşması")
        kum = (urun.cumsum() / urun.sum() * 100).reset_index(drop=True)
        x = (np.arange(1, len(kum) + 1) / len(kum) * 100)
        adim = max(len(kum) // 400, 1)
        fig = go.Figure(go.Scatter(x=x[::adim], y=kum.values[::adim], mode="lines", line=dict(color=MAVI, width=2),
                                   hovertemplate="Ürünlerin %%{x:.1f}'i → cironun %%{y:.1f}'i<extra></extra>"))
        fig.add_hline(y=80, line_dash="dash", line_color=GRI)
        fig.add_vline(x=yuzde, line_dash="dash", line_color=GRI)
        fig.update_layout(template=TEMA, height=300, xaxis_title="Ürünlerin yüzdesi (cirosuna göre sıralı)",
                          yaxis_title="Kümülatif ciro payı (%)")
        st.plotly_chart(fig, width="stretch")


# =====================================================================
# ÇAPRAZ SATIŞ & SEPET ANALİZİ
# =====================================================================
# Apriori 664 bin satırda tüm kalem kümelerini tarar ve yavaştır. Burada yalnızca ikili birliktelik gerekir:
# "A'yı içeren siparişlerde B ne sıklıkla var?" Bu, fatura × ürün ikili matrisinde tek bir sütunla
# (A) matrisin çarpımıdır; gruplama ve seyrek matris çarpımıyla saniyenin altında hesaplanır.
@st.cache_data(show_spinner="Sepetler çıkarılıyor…")
def sepet_hazirla(_df, anahtar):
    kalem = _df.drop_duplicates(["InvoiceNo", "StockCode"])[["InvoiceNo", "CustomerID", "StockCode"]]
    urun = _df.groupby("StockCode").agg(Ad=("Description", "first"), Grup=("KategoriGrubu", "first"),
                                        Ciro=("Tutar", "sum"), Adet=("Quantity", "sum"))
    urun["Siparis"] = kalem.groupby("StockCode").size()
    urun["Musteri"] = kalem.groupby("StockCode")["CustomerID"].nunique()
    urun = urun.sort_values(["Siparis", "Ciro"], ascending=False)
    musteri = _df.groupby("CustomerID").agg(Sehir=("City", "first"), Ciro=("Tutar", "sum"),
                                            Son=("InvoiceDate", "max"), Siparis=("InvoiceNo", "nunique"))
    sepet_boyu = kalem.groupby("InvoiceNo").size()
    return kalem.reset_index(drop=True), urun, musteri, sepet_boyu


def capraz_oneriler(kalem, urun, n_fatura, kod):
    """Seçilen ürünü içeren siparişlerde birlikte görülen ürünler: güven, müşteri oranı ve lift."""
    secili = kalem[kalem["StockCode"] == kod]
    faturalar = secili["InvoiceNo"].unique()
    alici = secili["CustomerID"].nunique()
    ilgili = kalem[kalem["InvoiceNo"].isin(faturalar) & (kalem["StockCode"] != kod)]
    t = ilgili.groupby("StockCode").agg(birlikte=("InvoiceNo", "size"), birlikte_musteri=("CustomerID", "nunique"))
    t = t.join(urun)
    t["guven"] = t["birlikte"] / max(len(faturalar), 1)
    t["musteri_orani"] = t["birlikte_musteri"] / max(alici, 1)
    t["lift"] = t["guven"] / (t["Siparis"] / n_fatura)
    return t, len(faturalar), alici, secili["CustomerID"].unique()


@st.cache_data(show_spinner=False)
def guclu_ciftler(_kalem, _urun, n_fatura, anahtar, ust=600, min_birlikte=8):
    """En çok sipariş edilen ürünler arasında tüm çiftlerin birliktelikleri: C = Xᵀ·X (seyrek)."""
    adaylar = _urun[_urun["Ad"] != BILINMIYOR].index[:ust]
    k = _kalem[_kalem["StockCode"].isin(adaylar)]
    fi, faturalar = pd.factorize(k["InvoiceNo"])
    ui = pd.Categorical(k["StockCode"], categories=adaylar).codes
    X = sparse.csr_matrix((np.ones(len(k), dtype=np.float32), (fi, ui)), shape=(len(faturalar), len(adaylar)))
    C = (X.T @ X).tocoo()
    m = (C.row < C.col) & (C.data >= min_birlikte)
    a, b, n = C.row[m], C.col[m], C.data[m]
    sip = _urun.loc[adaylar, "Siparis"].to_numpy(dtype=float)
    t = pd.DataFrame({
        "A": _urun.loc[adaylar[a], "Ad"].to_numpy(), "B": _urun.loc[adaylar[b], "Ad"].to_numpy(),
        "Birlikte sipariş": n.astype(int),
        "A → B (%)": n / sip[a] * 100, "B → A (%)": n / sip[b] * 100,
        "Lift": n * n_fatura / (sip[a] * sip[b]),
    })
    return t.sort_values("Lift", ascending=False)


@st.cache_data(show_spinner=False)
def grup_birlikteligi(_kalem, _urun, anahtar):
    """Kategori grupları arasında sipariş düzeyinde lift matrisi."""
    k = _kalem.assign(Grup=_kalem["StockCode"].map(_urun["Grup"]))
    k = k[~k["Grup"].isin([BILINMIYOR, "Diğer"])]
    G = pd.crosstab(k["InvoiceNo"], k["Grup"]).clip(upper=1)
    C = G.T @ G
    n = len(G)
    d = np.diag(C).astype(float)
    deger = np.array(C, dtype=float) * n / np.outer(d, d)   # yeni, yazılabilir dizi
    np.fill_diagonal(deger, np.nan)
    return pd.DataFrame(deger, index=C.index, columns=C.columns)


def sayfa_capraz():
    kalem, urun, musteri, sepet_boyu = sepet_hazirla(df, FILTRE_ANAHTARI)
    n_fatura = len(sepet_boyu)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analiz edilen sipariş (sepet)", tr_sayi(n_fatura))
    c2.metric("Ortalama sepet çeşidi", tr_sayi(sepet_boyu.mean(), 1), help="Bir siparişteki farklı ürün sayısı.")
    c3.metric("Çok ürünlü sepet payı", f"%{tr_sayi((sepet_boyu > 1).mean() * 100, 1)}",
              help="Birden fazla farklı ürün içeren siparişlerin payı: çapraz satışın potansiyel alanı.")
    c4.metric("Satılan ürün çeşidi", tr_sayi(len(urun)))

    secenek = urun[(urun["Ad"] != BILINMIYOR) & (urun["Siparis"] >= 5)].head(3000)
    if secenek.empty:
        st.info("Seçili filtrelerde çapraz satış analizi için yeterli sipariş yok. Filtreleri genişletin.")
        return

    with kart("Çapraz satış öneri motoru",
              "Bir ürün seçin: bu ürünü alan B2B müşterilerin aynı siparişte en çok hangi ürünleri eklediğini "
              "ve satış temsilcisinin kime ne önermesi gerektiğini gösterir."):
        s1, s2, s3 = st.columns([3.2, 1.3, 1.2])
        kod = s1.selectbox(
            "Ürün", secenek.index.tolist(), key="cs_urun",
            format_func=lambda k_: f"{secenek.at[k_, 'Ad']}  ·  {secenek.at[k_, 'Grup']}  ·  "
                                   f"{tr_sayi(secenek.at[k_, 'Siparis'])} sipariş")
        siralama = s2.segmented_control("Sıralama", ["Müşteri oranı", "Lift"], default="Müşteri oranı",
                                        key="cs_sira", help="Müşteri oranı: en sık birlikte alınan. "
                                        "Lift: rastlantıdan en çok sapan (en güçlü) ilişki.") or "Müşteri oranı"
        min_lift = s3.select_slider("En az lift", options=[1.0, 1.2, 1.5, 2.0, 3.0, 5.0], value=1.5, key="cs_lift",
                                    help="Lift = birlikte görülme / rastlantıyla beklenen. 1'in üstü gerçek ilişki; "
                                         "çok satan ürünlerin her sepete düşmesinin etkisini ayıklar.")

        t, n_p_fatura, n_p_alici, alicilar = capraz_oneriler(kalem, urun, n_fatura, kod)
        min_birlikte = max(3, int(n_p_fatura * 0.02))
        t = t[(t["Ad"] != BILINMIYOR) & (t["birlikte"] >= min(min_birlikte, 10)) & (t["lift"] >= min_lift)]
        t = t.sort_values(["musteri_orani", "lift"] if siralama == "Müşteri oranı" else ["lift", "musteri_orani"],
                          ascending=False)
        ad = secenek.at[kod, "Ad"]

        if t.empty:
            st.info("Bu ürün için seçili lift eşiğini geçen birliktelik bulunamadı. Lift eşiğini düşürün ya da "
                    "daha çok sipariş edilen bir ürün seçin.")
            return
        ilk3 = t.head(3)
        st.markdown(
            f"<div class='cumle'><b>{html.escape(ad)}</b> ürününü alan <b>{tr_sayi(n_p_alici)}</b> B2B müşteriden "
            f"<b>{tr_sayi(ilk3['birlikte_musteri'].iloc[0])}</b> tanesi (<b>%{tr_sayi(ilk3['musteri_orani'].iloc[0] * 100, 0)}</b>) "
            f"aynı siparişte <b>{html.escape(ilk3['Ad'].iloc[0])}</b> ürününü de sepete ekledi. "
            f"En güçlü 3 çapraz satış önerisi:</div>", unsafe_allow_html=True)
        kolonlar = st.columns(3)
        for i, (_, r) in enumerate(ilk3.iterrows()):
            kolonlar[i].markdown(
                f"<div class='oneri'><span class='oneri-sira'>{i + 1}</span>"
                f"<div class='oneri-ad' title='{html.escape(r['Ad'])}'>{html.escape(r['Ad'])}</div>"
                f"<div class='oneri-grup'>{html.escape(str(r['Grup']))}</div>"
                f"<div class='oneri-oran'>%{tr_sayi(r['musteri_orani'] * 100, 0)}<span>müşteri birlikte aldı</span></div>"
                f"<div class='oneri-alt'><span class='rozet'>lift ×{tr_sayi(r['lift'], 1)}</span>"
                f"<span class='rozet yesil'>{tr_sayi(r['birlikte'])} ortak sipariş</span>"
                f"<span class='rozet'>sipariş oranı %{tr_sayi(r['guven'] * 100, 0)}</span></div></div>",
                unsafe_allow_html=True)

    o1, o2 = st.columns([3, 2])
    with o1:
        with kart("Tüm öneriler", f"{html.escape(ad)} ile birlikte alınan ilk 25 ürün"):
            ot = t.head(25)
            tab = pd.DataFrame({
                "Ürün": ot["Ad"], "Grup": ot["Grup"], "Müşteri oranı (%)": ot["musteri_orani"] * 100,
                "Sipariş oranı (%)": ot["guven"] * 100, "Lift": ot["lift"], "Ortak sipariş": ot["birlikte"],
                "Ürünün cirosu": ot["Ciro"],
            })
            st.dataframe(tab.round({"Müşteri oranı (%)": 1, "Sipariş oranı (%)": 1, "Lift": 2, "Ürünün cirosu": 0}),
                         hide_index=True, width="stretch", height=420, column_config={
                             "Müşteri oranı (%)": st.column_config.ProgressColumn(
                                 "Müşteri oranı", format="%.0f%%", min_value=0,
                                 max_value=float(max(tab["Müşteri oranı (%)"].max(), 1)),
                                 help="Seçilen ürünü alan müşterilerden bu ürünü aynı siparişte de alanların payı."),
                             "Sipariş oranı (%)": st.column_config.NumberColumn("Sipariş oranı (%)", format="%.1f",
                                                                                help="Güven: P(B | A)"),
                             "Lift": st.column_config.NumberColumn("Lift", format="×%.2f"),
                             "Ürünün cirosu": sayi_kolonu("Ürünün cirosu")})
    with o2:
        with kart("Hedef müşteri listesi", "Seçilen ürünü alıp önerilen ürünü hiç almamış müşteriler: "
                                          "satış temsilcisinin arama listesi."):
            hedef = st.selectbox("Önerilecek ürün", t.head(10).index.tolist(), key="cs_hedef",
                                 format_func=lambda k_: t.at[k_, "Ad"], label_visibility="collapsed")
            almis = set(kalem.loc[kalem["StockCode"] == hedef, "CustomerID"])
            aday = musteri.loc[[c for c in alicilar if c not in almis]].sort_values("Ciro", ascending=False)
            st.markdown(f"<span class='rozet'>{tr_sayi(len(aday))} fırsat müşterisi</span> "
                        f"<span class='rozet yesil'>toplam ciro {kisa_sayi(aday['Ciro'].sum())}</span>",
                        unsafe_allow_html=True)
            hl = aday.head(30).reset_index().rename(columns={"CustomerID": "Müşteri", "Sehir": "Şehir",
                                                              "Siparis": "Sipariş", "Son": "Son sipariş"})
            hl["Son sipariş"] = hl["Son sipariş"].dt.strftime("%d.%m.%Y")
            st.dataframe(hl[["Müşteri", "Şehir", "Ciro", "Sipariş", "Son sipariş"]].round({"Ciro": 0}),
                         hide_index=True, width="stretch", height=330, column_config={"Ciro": sayi_kolonu("Ciro")})
            st.download_button("Listeyi CSV indir", width="stretch",
                               data=aday.reset_index().to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                               file_name="capraz_satis_hedef_musteriler.csv", mime="text/csv")

    p1, p2 = st.columns([2, 3])
    with p1:
        with kart("Kategori grupları arası birliktelik", "Lift > 1: iki grup aynı siparişte rastlantıdan sık görülür."):
            lift = grup_birlikteligi(kalem, urun, FILTRE_ANAHTARI)
            if len(lift) >= 2:
                # Renk log2(lift): 0 = ilişki yok, +1 = iki kat sık, −1 = yarı sıklıkta
                kisa = [g.replace(",", " ").split()[0] for g in lift.columns]
                z = np.log2(lift.values.astype(float))
                sinir = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
                fig = go.Figure(go.Heatmap(
                    z=z, x=kisa, y=lift.index, zmid=0, zmin=-sinir, zmax=sinir, customdata=lift.values,
                    text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.1f}".replace(".", ","))(lift.values),
                    texttemplate="%{text}", textfont=dict(size=9),
                    colorscale=[[0, "#f97316"], [0.5, "#ffffff"], [1, MAVI]],
                    hovertemplate="%{y} + %{x}<br>Lift ×%{customdata:.2f}<extra></extra>",
                    colorbar=dict(thickness=10, tickvals=[-2, -1, 0, 1, 2],
                                  ticktext=["×¼", "×½", "×1", "×2", "×4"])))
                fig.update_layout(template=TEMA, height=430, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis=dict(tickangle=-45, tickfont=dict(size=10)), yaxis=dict(tickfont=dict(size=10)))
                st.plotly_chart(fig, width="stretch")
    with p2:
        with kart("En güçlü ürün çiftleri", "En çok sipariş edilen 600 ürün arasında, en az 8 ortak siparişte "
                                            "birlikte görülen çiftler (seyrek matris Xᵀ·X)."):
            ciftler = guclu_ciftler(kalem, urun, n_fatura, FILTRE_ANAHTARI)
            st.dataframe(ciftler.head(40).round({"A → B (%)": 1, "B → A (%)": 1, "Lift": 1}), hide_index=True,
                         width="stretch", height=430,
                         column_config={"Lift": st.column_config.NumberColumn("Lift", format="×%.1f")})
    st.markdown("<div class='bilgi-not'>Yöntem: Fatura (InvoiceNo) = sepet. Müşteri oranı = seçilen ürünü alan "
                "müşterilerden ikinci ürünü aynı siparişte alanların payı; sipariş oranı = P(B | A); "
                "lift = P(B | A) / P(B). Hesap tam Apriori yerine ikili birliktelik için vektörize gruplama ve "
                "seyrek matris çarpımıyla yapılır.</div>", unsafe_allow_html=True)


# =====================================================================
# İSKONTO & MARJ ANALİZİ
# =====================================================================
def iskonto_ozet(d, kolon):
    t = d.groupby(kolon).agg(Ciro=("Tutar", "sum"), Liste=("ListeTutar", "sum"), Iskonto=("IskontoTutari", "sum"),
                             Beklenen=("BeklenenIskonto", "sum"), Fazla=("FazlaIskonto", "sum"),
                             Siparis=("InvoiceNo", "nunique"))
    t["Oran"] = t["Iskonto"] / t["Liste"] * 100
    t["BeklenenOran"] = t["Beklenen"] / t["Liste"] * 100
    t["Fark"] = t["Oran"] - t["BeklenenOran"]
    return t


def iskonto_orani(d):
    return d["IskontoTutari"].sum() / d["ListeTutar"].sum() * 100 if d["ListeTutar"].sum() else 0.0


def sayfa_iskonto():
    d = df
    oran = iskonto_orani(d)
    od = onceki_donem(bas, son)
    oran_once = iskonto_orani(od) if od is not None and len(od) else None
    mt = iskonto_ozet(d, "CustomerID").join(d.groupby("CustomerID")["City"].first().rename("Şehir"))
    fazla = mt["Fazla"].clip(lower=0).sum()
    yuksek_pay = d.loc[d["IskontoOrani"] >= 0.5, "Tutar"].sum() / d["Tutar"].sum() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ağırlıklı ortalama iskonto", f"%{tr_sayi(oran, 1)}",
              None if oran_once is None else f"{oran - oran_once:+.1f} puan".replace(".", ","), delta_color="inverse",
              help="Toplam iskonto / toplam liste tutarı. Değişim bir önceki eşit uzunluktaki döneme göre.")
    c2.metric("Verilen toplam iskonto", f"₺{kisa_sayi(d['IskontoTutari'].sum())}",
              f"liste tutarına oranı %{tr_sayi(oran, 1)}", delta_color="off")
    c3.metric("Fazladan verilen iskonto", f"₺{kisa_sayi(fazla)}",
              f"{tr_sayi(int((mt['Fazla'] > 0).sum()))} müşteride", delta_color="off",
              help="Müşterilerin aldığı ürünlerin olağan iskontosunun üzerinde verilen indirim toplamı "
                   "(ürün karması düzeltilmiş). Marjdan doğrudan kayıptır.")
    c4.metric("Yüksek iskontolu satış payı", f"%{tr_sayi(yuksek_pay, 1)}",
              help="İskontosu %50 ve üzeri olan satırların ciro payı.")

    # ---- İl ----
    it = iskonto_ozet(d, "City")
    it = it[it["Ciro"] > 0].reset_index().rename(columns={"City": "İl"})
    i1, i2 = st.columns([3, 2])
    with i1:
        with kart("İllere göre iskonto haritası",
                  "Yatay çizgi şirket ortalaması. Kırmızı iller ürün karmasının gerektirdiğinden fazla iskonto alıyor."):
            lim = float(max(it["Fark"].abs().quantile(0.95), 0.5))
            fig = px.scatter(it, x="Ciro", y="Oran", size="Liste", color="Fark", log_x=True, size_max=42,
                             color_continuous_scale=[[0, YESIL], [0.5, "#e2e8f0"], [1, KIRMIZI]],
                             range_color=[-lim, lim], hover_name="İl",
                             custom_data=["Fark", "Fazla", "Siparis"],
                             labels={"Ciro": "Ciro (log)", "Oran": "İskonto oranı (%)", "Fark": "Fark (puan)"})
            fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.9),
                              hovertemplate="<b>%{hovertext}</b><br>Ciro: %{x:,.0f}<br>İskonto: %%{y:.1f}"
                                            "<br>Beklenene göre: %{customdata[0]:+.2f} puan"
                                            "<br>Fazla iskonto: %{customdata[1]:,.0f}<extra></extra>")
            for _, r in it.nlargest(8, "Ciro").iterrows():
                fig.add_annotation(x=np.log10(r["Ciro"]), y=r["Oran"], text=r["İl"], showarrow=False, yshift=14,
                                   font=dict(size=10, color="#334155"))
            fig.add_hline(y=oran, line_dash="dash", line_color=GRI,
                          annotation_text=f"ortalama %{tr_sayi(oran, 1)}", annotation_font_color=GRI)
            fig.update_layout(template=TEMA, height=420, coloraxis_colorbar=dict(thickness=10, title="puan"))
            st.plotly_chart(fig, width="stretch")
    with i2:
        with kart("Fazla iskonto yapılan iller", "Ürün karmasına göre beklenenin üzerindeki iskonto"):
            ig = it.sort_values("Fazla", ascending=False).head(12)
            st.dataframe(pd.DataFrame({"İl": ig["İl"], "Ciro": ig["Ciro"].round(0), "İskonto (%)": ig["Oran"].round(1),
                                       "Fark (puan)": ig["Fark"].round(2), "Fazla iskonto": ig["Fazla"].round(0)}),
                         hide_index=True, width="stretch", height=372, column_config={
                             "Ciro": sayi_kolonu("Ciro"),
                             "Fark (puan)": st.column_config.NumberColumn("Fark (puan)", format="%+.2f"),
                             "Fazla iskonto": st.column_config.ProgressColumn(
                                 "Fazla iskonto (₺)", format="%.0f", min_value=0,
                                 max_value=float(max(ig["Fazla"].max(), 1)))})

    # ---- Müşteri ----
    with kart("Gereğinden fazla iskonto alan müşteriler",
              "Sağ üst bölge: yüksek ciro yapıp olağanın üstünde iskonto alan, kârlılığı en çok eriten müşteriler."):
        f1, f2 = st.columns([3, 2])
        with f1:
            mg = mt[mt["Ciro"] > 0].reset_index()
            mg["Fazla_poz"] = mg["Fazla"].clip(lower=0) + 1
            ciro_esik = mg["Ciro"].quantile(0.75)
            fig = px.scatter(mg, x="Ciro", y="Fark", log_x=True, size="Fazla_poz", size_max=26,
                             color=np.where((mg["Ciro"] >= ciro_esik) & (mg["Fark"] > 1), "Öncelikli inceleme",
                                            np.where(mg["Fark"] > 1, "Fazla iskonto", "Olağan")),
                             color_discrete_map={"Öncelikli inceleme": KIRMIZI, "Fazla iskonto": TURUNCU,
                                                 "Olağan": "#cbd5e1"},
                             hover_name="CustomerID", custom_data=["Şehir", "Oran", "Fazla"],
                             labels={"Ciro": "Müşteri cirosu (log)", "Fark": "Beklenene göre iskonto farkı (puan)",
                                     "color": ""})
            fig.update_traces(marker=dict(line=dict(width=0.5, color="white")),
                              hovertemplate="<b>%{hovertext}</b> · %{customdata[0]}<br>Ciro: %{x:,.0f}<br>"
                                            "İskonto: %%{customdata[1]:.1f} (%{y:+.2f} puan)<br>"
                                            "Fazla iskonto: %{customdata[2]:,.0f}<extra></extra>")
            fig.add_hline(y=0, line_color=GRI)
            fig.add_vline(x=np.log10(max(ciro_esik, 1)), line_dash="dot", line_color=GRI,
                          annotation_text="üst %25 ciro", annotation_font_color=GRI)
            fig.update_layout(template=TEMA, height=420, legend=dict(orientation="h", y=1.08, x=0))
            st.plotly_chart(fig, width="stretch")
        with f2:
            min_ciro = st.select_slider("En az müşteri cirosu", options=[0, 10_000, 50_000, 100_000, 250_000, 500_000],
                                        value=50_000, key="isk_min", format_func=lambda v: f"₺{kisa_sayi(v)}")
            ml = mt[(mt["Ciro"] >= min_ciro) & (mt["Fazla"] > 0)].sort_values("Fazla", ascending=False)
            tab = pd.DataFrame({"Müşteri": ml.index, "Şehir": ml["Şehir"], "Ciro": ml["Ciro"].round(0),
                                "İskonto (%)": ml["Oran"].round(1), "Olağan (%)": ml["BeklenenOran"].round(1),
                                "Fark (puan)": ml["Fark"].round(2), "Fazla iskonto": ml["Fazla"].round(0)})
            st.dataframe(tab, hide_index=True, width="stretch", height=330, column_config={
                "Ciro": sayi_kolonu("Ciro"), "Fazla iskonto": sayi_kolonu("Fazla iskonto (₺)"),
                "Fark (puan)": st.column_config.NumberColumn("Fark (puan)", format="%+.2f")})
            st.download_button("CSV indir", data=tab.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                               file_name="fazla_iskonto_musteriler.csv", mime="text/csv", width="stretch")

    # ---- Dağılım ve trend ----
    t1, t2 = st.columns(2)
    with t1:
        with kart("İskonto dilimlerine göre ciro", "Cironun hangi iskonto seviyelerinde yapıldığı"):
            dilim = pd.cut(d["IskontoOrani"] * 100, [-0.01, 20, 30, 35, 40, 45, 50, 100],
                           labels=["≤%20", "%20–30", "%30–35", "%35–40", "%40–45", "%45–50", "%50+"])
            dg = d.groupby(dilim, observed=False)["Tutar"].sum()
            dg = (dg / dg.sum() * 100).reset_index()
            dg.columns = ["Dilim", "Pay"]
            fig = px.bar(dg, x="Dilim", y="Pay", labels={"Dilim": "İskonto oranı", "Pay": "Ciro payı (%)"},
                         color="Dilim", color_discrete_sequence=["#e0e7ff", "#c7d2fe", "#a5b4fc", "#818cf8",
                                                                 "#6366f1", TURUNCU, KIRMIZI])
            fig.update_traces(hovertemplate="%{x}: cironun %%{y:.1f}<extra></extra>")
            fig.update_layout(template=TEMA, height=320, showlegend=False, bargap=0.25)
            st.plotly_chart(fig, width="stretch")
    with t2:
        with kart("Aylık iskonto eğilimi", "Gerçekleşen oran ve ürün karmasına göre olağan oran"):
            ay = d.groupby("Ay")[["IskontoTutari", "BeklenenIskonto", "ListeTutar"]].sum()
            ay = pd.DataFrame({"Ay": ay.index, "Gerçekleşen": ay["IskontoTutari"] / ay["ListeTutar"] * 100,
                               "Olağan (ürün karması)": ay["BeklenenIskonto"] / ay["ListeTutar"] * 100})
            fig = px.line(ay.melt(id_vars="Ay", var_name="Seri", value_name="Oran"), x="Ay", y="Oran", color="Seri",
                          markers=True, labels={"Ay": "", "Oran": "İskonto (%)"},
                          color_discrete_map={"Gerçekleşen": MAVI, "Olağan (ürün karması)": GRI})
            fig.update_traces(selector=dict(name="Olağan (ürün karması)"), line_dash="dot")
            fig.update_layout(template=TEMA, height=320, legend=dict(orientation="h", y=1.12, x=0, title=""))
            tr_ay_ekseni(fig, ay["Ay"])
            st.plotly_chart(fig, width="stretch")

    # ---- Marjı eriten siparişler ----
    with kart("Yüksek ciro, eriyen marj: incelenmesi gereken siparişler",
              "Ciro bakımından üst %10'daki siparişlerden, olağandan en az 2 puan fazla iskontolu olanlar."):
        ft = iskonto_ozet(d, "InvoiceNo")
        bilgi = d.groupby("InvoiceNo").agg(Müşteri=("CustomerID", "first"), Şehir=("City", "first"),
                                           Tarih=("InvoiceDate", "first"))
        ft = ft.join(bilgi)
        sec_ = ft[(ft["Ciro"] >= ft["Ciro"].quantile(0.9)) & (ft["Fark"] >= 2)].sort_values("Fazla", ascending=False)
        st.markdown(f"<span class='rozet'>{tr_sayi(len(sec_))} sipariş</span> "
                    f"<span class='rozet yesil'>fazla iskonto toplamı ₺{kisa_sayi(sec_['Fazla'].sum())}</span>",
                    unsafe_allow_html=True)
        tab = pd.DataFrame({"Fatura": sec_.index, "Tarih": sec_["Tarih"].dt.strftime("%d.%m.%Y"),
                            "Müşteri": sec_["Müşteri"], "Şehir": sec_["Şehir"], "Ciro": sec_["Ciro"].round(0),
                            "İskonto (%)": sec_["Oran"].round(1), "Olağan (%)": sec_["BeklenenOran"].round(1),
                            "Fazla iskonto": sec_["Fazla"].round(0)}).head(50)
        st.dataframe(tab, hide_index=True, width="stretch", height=360,
                     column_config={"Ciro": sayi_kolonu("Ciro"), "Fazla iskonto": sayi_kolonu("Fazla iskonto (₺)")})
    st.markdown("<div class='bilgi-not'>Veride ürün maliyeti yok; bu nedenle marj erimesi, liste fiyatından verilen ve "
                "ürün karmasıyla açıklanamayan iskonto üzerinden ölçülür. Olağan iskonto = her ürünün tüm "
                "müşterilere ortalama (ağırlıklı) iskonto oranı × liste tutarı. Fazla iskonto = verilen − olağan.</div>",
                unsafe_allow_html=True)


# =====================================================================
# COVID ETKİSİ
# =====================================================================
COVID_RENKLERI = {"COVID öncesi": "#94a3b8", "İlk dalga kısıtlamaları": "#ef4444",
                  "Normalleşme": "#10b981", "İkinci dalga kısıtlamaları": "#f97316"}
COVID_ACIKLAMA = {
    "COVID öncesi": "2 Oca – 10 Mar 2020 · ilk vaka açıklanmadan önce",
    "İlk dalga kısıtlamaları": "11 Mar – 31 May 2020 · okullar kapalı, sokağa çıkma kısıtlamaları",
    "Normalleşme": "1 Haz – 19 Kas 2020 · yeni normalleşme",
    "İkinci dalga kısıtlamaları": "20 Kas – 31 Ara 2020 · hafta sonu sokağa çıkma kısıtlamaları",
}


def covid_donem_tablosu(d):
    """Seçili filtrelerde her COVID döneminin gün sayısı ve günlük ortalama göstergeleri."""
    sin = [pd.Timestamp(t) for t in COVID_SINIRLARI]
    baslar = [pd.Timestamp(tarih_min)] + sin
    bitisler = [s - pd.Timedelta(days=1) for s in sin] + [pd.Timestamp(tarih_max)]
    donem = d["CovidDonemi"].astype(str)
    satirlar = []
    for etiket, b_, s_ in zip(COVID_ETIKETLERI, baslar, bitisler):
        gun = (min(s_, pd.Timestamp(son)) - max(b_, pd.Timestamp(bas))).days + 1
        alt = d[donem == etiket]
        if gun < 1 or alt.empty:
            continue
        satirlar.append({
            "Dönem": etiket, "Gün": gun, "Ciro": alt["Tutar"].sum(), "Günlük ciro": alt["Tutar"].sum() / gun,
            "Günlük sipariş": alt["InvoiceNo"].nunique() / gun, "Aktif müşteri": alt["CustomerID"].nunique(),
            "Ort. sipariş": alt.groupby("InvoiceNo")["Tutar"].sum().mean(),
            "İskonto (%)": iskonto_orani(alt),
        })
    return pd.DataFrame(satirlar)


def sayfa_covid():
    t = covid_donem_tablosu(df)
    if len(t) < 2:
        st.info("COVID karşılaştırması için dönem filtresini genişletin: en az iki farklı COVID dönemi gerekir.")
        return
    temel = t.iloc[0]["Dönem"] == "COVID öncesi"
    taban = t.iloc[0] if temel else None

    kolonlar = st.columns(len(t))
    for kol, (_, r) in zip(kolonlar, t.iterrows()):
        degisim = None
        if taban is not None and r["Dönem"] != "COVID öncesi":
            degisim = degisim_metni(r["Günlük ciro"], taban["Günlük ciro"])
        kol.metric(r["Dönem"], f"₺{kisa_sayi(r['Günlük ciro'])} / gün", degisim,
                   help=f"{COVID_ACIKLAMA[r['Dönem']]}. Günlük ortalama ciro; değişim COVID öncesine göre.")
    if taban is None:
        st.caption("Seçili filtrede 'COVID öncesi' dönemi bulunmuyor; değişimler gösterilemiyor.")

    c1, c2 = st.columns([2, 3])
    with c1:
        with kart("Dönemlere göre günlük ciro", "Dönem uzunlukları farklı olduğu için günlük ortalama karşılaştırılır"):
            fig = go.Figure(go.Bar(
                x=t["Dönem"], y=t["Günlük ciro"], marker_color=[COVID_RENKLERI[d_] for d_ in t["Dönem"]],
                text=[f"₺{kisa_sayi(v)}" for v in t["Günlük ciro"]], textposition="outside",
                customdata=t[["Gün", "Ciro"]].values,
                hovertemplate="%{x}<br>Günlük ciro: %{y:,.0f}<br>%{customdata[0]} gün · toplam %{customdata[1]:,.0f}"
                              "<extra></extra>"))
            fig.update_layout(template=TEMA, height=360, bargap=0.35, showlegend=False,
                              yaxis=dict(title="Günlük ortalama ciro", tickformat=",.0f"))
            fig.update_xaxes(tickfont=dict(size=10))
            st.plotly_chart(fig, width="stretch")
    with c2:
        with kart("Haftalık ciro, dönemlere göre renkli", "Gölge yerine her haftanın rengi dönemini gösterir"):
            hf = (df.assign(Dönem=df["CovidDonemi"].astype(str))
                  .groupby([pd.Grouper(key="InvoiceDate", freq="W-SUN"), "Dönem"])["Tutar"].sum().reset_index())
            hf = hf[hf["Tutar"] > 0].sort_values("Tutar", ascending=False).drop_duplicates("InvoiceDate")
            fig = px.bar(hf.sort_values("InvoiceDate"), x="InvoiceDate", y="Tutar", color="Dönem",
                         color_discrete_map=COVID_RENKLERI, category_orders={"Dönem": COVID_ETIKETLERI},
                         labels={"InvoiceDate": "", "Tutar": "Haftalık ciro"})
            fig.update_layout(template=TEMA, height=360, bargap=0.2, legend=dict(orientation="h", y=1.14, x=0, title=""))
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig, width="stretch")

    with kart("Dönem karşılaştırması", "Tüm göstergelerin dönem bazında dökümü"):
        tab = t.copy()
        tab["Açıklama"] = tab["Dönem"].map(COVID_ACIKLAMA)
        st.dataframe(tab.round({"Ciro": 0, "Günlük ciro": 0, "Günlük sipariş": 1, "Ort. sipariş": 0, "İskonto (%)": 1}),
                     hide_index=True, width="stretch", column_config={
                         "Ciro": sayi_kolonu("Ciro"), "Günlük ciro": sayi_kolonu("Günlük ciro"),
                         "Günlük sipariş": st.column_config.NumberColumn("Günlük sipariş", format="%.1f"),
                         "Aktif müşteri": sayi_kolonu("Aktif müşteri"), "Ort. sipariş": sayi_kolonu("Ort. sipariş"),
                         "İskonto (%)": st.column_config.NumberColumn("İskonto (%)", format="%.1f")})

    if taban is not None and len(t) > 1:
        with kart("Hangi kategoriler etkilendi?", "Seçilen dönemin günlük cirosu, COVID öncesi günlük ciroyla kıyaslanır"):
            diger = [d_ for d_ in t["Dönem"] if d_ != "COVID öncesi"]
            sec_d = st.segmented_control("Dönem", diger, default=diger[0], key="covid_kat_donem",
                                         label_visibility="collapsed") or diger[0]
            gun = t.set_index("Dönem")["Gün"]
            donem = df["CovidDonemi"].astype(str)
            g_ = lambda e: df[donem == e].groupby("KategoriGrubu")["Tutar"].sum() / gun[e]   # noqa: E731
            kt = pd.DataFrame({"once": g_("COVID öncesi"), "sonra": g_(sec_d)}).dropna()
            kt = kt[(kt["once"] > 0) & ~kt.index.isin([BILINMIYOR, "Diğer"])]
            kt = kt[kt["once"] >= kt["once"].sum() * 0.01]
            kt["deg"] = (kt["sonra"] / kt["once"] - 1) * 100
            kt = kt.sort_values("deg")
            fig = go.Figure(go.Bar(
                x=kt["deg"], y=kt.index, orientation="h",
                marker_color=[KIRMIZI if v < 0 else YESIL for v in kt["deg"]],
                text=[f"{'+' if v >= 0 else '−'}%{tr_sayi(abs(v), 0)}" for v in kt["deg"]], textposition="outside",
                customdata=kt[["once", "sonra"]].values,
                hovertemplate="%{y}<br>COVID öncesi: ₺%{customdata[0]:,.0f}/gün<br>"
                              + sec_d + ": ₺%{customdata[1]:,.0f}/gün<extra></extra>"))
            fig.add_vline(x=0, line_color=GRI)
            fig.update_layout(template=TEMA, height=max(320, 34 * len(kt) + 90), bargap=0.3,
                              xaxis_title="Günlük ciro değişimi (%)", yaxis=dict(automargin=True))
            st.plotly_chart(fig, width="stretch")
    st.markdown("<div class='bilgi-not'>Veri yalnızca 2020 yılını kapsıyor; mevsimsellik ölçülemediği için dönemler "
                "arasındaki farkın tamamı COVID'e atfedilemez (örn. yaz aylarında kitap satışı zaten düşebilir). "
                "COVID öncesi dönem yalnızca ~10 haftadır, bu yüzden kıyaslama tabanı kısadır.</div>",
                unsafe_allow_html=True)


# =====================================================================
# DİNAMİK YÖNETİCİ ÖZETİ
# =====================================================================
def kayip_riski(d):
    """Önceki döneme göre alımı yarıdan fazla düşen değerli müşteriler."""
    bit, bas_ = d["InvoiceDate"].max(), d["InvoiceDate"].min()
    gun = (bit - bas_).days
    if gun < 56:
        return None
    if gun >= 180:
        pencere, ad = pd.DateOffset(months=3), "son 3 ayda"
    else:
        pencere, ad = pd.Timedelta(days=gun // 2), f"son {gun // 2} günde"
    s = d["InvoiceDate"] > bit - pencere
    o = (d["InvoiceDate"] > bit - pencere - pencere) & ~s
    t = pd.DataFrame({"son": d[s].groupby("CustomerID")["Tutar"].sum(),
                      "onceki": d[o].groupby("CustomerID")["Tutar"].sum()}).fillna(0)
    t = t[t["onceki"] > 0]
    if t.empty:
        return None
    degerli = t["onceki"] >= t["onceki"].quantile(0.5)
    risk = t[degerli & (t["son"] <= 0.5 * t["onceki"])].sort_values("onceki", ascending=False)
    return risk, ad


def yonetici_ozeti():
    yer = f"{sehirler[0]}:" if len(sehirler) == 1 else ("Tüm iller:" if len(sehirler) == len(sehir_ciro)
                                                         else f"Seçili {len(sehirler)} il:")
    ogeler = []

    # 1) Satış: kayıp riski
    r = kayip_riski(df)
    if r is not None and len(r[0]):
        risk, ad = r
        kayip = (risk["onceki"] - risk["son"]).sum()
        ilk = ", ".join(risk.index[:3])
        ogeler.append(("#ef4444", "Satış · Kayıp riski", f"{tr_sayi(len(risk))} kurumsal müşteri · ₺{kisa_sayi(kayip)}",
                       f"{yer} {ad} alımı önceki döneme göre yarıdan fazla düşen {tr_sayi(len(risk))} değerli müşteri "
                       f"tespit edildi; önceki dönemde ₺{kisa_sayi(risk['onceki'].sum())} ciro yapmışlardı, "
                       f"risk altındaki ciro ₺{kisa_sayi(kayip)}.",
                       f"→ Öncelikli ziyaret: {ilk}"))
    else:
        ogeler.append(("#10b981", "Satış · Kayıp riski", "Belirgin risk yok",
                       f"{yer} seçili dönemde alımı yarıdan fazla düşen değerli müşteri bulunmadı.",
                       "→ Mevcut ziyaret planına devam"))

    # 2) Tedarik: gelecek ay kategori talebi
    gruplar = [g for g in df["KategoriGrubu"].unique() if g not in (BILINMIYOR, "Diğer")]
    kt = adet_tahmin_tablosu(df, "KategoriGrubu", gruplar, seri.index) if len(seri) >= BASLANGIC + TEST_AY + 1 else None
    if kt is not None and len(kt) and kt["tahmin"].notna().any():
        kt = kt.dropna(subset=["tahmin"]).assign(fark=lambda x: x["tahmin"] - x["ort3"])
        art = kt.loc[kt["fark"].idxmax()]
        dus = kt.loc[kt["fark"].idxmin()]
        gelecek_ad = tr_ay(seri.index[-1] + pd.DateOffset(months=1))
        if art["fark"] > 0:
            yuzde = art["fark"] / max(art["ort3"], 1) * 100
            ogeler.append(("#4f46e5", "Tedarik · Talep", f"{art['KategoriGrubu']} +%{tr_sayi(yuzde, 0)}",
                           f"{gelecek_ad} için {art['KategoriGrubu']} talebinin ≈{tr_sayi(art['tahmin'])} adet olması "
                           f"bekleniyor (son 3 ay ortalaması {tr_sayi(art['ort3'])}). "
                           f"En büyük düşüş: {dus['KategoriGrubu']} ({tr_sayi(dus['fark'])} adet).",
                           f"→ {art['KategoriGrubu']} stoklarını ≈{tr_sayi(art['fark'])} adet artırın"))
        else:
            ogeler.append(("#4f46e5", "Tedarik · Talep", f"{dus['KategoriGrubu']} {tr_sayi(dus['fark'])} adet",
                           f"{gelecek_ad} için hiçbir grupta artış beklenmiyor; en büyük düşüş {dus['KategoriGrubu']} "
                           f"grubunda (≈{tr_sayi(dus['tahmin'])} adet).",
                           "→ Tedarik siparişlerini azaltın, stok fazlasını önleyin"))
    else:
        ogeler.append(("#4f46e5", "Tedarik · Talep", "Tahmin için kısa dönem",
                       "Gelecek ay tahmini için en az 8 aylık veri gerekir. Dönem filtresini genişletin.", ""))

    # 3) Fiyatlama: fazla iskonto
    mt = iskonto_ozet(df, "CustomerID")
    it = iskonto_ozet(df, "City")
    fazla = mt["Fazla"].clip(lower=0).sum()
    if fazla > 0 and len(it):
        il_ = it["Fazla"].idxmax()
        mus = mt["Fazla"].idxmax()
        ogeler.append(("#f97316", "Fiyatlama · Marj", f"₺{kisa_sayi(fazla)} fazladan iskonto",
                       f"Ürün karmasının gerektirdiğinin üzerinde iskonto verildi. En çok {il_} "
                       f"(₺{kisa_sayi(it.at[il_, 'Fazla'])}, {'+' if it.at[il_, 'Fark'] >= 0 else '−'}"
                       f"{tr_sayi(abs(it.at[il_, 'Fark']), 1)} puan); en büyük tek müşteri "
                       f"{mus} (₺{kisa_sayi(mt.at[mus, 'Fazla'])}).",
                       "→ İskonto onay limitlerini gözden geçirin"))
    sutunlar = "".join(
        f"<div class='ozet-oge' style='--renk:{renk}'><div class='ozet-etiket'>{html.escape(etiket)}</div>"
        f"<div class='ozet-sayi'>{html.escape(sayi)}</div><div class='ozet-metin'>{html.escape(metin)}</div>"
        + (f"<div class='ozet-aksiyon'>{html.escape(aksiyon)}</div>" if aksiyon else "") + "</div>"
        for renk, etiket, sayi, metin, aksiyon in ogeler)
    st.markdown(
        "<div class='ozet'><div class='ozet-bas'><div class='ozet-baslik'>⚡ Satış & Tedarik Aksiyon Özeti "
        "<span class='ozet-canli'>● filtrelerle canlı</span></div>"
        f"<div class='ozet-not'>{html.escape(tr_tarih(bas))} – {html.escape(tr_tarih(son))} · "
        f"{html.escape(sehir_etiket)} · {html.escape(kat_etiket)}</div></div>"
        f"<div class='ozet-grid'>{sutunlar}</div></div>", unsafe_allow_html=True)


# =====================================================================
# ASİSTAN (basit, kural tabanlı sohbet: harici servis ve API anahtarı gerektirmez)
# =====================================================================
_ASCII = str.maketrans("İIıŞşĞğÜüÖöÇç", "IIISSGGUUOOCC")


def _norm(s):
    return "".join(ch if ch.isalnum() else " " for ch in str(s).translate(_ASCII).upper())


def asistan_cevap(soru):
    """Seçili filtrelerdeki veriye bakarak soruyu anahtar kelimelerle yanıtlar."""
    q = _norm(soru)
    kelimeler = set(q.split())

    def var(*anahtar):
        return any(a in q for a in anahtar)

    # Belirli bir il sorulduysa
    for sehir in df["City"].unique():
        if _norm(sehir) in kelimeler:
            d = df[df["City"] == sehir]
            en_iyi = d.groupby("Description")["Tutar"].sum().nlargest(3)
            return (f"**{sehir}**: ciro ₺{tr_sayi(d['Tutar'].sum())} (toplamın %{tr_sayi(d['Tutar'].sum() / df['Tutar'].sum() * 100, 1)}'i), "
                    f"{tr_sayi(d['CustomerID'].nunique())} müşteri, {tr_sayi(d['InvoiceNo'].nunique())} sipariş, "
                    f"iskonto %{tr_sayi(iskonto_orani(d), 1)}.  \nEn çok ciro getiren ürünler: "
                    + "; ".join(en_iyi.index) + ".")
    if var("RISK", "KAYIP", "KAYBE"):
        r = kayip_riski(df)
        if r is None or not len(r[0]):
            return "Seçili dönemde alımı yarıdan fazla düşen değerli müşteri yok."
        risk, ad = r
        return (f"{ad.capitalize()} alımı önceki döneme göre yarıdan fazla düşen **{len(risk)} değerli müşteri** var; "
                f"risk altındaki ciro ≈₺{kisa_sayi((risk['onceki'] - risk['son']).sum())}.  \n"
                "En büyük 5: " + ", ".join(f"{c} (₺{kisa_sayi(v)})" for c, v in risk['onceki'].head(5).items()))
    if var("ISKONTO", "INDIRIM", "MARJ"):
        mt = iskonto_ozet(df, "CustomerID")
        it = iskonto_ozet(df, "City")
        return (f"Ağırlıklı ortalama iskonto **%{tr_sayi(iskonto_orani(df), 1)}**. Ürün karmasının üzerinde verilen fazla "
                f"iskonto ≈₺{kisa_sayi(mt['Fazla'].clip(lower=0).sum())}.  \nEn çok fazla iskonto: il {it['Fazla'].idxmax()}, "
                f"müşteri {mt['Fazla'].idxmax()}.")
    if var("TAHMIN", "GELECEK", "BEKLEN", "OCAK"):
        s = f"{tr_ay(tahmin_ay)} için beklenen büyük sipariş veren müşteri sayısı ≈{tr_sayi(tah['musteri']['p'].sum())}."
        if sec is not None:
            s = f"{tr_ay(seri.index[-1] + pd.DateOffset(months=1))} ciro beklentisi **₺{tr_sayi(sec['tahmin'][0])}**. " + s
        return s
    if var("MUSTERI") and var("EN ", "ILK", "BUYUK", "COK"):
        m = df.groupby("CustomerID")["Tutar"].sum().nlargest(5)
        return "Ciroya göre en büyük 5 müşteri: " + ", ".join(f"{c} (₺{kisa_sayi(v)})" for c, v in m.items())
    if var("URUN", "KITAP"):
        u = df.groupby("Description")["Tutar"].sum().nlargest(5)
        return "Ciroya göre en çok satan 5 ürün:  \n" + "  \n".join(f"{i + 1}. {a} (₺{kisa_sayi(v)})"
                                                              for i, (a, v) in enumerate(u.items()))
    if var("KATEGORI", "GRUP", "TUR"):
        k_ = df.groupby("KategoriGrubu")["Tutar"].sum().nlargest(5)
        return "En büyük 5 kategori grubu: " + ", ".join(f"{a} (%{tr_sayi(v / df['Tutar'].sum() * 100, 1)})" for a, v in k_.items())
    if var("SEHIR", " IL ", "BOLGE", "HARITA") or q.strip() in ("IL", "ILLER"):
        i_ = df.groupby("City")["Tutar"].sum().nlargest(5)
        return "Ciroya göre ilk 5 il: " + ", ".join(f"{a} (%{tr_sayi(v / df['Tutar'].sum() * 100, 1)})" for a, v in i_.items())
    if var("CIRO", "SATIS", "TOPLAM", "KPI", "OZET", "DURUM", "SIPARIS", "MUSTERI"):
        k_ = kpi_hesapla(df)
        return (f"{tr_tarih(bas)} – {tr_tarih(son)}: ciro **₺{tr_sayi(k_['ciro'])}**, {tr_sayi(k_['fatura'])} sipariş, "
                f"{tr_sayi(k_['musteri'])} aktif müşteri, ortalama sipariş ₺{tr_sayi(k_['ort_fatura'])}.")
    return ("Bunu anlayamadım. Şunları sorabilirsiniz: *toplam ciro*, *en çok satan ürünler*, *en büyük müşteriler*, "
            "*kayıp riski*, *iskonto*, *gelecek ay tahmini*, *kategoriler*, *iller* ya da bir il adı (örn. *İzmir*).")


ASISTAN_CSS = """
<style>
/* Sağ altta yüzen sohbet düğmesi */
.st-key-asistan_kutu { position: fixed; right: 24px; bottom: 24px; z-index: 1000; width: auto !important; }
.st-key-asistan_kutu button[data-testid="stPopoverButton"] {
  border-radius: 999px; padding: .75rem 1.2rem; font-weight: 700; color: #fff; border: none;
  background: linear-gradient(135deg, #4f46e5, #7c3aed); box-shadow: 0 10px 28px rgba(79,70,229,.45); }
.st-key-asistan_kutu button[data-testid="stPopoverButton"]:hover { filter: brightness(1.08); color:#fff; }
.st-key-asistan_kutu button[data-testid="stPopoverButton"] p { color:#fff; }
div[data-testid="stPopoverBody"]:has(.asistan-baslik) { width: min(400px, 92vw); border-radius: 18px; padding: 0;
  box-shadow: 0 20px 50px rgba(16,24,40,.25); }
.asistan-baslik { background: linear-gradient(135deg,#312e81,#4f46e5 60%,#6d28d9); color:#fff; padding:.85rem 1.1rem;
  font-weight:700; border-radius:18px 18px 0 0; }
.asistan-baslik small { display:block; font-weight:400; opacity:.8; font-size:.72rem; }
</style>
"""
ASISTAN_ORNEKLER = ["Toplam ciro ne kadar?", "Kayıp riski olan müşteriler", "En çok satan ürünler",
                    "Fazla iskonto nerede?", "Gelecek ay tahmini"]


def _soru_ekle(soru):
    soru = (soru or "").strip()
    if soru:
        st.session_state["sohbet"] += [("user", soru), ("assistant", asistan_cevap(soru))]


def _form_gonder():
    _soru_ekle(st.session_state.get("asistan_giris"))


def asistan_kutusu():
    """Sağ altta yüzen sohbet kutusu. Tüm sayfalarda görünür; kural tabanlı, harici servise veri göndermez."""
    st.markdown(ASISTAN_CSS, unsafe_allow_html=True)
    if "sohbet" not in st.session_state:
        st.session_state["sohbet"] = [("assistant", "Merhaba! Ciro, müşteri riski, iskonto ya da bir il hakkında soru sorabilirsiniz.")]
    with st.container(key="asistan_kutu"):
        with st.popover("💬 Asistan"):
            st.markdown("<div class='asistan-baslik'>Satış asistanı<small>Seçili filtrelere göre yanıtlar</small></div>",
                        unsafe_allow_html=True)
            with st.container(height=330, border=False):
                for rol, metin in st.session_state["sohbet"]:
                    with st.chat_message(rol):
                        st.markdown(metin)
            for o in ASISTAN_ORNEKLER[:3]:
                st.button(o, key=f"ornek_{o}", on_click=_soru_ekle, args=(o,), width="stretch")
            with st.form("asistan_form", clear_on_submit=True, border=False):
                c1, c2 = st.columns([5, 1.4])
                c1.text_input("Soru", key="asistan_giris", placeholder="Bir soru yazın…", label_visibility="collapsed")
                c2.form_submit_button("Gönder", on_click=_form_gonder, width="stretch")


# =====================================================================
# NAVİGASYON
# =====================================================================
sayfalar = [
    st.Page(sayfa_genel, title="Genel bakış", icon=":material/space_dashboard:", url_path="genel", default=True),
    st.Page(sayfa_tahmin, title=f"{tr_ay(tahmin_ay)} tahmini", icon=":material/insights:", url_path="tahmin"),
    st.Page(sayfa_capraz, title="Çapraz satış", icon=":material/shopping_basket:", url_path="capraz-satis"),
    st.Page(sayfa_iskonto, title="İskonto & marj", icon=":material/percent:", url_path="iskonto"),
    st.Page(sayfa_urun, title="Ürünler & tedarik", icon=":material/inventory_2:", url_path="urunler"),
    st.Page(sayfa_kategori, title="Kategoriler", icon=":material/category:", url_path="kategoriler"),
    st.Page(sayfa_covid, title="COVID etkisi", icon=":material/coronavirus:", url_path="covid"),
]
secili_sayfa = st.navigation(sayfalar, position="top")
hero()
secili_sayfa.run()
asistan_kutusu()
