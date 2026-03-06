"""
=============================================================================
UYGULAMA 01 — Veri Ön İşleme Temelleri
=============================================================================
Kapsam:
  - Veri yükleme ve keşifsel analiz (EDA)
  - Eksik değer tespiti ve SimpleImputer ile doldurma
  - StandardScaler ile ölçekleme
  - OneHotEncoder ve OrdinalEncoder ile kategorik dönüşüm
  - fit() ve transform() arasındaki kritik fark (Data Leakage)
  - Preprocessing öncesi ve sonrası görselleştirme

Veri seti: UCI Bank Marketing (ya da yerleşik örnek veri)
Kurulum:   pip install scikit-learn pandas numpy matplotlib seaborn
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ─── Stil ayarı ───────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False})

print("=" * 60)
print("  UYGULAMA 01 — Veri Ön İşleme Temelleri")
print("=" * 60)

# ═════════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME
# ═════════════════════════════════════════════════════════════
print("\n[1] Veri yükleniyor...")

url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
       "/00222/bank-additional-full.csv")
try:
    df = pd.read_csv(url, sep=";")
    print(f"    ✅ UCI Bank Marketing: {df.shape[0]:,} satır, {df.shape[1]} sütun")
except Exception:
    print("    ⚠️  İnternet yok — örnek veri oluşturuluyor...")
    np.random.seed(42)
    n = 4000
    df = pd.DataFrame({
        "age":       np.random.randint(18, 75, n).astype(float),
        "job":       np.random.choice(
                         ["admin.", "blue-collar", "technician",
                          "services", "management", "retired"], n),
        "marital":   np.random.choice(["married", "single", "divorced"], n),
        "education": np.random.choice(
                         ["basic.4y", "high.school",
                          "university.degree", "professional.course"], n),
        "balance":   np.random.normal(1500, 3000, n),
        "duration":  np.abs(np.random.normal(260, 260, n)),
        "campaign":  np.random.randint(1, 10, n).astype(float),
        "pdays":     np.random.choice(
                         [999] + list(range(0, 30)), n,
                         p=[0.85] + [0.005] * 30).astype(float),
        "previous":  np.random.randint(0, 5, n).astype(float),
        "contact":   np.random.choice(["telephone", "cellular"], n),
        "poutcome":  np.random.choice(
                         ["failure", "success", "nonexistent"], n),
        "y":         np.random.choice(["yes", "no"], n, p=[0.12, 0.88]),
    })
    print(f"    ✅ Örnek veri oluşturuldu: {df.shape}")

# Target sütunu
df["target"] = (df["y"] == "yes").astype(int)

# ─── Yapay eksik değer ekle (gerçekçi senaryo) ───────────────
np.random.seed(0)
for col in ["age", "balance", "duration"]:
    df.loc[np.random.rand(len(df)) < 0.05, col] = np.nan   # %5 eksik
for col in ["job", "education"]:
    df.loc[np.random.rand(len(df)) < 0.07, col] = np.nan   # %7 eksik

print("\n    Eksik değer sayıları:")
miss = df.isnull().sum()
print(miss[miss > 0].to_string())

# ═════════════════════════════════════════════════════════════
# 2. EĞİTİM / TEST AYIRIMI
# ═════════════════════════════════════════════════════════════
print("\n[2] Eğitim/Test bölmesi yapılıyor...")

num_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]
cat_ohe  = ["job", "marital", "contact", "poutcome"]
cat_ord  = ["education"]

# Sadece mevcut sütunları al
num_cols = [c for c in num_cols if c in df.columns]
cat_ohe  = [c for c in cat_ohe  if c in df.columns]
cat_ord  = [c for c in cat_ord  if c in df.columns]
all_cols = num_cols + cat_ohe + cat_ord

X = df[all_cols].copy()
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"    Eğitim: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ═════════════════════════════════════════════════════════════
# 3. SimpleImputer — Eksik Değer Doldurma
# ═════════════════════════════════════════════════════════════
print("\n[3] SimpleImputer ile eksik değer doldurma...")

# Sayısal: medyan stratejisi
num_imputer = SimpleImputer(strategy="median")
X_train_num_imp = num_imputer.fit_transform(X_train[num_cols])
X_test_num_imp  = num_imputer.transform(X_test[num_cols])   # ← fit() YOK!

print("    Sayısal imputer öğrenilen medyanlar:")
for col, stat in zip(num_cols, num_imputer.statistics_):
    print(f"      {col:<12}: {stat:.2f}")

# Kategorik: en sık değer stratejisi
cat_imputer = SimpleImputer(strategy="most_frequent")
X_train_cat_imp = cat_imputer.fit_transform(X_train[cat_ohe + cat_ord])
X_test_cat_imp  = cat_imputer.transform(X_test[cat_ohe + cat_ord])

print("\n    Kategorik imputer öğrenilen mod değerleri:")
for col, stat in zip(cat_ohe + cat_ord, cat_imputer.statistics_):
    print(f"      {col:<12}: {stat}")

print("\n    ⚠️  Data Leakage Neden Oluşur?")
print("        Yanlış: imputer.fit_transform(X_tüm) → sonra ikiye bol")
print("        Doğru:  önce bol, imputer.fit(X_train), transform(X_test)")

# ═════════════════════════════════════════════════════════════
# 4. StandardScaler — Ölçekleme
# ═════════════════════════════════════════════════════════════
print("\n[4] StandardScaler ile ölçekleme...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num_imp)
X_test_scaled  = scaler.transform(X_test_num_imp)

print("    Öğrenilen istatistikler (ilk 3 özellik):")
for i, col in enumerate(num_cols[:3]):
    print(f"      {col:<12}: ortalama={scaler.mean_[i]:.2f}, std={scaler.scale_[i]:.2f}")

# Ölçekleme sonrası kontrol
df_scaled = pd.DataFrame(X_train_scaled, columns=num_cols)
print(f"\n    Ölçekleme sonrası 'age' özeti:")
print(f"      ortalama ≈ {df_scaled['age'].mean():.4f}  (beklenen: ~0)")
print(f"      std      ≈ {df_scaled['age'].std():.4f}  (beklenen: ~1)")

# ═════════════════════════════════════════════════════════════
# 5. OneHotEncoder
# ═════════════════════════════════════════════════════════════
print("\n[5] OneHotEncoder ile kategorik dönüşüm...")

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_cat_df = pd.DataFrame(X_train_cat_imp, columns=cat_ohe + cat_ord)
X_test_cat_df  = pd.DataFrame(X_test_cat_imp,  columns=cat_ohe + cat_ord)

X_train_ohe = ohe.fit_transform(X_train_cat_df[cat_ohe])
X_test_ohe  = ohe.transform(X_test_cat_df[cat_ohe])

ohe_names = ohe.get_feature_names_out(cat_ohe)
print(f"    {len(cat_ohe)} kategorik sütun → {len(ohe_names)} binary sütun")
print(f"    Örnekler: {list(ohe_names[:6])} ...")

# handle_unknown='ignore' önemi
print("\n    handle_unknown='ignore' neden önemli?")
print("    Test setinde eğitimde görülmemiş bir kategori olabilir.")
print("    'ignore' → o sütun için tüm 0'lar üretilir, hata fırlatmaz.")

# ═════════════════════════════════════════════════════════════
# 6. OrdinalEncoder
# ═════════════════════════════════════════════════════════════
print("\n[6] OrdinalEncoder ile sıralı kategorik dönüşüm...")

education_sirasi = [["basic.4y", "basic.6y", "basic.9y",
                     "high.school", "professional.course",
                     "university.degree", "illiterate", "unknown"]]

ord_enc = OrdinalEncoder(
    categories=education_sirasi,
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)
X_train_ord = ord_enc.fit_transform(X_train_cat_df[cat_ord])
X_test_ord  = ord_enc.transform(X_test_cat_df[cat_ord])

print(f"    Education kategorileri → sayı eşlemeleri:")
for cat, val in zip(ord_enc.categories_[0], range(len(ord_enc.categories_[0]))):
    print(f"      {cat:<25} → {val}")

# ═════════════════════════════════════════════════════════════
# 7. GÖRSELLEŞTİRME
# ═════════════════════════════════════════════════════════════
print("\n[7] Görselleştirmeler hazırlanıyor...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Veri Ön İşleme — Etki Analizi", fontsize=15, fontweight="bold")

cols_plot = [c for c in ["age", "balance", "duration"] if c in num_cols]

for i, col in enumerate(cols_plot):
    cidx = num_cols.index(col)

    # Ham veri (eksik hariç)
    raw = X_train[col].dropna()
    axes[0, i].hist(raw, bins=35, color="#2563EB", alpha=0.78, edgecolor="white")
    axes[0, i].set_title(f"Ham: {col}", fontweight="bold", fontsize=11)
    axes[0, i].set_ylabel("Frekans" if i == 0 else "")
    axes[0, i].grid(axis="y", alpha=0.3)

    # İşlenmiş veri
    proc = X_train_scaled[:, cidx]
    axes[1, i].hist(proc, bins=35, color="#16A34A", alpha=0.78, edgecolor="white")
    axes[1, i].set_title(f"İşlenmiş: {col} (z-score)", fontweight="bold", fontsize=11)
    axes[1, i].set_ylabel("Frekans" if i == 0 else "")
    axes[1, i].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("01_on_isleme_etki.png", dpi=150, bbox_inches="tight")
print("    ✅ Grafik kaydedildi: 01_on_isleme_etki.png")
plt.close()

# ═════════════════════════════════════════════════════════════
# 8. FIT vs TRANSFORM — CANLI GÖSTERIM
# ═════════════════════════════════════════════════════════════
print("\n[8] fit() ile transform() farkı — canlı demo")
print("─" * 45)

demo = pd.DataFrame({
    "A": [1.0, 2.0, np.nan, 4.0, 5.0],
    "B": [10.0, np.nan, 30.0, 40.0, 50.0],
})
demo_yeni = pd.DataFrame({"A": [np.nan, 3.0], "B": [20.0, np.nan]})

imp_demo = SimpleImputer(strategy="mean")
imp_demo.fit(demo)

print("  Eğitim verisi:")
print(demo.to_string())
print(f"\n  Öğrenilen ortalamalar: A={imp_demo.statistics_[0]:.1f}, B={imp_demo.statistics_[1]:.1f}")

print("\n  Yeni veri (test):")
print(demo_yeni.to_string())
print("\n  transform() uygulandı (eğitim ortalamalarıyla):")
print(pd.DataFrame(imp_demo.transform(demo_yeni), columns=["A", "B"]).to_string())
print("\n  ↑ Test verisinin kendi ortalaması KULLANILMADI — Doğru!")

print("\n" + "=" * 60)
print("  ✅ UYGULAMA 01 TAMAMLANDI")
print("  Çıktılar: 01_on_isleme_etki.png")
print("=" * 60)
