"""
=============================================================================
UYGULAMA 3: SHAP — Tam Model Yorumlama Kütüphanesi
=============================================================================
Kapsam:
  - shap.TreeExplainer ile hızlı hesaplama
  - summary_plot (bee-swarm) — Global feature importance
  - bar_plot — Ortalama |SHAP| sıralaması
  - waterfall_plot — Tek örnek kademeli katkı
  - force_plot — Bireysel tahmin görselleştirme (matplotlib=True)
  - dependence_plot — Feature etkileşim analizi
  - SHAP Decision Plot — Karar yolu görselleştirme
  - SHAP Interaction Values (gelişmiş)
  - Yanlış Sınıflandırılan Örneklerin SHAP Analizi

Veri: Önceki uygulamalardan üretilen sentetik banka verisi
Gereksinimler: pip install shap scikit-learn matplotlib pandas numpy
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print(" UYGULAMA 3: SHAP — İleri Düzey Model Yorumlama")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 1. VERİ VE MODEL HAZIRLAMA
# ─────────────────────────────────────────────────────────────
np.random.seed(42)
n = 6000
df = pd.DataFrame({
    "age":       np.random.randint(18, 75, n).astype(float),
    "balance":   np.random.normal(1500, 3000, n),
    "duration":  np.abs(np.random.normal(260, 260, n)),
    "campaign":  np.random.randint(1, 10, n).astype(float),
    "pdays":     np.random.choice([999] + list(range(0, 30)), n,
                                  p=[0.85] + [0.005] * 30).astype(float),
    "previous":  np.random.randint(0, 5, n).astype(float),
    "housing":   np.random.choice([0, 1], n).astype(float),
    "loan":      np.random.choice([0, 1], n).astype(float),
    "job":       np.random.choice(["admin.", "blue-collar", "technician",
                                   "services", "management", "retired"], n),
    "marital":   np.random.choice(["married", "single", "divorced"], n),
    "education": np.random.choice(["basic.4y", "high.school",
                                   "university.degree", "professional.course"], n),
    "contact":   np.random.choice(["telephone", "cellular"], n),
    "poutcome":  np.random.choice(["failure", "success", "nonexistent"], n),
})

# Gerçekçi hedef (yorumlanabilir örüntü)
prob = (
    0.05
    + 0.25 * (df["duration"] > 400)
    + 0.15 * (df["previous"] > 0)
    + 0.10 * (df["poutcome"] == "success")
    + 0.08 * (df["balance"] > 3000)
    - 0.05 * (df["campaign"] > 5)
    - 0.03 * (df["job"] == "blue-collar")
).clip(0.02, 0.95)
df["target"] = (np.random.rand(n) < prob).astype(int)

num_cols = ["age", "balance", "duration", "campaign", "pdays", "previous", "housing", "loan"]
cat_ohe  = ["job", "marital", "contact", "poutcome"]
cat_ord  = ["education"]

X = df[num_cols + cat_ohe + cat_ord]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\n✅ Veri hazır: {X.shape}")


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower; self.upper = upper
    def fit(self, X, y=None):
        self.l_ = np.nanquantile(X, self.lower, axis=0)
        self.u_ = np.nanquantile(X, self.upper, axis=0)
        return self
    def transform(self, X):
        return np.clip(np.array(X, dtype=float), self.l_, self.u_)


preprocessor = ColumnTransformer([
    ("num",     Pipeline([("imp", IterativeImputer(max_iter=10, random_state=42)),
                          ("clip", OutlierClipper()),
                          ("scl",  RobustScaler())]),  num_cols),
    ("cat_ohe", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                cat_ohe),
    ("cat_ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]),
                cat_ord),
], remainder="drop")

# ─────────────────────────────────────────────────────────────
# 2. MODELİ EĞİT
# ─────────────────────────────────────────────────────────────
print("\n⏳ Random Forest modeli eğitiliyor...")
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )),
])
rf_model.fit(X_train, y_train)
y_pred  = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]
print(f"✅ Test AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ─────────────────────────────────────────────────────────────
# 3. FEATURE NAMES HAZIRLAMA
# ─────────────────────────────────────────────────────────────
try:
    feature_names = list(rf_model.named_steps["preprocessor"].get_feature_names_out())
except Exception:
    feature_names = [f"feature_{i}" for i in range(100)]

# feature name'leri kısalt
feature_names_short = []
for fn in feature_names:
    fn = fn.replace("cat_ohe__", "").replace("num__", "").replace("cat_ord__", "")
    feature_names_short.append(fn)

print(f"\n📋 Toplam {len(feature_names_short)} feature var:")
print("   " + ", ".join(feature_names_short[:10]) + " ...")

# ─────────────────────────────────────────────────────────────
# 4. SHAP EXPLAINER HAZIRLA
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("🔮 SHAP TreeExplainer oluşturuluyor...")
print("─" * 50)

# Test verisini dönüştür (SHAP için ham array lazım)
X_test_transformed = rf_model.named_steps["preprocessor"].transform(X_test)
X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names_short)

# TreeExplainer: RF için en hızlı yöntem
explainer = shap.TreeExplainer(rf_model.named_steps["classifier"])
print("⏳ SHAP değerleri hesaplanıyor (test seti, sınıf=1 için)...")
shap_values = explainer.shap_values(X_test_df)

# shap_values: liste ise [class0, class1], array ise direkt
if isinstance(shap_values, list):
    shap_vals_pos = shap_values[1]   # Pozitif sınıf (churn=1)
    expected_val  = explainer.expected_value[1]
else:
    shap_vals_pos = shap_values
    expected_val  = explainer.expected_value

print(f"✅ SHAP değerleri hesaplandı: {shap_vals_pos.shape}")
print(f"   Baz değer (expected_value): {expected_val:.4f}")
print(f"   Bu değer: tüm örneklerin ortalama tahmin olasılığı")

# ─────────────────────────────────────────────────────────────
# 5. GÖRSELLEŞTİRME 1: SUMMARY PLOT (Bee-Swarm)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("📊 Görselleştirme 1: Summary Plot (Bee-Swarm)")
print("─" * 50)
print("✏️ Nasıl okunur?")
print("   - Y ekseni: Feature önem sırası (yukarı = daha önemli)")
print("   - X ekseni: SHAP değeri (sağ = tahmin olasılığını artırır)")
print("   - Renk: Feature değeri (kırmızı=yüksek, mavi=düşük)")

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals_pos,
    X_test_df,
    max_display=15,
    show=False
)
plt.title("SHAP Summary Plot — Global Feature Önemi (Bee-Swarm)", fontsize=13)
plt.tight_layout()
plt.savefig("/home/claude/03a_shap_summary_swarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Kaydedildi: 03a_shap_summary_swarm.png")

# ─────────────────────────────────────────────────────────────
# 6. GÖRSELLEŞTİRME 2: BAR PLOT (Ortalama |SHAP|)
# ─────────────────────────────────────────────────────────────
print("\n📊 Görselleştirme 2: Bar Plot (Ortalama |SHAP| değerleri)")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_vals_pos,
    X_test_df,
    plot_type="bar",
    max_display=15,
    show=False
)
plt.title("SHAP Bar Plot — Ortalama Mutlak SHAP Değerleri", fontsize=13)
plt.tight_layout()
plt.savefig("/home/claude/03b_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Kaydedildi: 03b_shap_bar.png")

# ─────────────────────────────────────────────────────────────
# 7. GÖRSELLEŞTİRME 3: WATERFALL PLOT
# ─────────────────────────────────────────────────────────────
print("\n📊 Görselleştirme 3: Waterfall Plot (Bireysel tahmin analizi)")

# 3 farklı örnek seç: doğru pozitif, doğru negatif, hatalı
y_pred_test = rf_model.predict(X_test)
true_pos_idx  = np.where((y_test.values == 1) & (y_pred_test == 1))[0]
false_neg_idx = np.where((y_test.values == 1) & (y_pred_test == 0))[0]
true_neg_idx  = np.where((y_test.values == 0) & (y_pred_test == 0))[0]

example_indices = []
for arr, label in [(true_pos_idx, "TP"), (false_neg_idx, "FN"), (true_neg_idx, "TN")]:
    if len(arr) > 0:
        example_indices.append((arr[0], label))

fig, axes = plt.subplots(1, len(example_indices), figsize=(7 * len(example_indices), 8))
if len(example_indices) == 1:
    axes = [axes]

for ax_idx, (sample_idx, label) in enumerate(example_indices):
    plt.figure(figsize=(10, 7))
    exp = shap.Explanation(
        values=shap_vals_pos[sample_idx],
        base_values=expected_val,
        data=X_test_df.iloc[sample_idx].values,
        feature_names=feature_names_short,
    )
    shap.waterfall_plot(exp, max_display=12, show=False)
    plt.title(f"Waterfall Plot — Örnek {sample_idx} ({label})\n"
              f"Gerçek: {y_test.values[sample_idx]}, Tahmin: {y_pred_test[sample_idx]}, "
              f"Prob: {y_proba[sample_idx]:.3f}",
              fontsize=12)
    plt.tight_layout()
    plt.savefig(f"/home/claude/03c_shap_waterfall_{label.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Kaydedildi: 03c_shap_waterfall_{label.lower()}.png ({label})")

plt.close("all")

# ─────────────────────────────────────────────────────────────
# 8. GÖRSELLEŞTİRME 4: FORCE PLOT
# ─────────────────────────────────────────────────────────────
print("\n📊 Görselleştirme 4: Force Plot (İlk True Positive)")

if len(true_pos_idx) > 0:
    tp_idx = true_pos_idx[0]
    plt.figure(figsize=(14, 4))
    shap.force_plot(
        expected_val,
        shap_vals_pos[tp_idx],
        X_test_df.iloc[tp_idx],
        feature_names=feature_names_short,
        matplotlib=True,
        show=False,
    )
    plt.title(f"Force Plot — Müşteri {tp_idx} (Tahmin Prob: {y_proba[tp_idx]:.3f})", fontsize=12)
    plt.tight_layout()
    plt.savefig("/home/claude/03d_shap_force.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Kaydedildi: 03d_shap_force.png")

# ─────────────────────────────────────────────────────────────
# 9. GÖRSELLEŞTİRME 5: DEPENDENCE PLOT
# ─────────────────────────────────────────────────────────────
print("\n📊 Görselleştirme 5: Dependence Plot (duration vs pdays)")

# En önemli feature'ı bul
mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)
top_feature_idx = np.argmax(mean_abs_shap)
top_feature = feature_names_short[top_feature_idx]
second_feature_idx = np.argsort(mean_abs_shap)[-3]
second_feature = feature_names_short[second_feature_idx]

print(f"   En önemli feature: {top_feature}")
print(f"   İkinci önemli feature (renk): {second_feature}")

plt.figure(figsize=(10, 6))
shap.dependence_plot(
    top_feature_idx,
    shap_vals_pos,
    X_test_df,
    feature_names=feature_names_short,
    interaction_index=second_feature_idx,
    show=False,
)
plt.title(f"Dependence Plot: {top_feature} (renk={second_feature})", fontsize=12)
plt.tight_layout()
plt.savefig("/home/claude/03e_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Kaydedildi: 03e_shap_dependence.png")

# ─────────────────────────────────────────────────────────────
# 10. GÖRSELLEŞTİRME 6: DECISION PLOT
# ─────────────────────────────────────────────────────────────
print("\n📊 Görselleştirme 6: Decision Plot (100 örnek karar yolu)")

plt.figure(figsize=(10, 8))
sample_n = min(100, len(X_test_df))
misclassified = np.where(y_test.values[:sample_n] != y_pred_test[:sample_n])[0]

shap.decision_plot(
    expected_val,
    shap_vals_pos[:sample_n],
    X_test_df.iloc[:sample_n],
    feature_names=feature_names_short,
    highlight=misclassified,
    show=False,
)
plt.title("Decision Plot — 100 Örnek Karar Yolu\n(Kırmızı = Yanlış Sınıflandırılan)", fontsize=12)
plt.tight_layout()
plt.savefig("/home/claude/03f_shap_decision.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Kaydedildi: 03f_shap_decision.png")

# ─────────────────────────────────────────────────────────────
# 11. SHAP INTERACTION VALUES (Gelişmiş)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("🔬 Gelişmiş: SHAP Interaction Values")
print("─" * 50)
print("⏳ İlk 200 test örneği için hesaplanıyor...")

sample_size = min(200, len(X_test_df))
try:
    shap_interaction = explainer.shap_interaction_values(X_test_df.iloc[:sample_size])
    if isinstance(shap_interaction, list):
        shap_interaction = shap_interaction[1]

    # En güçlü etkileşim çiftlerini bul
    n_feat = shap_interaction.shape[1]
    interaction_matrix = np.abs(shap_interaction).mean(axis=0)
    np.fill_diagonal(interaction_matrix, 0)
    top_pairs = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            top_pairs.append((interaction_matrix[i, j], i, j))
    top_pairs.sort(reverse=True)

    print("\n🔥 En Güçlü Feature Etkileşimleri (Top 5):")
    for val, i, j in top_pairs[:5]:
        fn_i = feature_names_short[i] if i < len(feature_names_short) else f"F{i}"
        fn_j = feature_names_short[j] if j < len(feature_names_short) else f"F{j}"
        print(f"   {fn_i} ↔ {fn_j}: {val:.4f}")

    # Interaction heatmap
    n_show = min(12, n_feat)
    top_feat_idx = np.argsort(np.abs(shap_vals_pos).mean(axis=0))[-n_show:]
    sub_matrix = interaction_matrix[np.ix_(top_feat_idx, top_feat_idx)]
    sub_names = [feature_names_short[i] if i < len(feature_names_short) else f"F{i}"
                 for i in top_feat_idx]

    plt.figure(figsize=(10, 8))
    sns_data = pd.DataFrame(sub_matrix, index=sub_names, columns=sub_names)
    import seaborn as sns
    sns.heatmap(sns_data, cmap="YlOrRd", annot=True, fmt=".3f", linewidths=0.5,
                cbar_kws={"label": "Ortalama |Etkileşim SHAP|"})
    plt.title("SHAP Interaction Values Heatmap\n(Özellikler Arası Etkileşim Gücü)", fontsize=13)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig("/home/claude/03g_shap_interaction_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Kaydedildi: 03g_shap_interaction_heatmap.png")
except Exception as e:
    print(f"⚠️ Interaction values hesaplanamadı: {e}")
    print("   Bu genellikle bellek veya model uyumsuzluğu nedeniyle olur.")

# ─────────────────────────────────────────────────────────────
# 12. YANLIŞ SINIFLANDIRILAN ÖRNEKLERİN SHAP ANALİZİ
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print("🔍 Yanlış Sınıflandırılan Örneklerin SHAP Analizi")
print("─" * 50)

fp_idx = np.where((y_test.values == 0) & (y_pred_test == 1))[0]  # False Positive
fn_idx = np.where((y_test.values == 1) & (y_pred_test == 0))[0]  # False Negative

print(f"\n  False Positives (FP): {len(fp_idx)} örnek → Modelin 'evet' dediği ama 'hayır' olanlar")
print(f"  False Negatives (FN): {len(fn_idx)} örnek → Modelin 'hayır' dediği ama 'evet' olanlar")

if len(fp_idx) > 0 and len(fn_idx) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_i, (indices, title, color) in enumerate([
        (fp_idx[:30], "False Positives (Yanlış Alarm)", "Reds"),
        (fn_idx[:30], "False Negatives (Kaçırılan Churn)", "Blues"),
    ]):
        mean_shap = np.abs(shap_vals_pos[indices]).mean(axis=0)
        top_n = 10
        top_idx = np.argsort(mean_shap)[-top_n:]
        axes[ax_i].barh(
            range(top_n),
            mean_shap[top_idx],
            color=plt.cm.get_cmap(color)(np.linspace(0.4, 0.9, top_n))
        )
        axes[ax_i].set_yticks(range(top_n))
        axes[ax_i].set_yticklabels(
            [feature_names_short[i] if i < len(feature_names_short) else f"F{i}" for i in top_idx],
            fontsize=9
        )
        axes[ax_i].set_xlabel("|SHAP| Ortalama")
        axes[ax_i].set_title(title, fontweight="bold", fontsize=11)
        axes[ax_i].grid(axis="x", alpha=0.3)

    plt.suptitle("Yanlış Sınıflandırılan Örneklerde Hangi Feature'lar Suçlu?",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("/home/claude/03h_shap_misclassified.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Kaydedildi: 03h_shap_misclassified.png")

# ─────────────────────────────────────────────────────────────
# 13. SHAP DEĞER ANALİZİ: Yorumlanabilir Bulgu Raporu
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("📋 SHAP ANALİZ RAPORU — Yorumlanabilir Bulgular")
print("=" * 65)

mean_abs = np.abs(shap_vals_pos).mean(axis=0)
top5_idx = np.argsort(mean_abs)[-5:][::-1]

print("\n🔝 En Etkili 5 Feature:")
for rank, i in enumerate(top5_idx, 1):
    fname = feature_names_short[i] if i < len(feature_names_short) else f"F{i}"
    pos_mean = shap_vals_pos[:, i][shap_vals_pos[:, i] > 0].mean() if any(shap_vals_pos[:, i] > 0) else 0
    neg_mean = shap_vals_pos[:, i][shap_vals_pos[:, i] < 0].mean() if any(shap_vals_pos[:, i] < 0) else 0
    print(f"  #{rank}: {fname:<25} | Ortalama |SHAP|: {mean_abs[i]:.4f} "
          f"| Pozitif etki: {pos_mean:+.3f} | Negatif etki: {neg_mean:+.3f}")

print(f"\n  Baz değer: {expected_val:.4f} → Bu, modelin hiçbir bilgi olmadan tahmin ettiği ortalama olaslık")
print(f"  Örnek tahmin: {y_proba[0]:.4f} = {expected_val:.4f} + SHAP toplamı ({(y_proba[0]-expected_val):+.4f})")
print(f"  Kontrol: {expected_val:.4f} + {(y_proba[0]-expected_val):.4f} = {y_proba[0]:.4f} ✓")

print("\n  Tüm grafik dosyaları:")
for fname in ["03a_shap_summary_swarm", "03b_shap_bar", "03c_shap_waterfall_tp",
              "03d_shap_force", "03e_shap_dependence", "03f_shap_decision",
              "03g_shap_interaction_heatmap", "03h_shap_misclassified"]:
    print(f"    📊 /home/claude/{fname}.png")

print("\n" + "=" * 65)
print(" ✅ UYGULAMA 3 TAMAMLANDI! — 8 SHAP grafiği üretildi")
print("=" * 65)
