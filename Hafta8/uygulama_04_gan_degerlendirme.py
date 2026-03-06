"""
=============================================================================
HAFTA 4 PAZAR — UYGULAMA 04
GAN Değerlendirme & Sorunlar: FID, IS, Mode Collapse, Denge Analizi
=============================================================================
Kapsam:
  - FID (Fréchet Inception Distance): gerçek vs üretilmiş dağılım farkı
  - Inception Score (IS): kalite × çeşitlilik metriği
  - Mode Collapse tespiti ve önleme stratejileri (WGAN, Spektral Norm, MiniMax)
  - Discriminator / Generator denge analizi (Nash dengesi takibi)
  - VAE vs GAN görsel kalite karşılaştırması (FID bazlı)
  - Çok sınıflı üretim çeşitliliği (10 Fashion-MNIST sınıfı)
  - Hiperparametre ablasyonu: lr, β₁, latent_dim etkisi
  - 9 grafiklik kapsamlı görselleştirme
  - Simülasyon modunda tamamen çalışır

Kurulum: pip install numpy matplotlib scikit-learn scipy
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

try:
    from scipy.stats import entropy
    from scipy.linalg import sqrtm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    def gaussian_filter(x, sigma=1): return x

print("=" * 65)
print("  HAFTA 4 PAZAR — UYGULAMA 04")
print("  GAN Değerlendirme & Sorunlar")
print("=" * 65)
print(f"  scipy  : {'✅' if SCIPY_AVAILABLE else '❌  pip install scipy'}")
print(f"  sklearn: {'✅' if SKLEARN_AVAILABLE else '❌  pip install scikit-learn'}")
print()

# ─────────────────────────────────────────────────────────────
# BÖLÜM 1: FID HESAPLAMA
# ─────────────────────────────────────────────────────────────
print("─" * 65)
print("  BÖLÜM 1: FID (Fréchet Inception Distance)")
print("─" * 65)
print("""
  FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2√(Σ_r·Σ_g))

  μ_r, Σ_r : Gerçek görüntülerin Inception özelliklerinin istatistikleri
  μ_g, Σ_g : Üretilen görüntülerin Inception özelliklerinin istatistikleri

  FID ↓ → Daha iyi (dağılımlar birbirine yakın)
  FID = 0 → Mükemmel (üretilen = gerçek dağılımı)
  FID > 200 → Kötü kalite
""")

def fid_hesapla(mu1, sigma1, mu2, sigma2):
    """FID hesaplama (özellik uzayında)."""
    diff = mu1 - mu2
    diff_sq = np.dot(diff, diff)
    if SCIPY_AVAILABLE:
        kovmatris_carpim = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(kovmatris_carpim):
            kovmatris_carpim = kovmatris_carpim.real
    else:
        # sqrtm yaklaşımı: Newton-Schulz iterasyonu (5 adım)
        A   = sigma1.dot(sigma2)
        Y   = A.copy()
        Z   = np.eye(len(A))
        for _ in range(10):
            Y_new = 0.5 * (Y + np.linalg.solve(Z.T, A).T)
            Z_new = 0.5 * (Z + np.linalg.solve(Y.T, A).T)
            Y, Z  = Y_new, Z_new
        kovmatris_carpim = Y
    iz = np.trace(sigma1 + sigma2 - 2 * kovmatris_carpim)
    return float(diff_sq + iz)

def fid_sim_hesapla(model_adi, epoch, kalite_carpan=1.0):
    """Epoch ve model kalitesine göre FID simüle eder."""
    np.random.seed(abs(hash(model_adi + str(epoch))) % 2**20)
    # FID epoch arttıkça azalır (kalite artar)
    temel_fid = 280 * np.exp(-0.06 * epoch) + 30 * kalite_carpan
    gurultu   = np.random.normal(0, temel_fid * 0.06)
    return max(8.0, float(temel_fid + gurultu))

MODELLER_FID = {
    "DCGAN (baseline)":   {"renk": "#0D9488", "carpan": 1.0},
    "DCGAN + Spec. Norm": {"renk": "#6D28D9", "carpan": 0.78},
    "WGAN-GP":            {"renk": "#059669", "carpan": 0.68},
    "VAE (referans)":     {"renk": "#D97706", "carpan": 2.10},
}

KONTROL_NOKTALARI = [1, 5, 10, 20, 30, 40, 50]

fid_tarih = {m: [] for m in MODELLER_FID}
for model_adi, bilgi in MODELLER_FID.items():
    for ep in range(1, 51):
        fid_tarih[model_adi].append(
            fid_sim_hesapla(model_adi, ep, bilgi["carpan"])
        )

print(f"  {'Model':<25} {'FID @ Ep5':>10} {'FID @ Ep20':>10}"
      f" {'FID @ Ep50':>10}")
print("  " + "-" * 60)
for model_adi in MODELLER_FID:
    print(f"  {model_adi:<25} {fid_tarih[model_adi][4]:>10.1f}"
          f" {fid_tarih[model_adi][19]:>10.1f}"
          f" {fid_tarih[model_adi][49]:>10.1f}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 2: INCEPTION SCORE (IS)
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 2: Inception Score (IS)")
print("─" * 65)
print("""
  IS = exp(𝔼_x [KL(p(y|x) || p(y))])

  p(y|x) : Inception'ın üretilen x için sınıf tahminleri
  p(y)   : Tüm üretilen görüntülerin ortalama tahminleri

  IS ↑ → Daha iyi  (kalite VE çeşitlilik yüksek)
  Kalite  : p(y|x) keskin (model belirgin sınıf üretiyor)
  Çeşitlilik: p(y) düz (tüm sınıflar üretiliyor)

  Mode Collapse: p(y) ← tek sınıf → IS düşer
""")

def inception_score_hesapla(p_y_verilen_x):
    """
    p_y_verilen_x: (n_görüntü, n_sınıf) — sınıf olasılıkları
    """
    n_sinif  = p_y_verilen_x.shape[1]
    p_y      = p_y_verilen_x.mean(axis=0)
    p_y      = np.clip(p_y, 1e-8, 1.0)
    kl_ler   = []
    for i in range(len(p_y_verilen_x)):
        p_yi  = np.clip(p_y_verilen_x[i], 1e-8, 1.0)
        kl    = np.sum(p_yi * (np.log(p_yi) - np.log(p_y)))
        kl_ler.append(kl)
    return float(np.exp(np.mean(kl_ler)))

def is_sim_hesapla(model_adi, epoch, kalite_carpan=1.0):
    """IS simülasyonu."""
    np.random.seed(abs(hash("IS" + model_adi + str(epoch))) % 2**20)
    n_goruntu = 200
    n_sinif   = 10
    # Epoch arttıkça kalite artar → IS artar
    guc = min(2.0 + epoch * 0.06 * (1 / kalite_carpan), 8.0)
    # Sınıf dağılımı: kalite arttıkça daha iyi ayrılır
    p_y_x = np.random.dirichlet(
        [guc] * n_sinif,
        size=n_goruntu
    )
    return inception_score_hesapla(p_y_x)

IS_tarih = {m: [] for m in MODELLER_FID}
for model_adi, bilgi in MODELLER_FID.items():
    for ep in range(1, 51):
        IS_tarih[model_adi].append(
            is_sim_hesapla(model_adi, ep, bilgi["carpan"])
        )

print(f"  {'Model':<25} {'IS @ Ep5':>10} {'IS @ Ep20':>10}"
      f" {'IS @ Ep50':>10}")
print("  " + "-" * 60)
for model_adi in MODELLER_FID:
    print(f"  {model_adi:<25} {IS_tarih[model_adi][4]:>10.3f}"
          f" {IS_tarih[model_adi][19]:>10.3f}"
          f" {IS_tarih[model_adi][49]:>10.3f}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 3: MODE COLLAPSE ANALİZİ
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 3: Mode Collapse Senaryoları")
print("─" * 65)
print("""
  3 Senaryo:
    ① Sağlıklı eğitim  : Tüm 10 sınıf dengeli üretiliyor
    ② Kısmi collapse   : 3-4 sınıf dominant, diğerleri nadir
    ③ Tam collapse      : Tek sınıf (veya çok az çeşitlilik)

  Önleme Stratejileri:
    - WGAN-GP (Wasserstein mesafesi + gradient penalty)
    - Spectral Normalization (D ağırlıklarını normalize eder)
    - Mini-batch discrimination (D'ye çeşitlilik bilgisi ver)
    - Feature matching (G'nin ara özelliklerini eşleştir)
    - Unrolled GAN (G, D'nin gelecek adımlarını tahmin eder)
""")

def collapse_senaryosu_uret(senaryo, n_goruntu=300):
    """Her senaryo için sınıf dağılımı üretir."""
    np.random.seed(senaryo * 17)
    if senaryo == 0:   # Sağlıklı
        dag = np.ones(10) * 30 + np.random.uniform(-5, 5, 10)
    elif senaryo == 1: # Kısmi
        dag = np.array([2, 2, 2, 60, 55, 3, 52, 2, 2, 1], dtype=float)
        dag += np.random.uniform(0, 3, 10)
    else:              # Tam
        dag = np.array([0, 0, 0, 0, 0, 290, 5, 2, 2, 1], dtype=float)
        dag += np.random.uniform(0, 2, 10)
    dag = dag / dag.sum() * n_goruntu
    return dag.astype(int)

SENARYO_ADLAR = ["Sağlıklı Eğitim", "Kısmi Collapse", "Tam Collapse"]
senaryo_daglari = [collapse_senaryosu_uret(i) for i in range(3)]

print(f"  {'Sınıf':<20}", end="")
for ad in SENARYO_ADLAR:
    print(f" {ad:>18}", end="")
print()
print("  " + "-" * 78)
for si, sinif in enumerate(["T-shirt","Trouser","Pullover","Dress","Coat",
                              "Sandal","Shirt","Sneaker","Bag","Ankle boot"]):
    print(f"  {sinif:<20}", end="")
    for dag in senaryo_daglari:
        bar = "█" * int(dag[si] / 10)
        print(f" {dag[si]:>3} {bar:<15}", end="")
    print()

# ─────────────────────────────────────────────────────────────
# BÖLÜM 4: HİPERPARAMETRE ABLASYONU
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 4: Hiperparametre Ablasyonu")
print("─" * 65)

ablasyon_sonuclari = []

# Öğrenme hızı ablasyonu
lr_deneyleri = [1e-3, 5e-4, 2e-4, 1e-4]
print(f"\n  Öğrenme Hızı (β₁=0.5, z_dim=100):")
print(f"  {'lr':>10} {'FID@50':>10} {'IS@50':>8} {'Kararlılık':>14}")
print("  " + "-" * 48)
for lr in lr_deneyleri:
    np.random.seed(int(lr * 1e7) % 999)
    # Düşük lr → yavaş ama kararlı; yüksek lr → hızlı ama dengesiz
    kalite_f = 1.0 if lr == 2e-4 else (0.8 if lr == 1e-4 else 1.3 if lr == 5e-4 else 1.9)
    fid = fid_sim_hesapla(f"lr_{lr}", 50, kalite_f)
    is_ = is_sim_hesapla(f"lr_{lr}", 50, kalite_f)
    kararlilik = "✅ İdeal" if lr == 2e-4 else ("⚠️ Yavaş" if lr == 1e-4
                                                 else ("⚠️ Orta" if lr == 5e-4 else "❌ Kararsız"))
    print(f"  {lr:>10.0e} {fid:>10.1f} {is_:>8.3f} {kararlilik:>14}")
    ablasyon_sonuclari.append({"tip": "lr", "deger": lr, "fid": fid, "is": is_})

# Beta_1 ablasyonu
beta1_deneyleri = [0.9, 0.7, 0.5, 0.3]
print(f"\n  Adam β₁ (lr=2e-4, z_dim=100):")
print(f"  {'β₁':>8} {'FID@50':>10} {'IS@50':>8} {'Yorum':>18}")
print("  " + "-" * 50)
for b1 in beta1_deneyleri:
    np.random.seed(int(b1 * 100))
    kalite_f = 0.85 if b1 == 0.5 else (1.1 if b1 == 0.3 else 1.4 if b1 == 0.7 else 1.8)
    fid = fid_sim_hesapla(f"b1_{b1}", 50, kalite_f)
    is_ = is_sim_hesapla(f"b1_{b1}", 50, kalite_f)
    yorum = "✅ Standart DCGAN" if b1 == 0.5 else ("Kararsız" if b1 == 0.9 else "Yavaş" if b1 == 0.3 else "Orta")
    print(f"  {b1:>8.1f} {fid:>10.1f} {is_:>8.3f} {yorum:>18}")

# Latent dim ablasyonu
ldim_deneyleri = [32, 64, 100, 256]
print(f"\n  Latent Boyut (lr=2e-4, β₁=0.5):")
print(f"  {'z_dim':>8} {'FID@50':>10} {'IS@50':>8} {'Yorum':>24}")
print("  " + "-" * 56)
for ld in ldim_deneyleri:
    np.random.seed(ld)
    kalite_f = 0.9 if ld == 100 else (1.2 if ld == 64 else 1.5 if ld == 32 else 0.85)
    fid = fid_sim_hesapla(f"ld_{ld}", 50, kalite_f)
    is_ = is_sim_hesapla(f"ld_{ld}", 50, kalite_f)
    yorum = "✅ Standart" if ld == 100 else ("Kısıtlı çeşit" if ld == 32 else "İyi" if ld == 64 else "Zengin latent")
    print(f"  {ld:>8} {fid:>10.1f} {is_:>8.3f} {yorum:>24}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 5: VAE vs GAN KARŞILAŞTIRMA
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 5: VAE vs GAN Karşılaştırması")
print("─" * 65)

def pca_ozellikleri_sim(n_goruntu, sinif_sayisi=10, kalite=1.0, seed=0):
    """PCA özellik uzayında gerçek/üretilmiş dağılım simülasyonu."""
    np.random.seed(seed)
    merkezler  = np.random.randn(sinif_sayisi, 2) * 3
    gercek_X, gercek_y = [], []
    for si in range(sinif_sayisi):
        n = n_goruntu // sinif_sayisi
        X = np.random.multivariate_normal(
            merkezler[si], [[0.4, 0.05], [0.05, 0.4]], n
        )
        gercek_X.append(X)
        gercek_y.extend([si] * n)

    uretilen_X, uretilen_y = [], []
    for si in range(sinif_sayisi):
        n = n_goruntu // sinif_sayisi
        # Kalite arttıkça üretilen, gerçeğe yaklaşır
        sapma   = (1 / kalite) * 0.8
        X = np.random.multivariate_normal(
            merkezler[si] + np.random.randn(2) * sapma,
            [[0.4 + sapma * 0.2, 0.05], [0.05, 0.4 + sapma * 0.2]],
            n
        )
        uretilen_X.append(X)
        uretilen_y.extend([si] * n)

    return (np.vstack(gercek_X), np.array(gercek_y),
            np.vstack(uretilen_X), np.array(uretilen_y))

# 3 model karşılaştırması: VAE, GAN, WGAN
pca_modeller = {
    "VAE":     pca_ozellikleri_sim(500, kalite=0.55, seed=1),
    "DCGAN":   pca_ozellikleri_sim(500, kalite=1.20, seed=2),
    "WGAN-GP": pca_ozellikleri_sim(500, kalite=1.65, seed=3),
}

SINIF_RENKLERI = [
    "#EF4444","#F97316","#EAB308","#22C55E","#14B8A6",
    "#3B82F6","#8B5CF6","#EC4899","#6B7280","#0EA5E9"
]

print(f"  {'Model':<12} {'FID@50':>10} {'IS@50':>10}")
print("  " + "-" * 36)
for model_adi in ["VAE","DCGAN","WGAN-GP"]:
    c = {"VAE": 2.1, "DCGAN": 1.0, "WGAN-GP": 0.68}[model_adi]
    f = fid_sim_hesapla(model_adi, 50, c)
    i = is_sim_hesapla(model_adi, 50, c)
    print(f"  {model_adi:<12} {f:>10.1f} {i:>10.3f}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 6: GÖRSELLEŞTİRME (9 panel)
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 6: Görselleştirme (9 panel)")
print("─" * 65)

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor("#F0FDF4")
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.44, wspace=0.36,
                        top=0.93, bottom=0.05)

ep_arr = np.arange(1, 51)

# ── GRAFİK 1: FID Eğrisi ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("white")
for model_adi, bilgi in MODELLER_FID.items():
    stil = "-" if "WGAN" in model_adi or "Spec" in model_adi else "--" if "VAE" in model_adi else "-"
    ax1.plot(ep_arr, fid_tarih[model_adi], color=bilgi["renk"],
             linewidth=2.0, linestyle=stil,
             label=f"{model_adi} ({fid_tarih[model_adi][-1]:.0f})")
ax1.axhline(y=30, color="#94A3B8", linestyle=":", linewidth=1.2, alpha=0.8,
            label="İyi eşik ≈ 30")
ax1.set_title("FID Skoru vs Epoch\n(Düşük = Daha İyi)", fontsize=13, fontweight="bold", pad=10)
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("FID ↓", fontsize=11)
ax1.legend(fontsize=8.5, loc="upper right")
ax1.grid(alpha=0.4)
ax1.set_yscale("log")

# ── GRAFİK 2: IS Eğrisi ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("white")
for model_adi, bilgi in MODELLER_FID.items():
    stil = "-" if "WGAN" in model_adi or "Spec" in model_adi else "--" if "VAE" in model_adi else "-"
    ax2.plot(ep_arr, IS_tarih[model_adi], color=bilgi["renk"],
             linewidth=2.0, linestyle=stil,
             label=f"{model_adi} ({IS_tarih[model_adi][-1]:.2f})")
ax2.set_title("Inception Score vs Epoch\n(Yüksek = Daha İyi)", fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("IS ↑", fontsize=11)
ax2.legend(fontsize=8.5, loc="lower right")
ax2.grid(alpha=0.4)

# ── GRAFİK 3: Mode Collapse Sınıf Dağılımı ───────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("white")
x_pos    = np.arange(10)
gen3     = 0.25
ren3     = ["#22C55E", "#F97316", "#EF4444"]
for ki, (dag, ad, renk) in enumerate(zip(senaryo_daglari, SENARYO_ADLAR, ren3)):
    offset = (ki - 1) * gen3
    ax3.bar(x_pos + offset, dag, gen3, color=renk, alpha=0.82,
            label=ad, edgecolor="white")
ax3.axhline(y=30, color="#94A3B8", linestyle="--", linewidth=1.5,
            label="İdeal denge (=30)")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(["T-sh","Trou","Pull","Dres","Coat",
                     "Sand","Shir","Snek","Bag","Boot"], fontsize=8, rotation=18)
ax3.set_title("Mode Collapse Senaryoları\n(Sınıf Başına Üretim Sayısı)",
              fontsize=12, fontweight="bold", pad=10)
ax3.set_ylabel("Üretim Sayısı / 300", fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.4)

# ── GRAFİK 4: PCA — VAE özellikleri ─────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor("#FAFAFA")
gercek_X, gercek_y, uret_X, uret_y = pca_modeller["VAE"]
for si in range(10):
    maske = gercek_y == si
    ax4.scatter(gercek_X[maske, 0], gercek_X[maske, 1],
                c=SINIF_RENKLERI[si], s=12, alpha=0.5, marker="o")
for si in range(10):
    maske = uret_y == si
    ax4.scatter(uret_X[maske, 0], uret_X[maske, 1],
                c=SINIF_RENKLERI[si], s=20, alpha=0.6, marker="x", linewidths=1.2)
ax4.set_title("PCA Özellik Uzayı — VAE\n(○=Gerçek  ×=Üretilen)",
              fontsize=12, fontweight="bold", pad=8)
ax4.set_xlabel("PC₁", fontsize=10); ax4.set_ylabel("PC₂", fontsize=10)
fid_vae = fid_sim_hesapla("VAE", 50, 2.1)
ax4.text(0.03, 0.96, f"FID={fid_vae:.1f}", transform=ax4.transAxes,
         fontsize=11, fontweight="bold", color="#D97706",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D97706"))
ax4.grid(alpha=0.3)

# ── GRAFİK 5: PCA — DCGAN özellikleri ────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor("#FAFAFA")
gercek_X, gercek_y, uret_X, uret_y = pca_modeller["DCGAN"]
for si in range(10):
    maske = gercek_y == si
    ax5.scatter(gercek_X[maske, 0], gercek_X[maske, 1],
                c=SINIF_RENKLERI[si], s=12, alpha=0.5, marker="o")
for si in range(10):
    maske = uret_y == si
    ax5.scatter(uret_X[maske, 0], uret_X[maske, 1],
                c=SINIF_RENKLERI[si], s=20, alpha=0.7, marker="x", linewidths=1.2)
ax5.set_title("PCA Özellik Uzayı — DCGAN\n(○=Gerçek  ×=Üretilen)",
              fontsize=12, fontweight="bold", pad=8)
ax5.set_xlabel("PC₁", fontsize=10); ax5.set_ylabel("PC₂", fontsize=10)
fid_dcgan = fid_sim_hesapla("DCGAN", 50, 1.0)
ax5.text(0.03, 0.96, f"FID={fid_dcgan:.1f}", transform=ax5.transAxes,
         fontsize=11, fontweight="bold", color="#0D9488",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#0D9488"))
ax5.grid(alpha=0.3)

# ── GRAFİK 6: PCA — WGAN-GP özellikleri ─────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("#FAFAFA")
gercek_X, gercek_y, uret_X, uret_y = pca_modeller["WGAN-GP"]
for si in range(10):
    maske = gercek_y == si
    ax6.scatter(gercek_X[maske, 0], gercek_X[maske, 1],
                c=SINIF_RENKLERI[si], s=12, alpha=0.5, marker="o")
for si in range(10):
    maske = uret_y == si
    ax6.scatter(uret_X[maske, 0], uret_X[maske, 1],
                c=SINIF_RENKLERI[si], s=20, alpha=0.75, marker="x", linewidths=1.2)
ax6.set_title("PCA Özellik Uzayı — WGAN-GP\n(○=Gerçek  ×=Üretilen)",
              fontsize=12, fontweight="bold", pad=8)
ax6.set_xlabel("PC₁", fontsize=10); ax6.set_ylabel("PC₂", fontsize=10)
fid_wgan = fid_sim_hesapla("WGAN-GP", 50, 0.68)
ax6.text(0.03, 0.96, f"FID={fid_wgan:.1f}", transform=ax6.transAxes,
         fontsize=11, fontweight="bold", color="#059669",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#059669"))
ax6.grid(alpha=0.3)

# ── GRAFİK 7: lr Ablasyon ────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
ax7.set_facecolor("white")
lr_etiketler = ["1e-3","5e-4","2e-4","1e-4"]
lr_fidler = [ablasyon_sonuclari[i]["fid"] for i in range(4)]
lr_isler  = [ablasyon_sonuclari[i]["is"]  for i in range(4)]
x7 = np.arange(4)
ax7b = ax7.twinx()
b_fid = ax7.bar(x7 - 0.18, lr_fidler, 0.32, color="#6D28D9", alpha=0.8,
                label="FID ↓", edgecolor="white")
ax7b.plot(x7, lr_isler, "o-", color="#D97706", linewidth=2.2,
          markersize=9, label="IS ↑")
ax7.set_xticks(x7)
ax7.set_xticklabels([f"lr={l}" for l in lr_etiketler], fontsize=10)
ax7.set_ylabel("FID ↓", fontsize=10, color="#6D28D9")
ax7b.set_ylabel("IS ↑", fontsize=10, color="#D97706")
ax7.set_title("Öğrenme Hızı Ablasyonu\n(β₁=0.5, z_dim=100, epoch=50)",
              fontsize=12, fontweight="bold", pad=10)
lines_a = [b_fid] + ax7b.get_lines()
lbls_a  = ["FID ↓", "IS ↑"]
ax7.legend(lines_a, lbls_a, fontsize=10, loc="upper left")
ax7.grid(axis="y", alpha=0.4)
# lr=2e-4 vurgula
ax7.bar(2 - 0.18, lr_fidler[2], 0.32, color="#6D28D9",
        edgecolor="#FFD700", linewidth=2.5, label="_")

# ── GRAFİK 8: FID vs IS scatter (tüm modeller) ───────────────
ax8 = fig.add_subplot(gs[2, 1])
ax8.set_facecolor("white")
for model_adi, bilgi in MODELLER_FID.items():
    fid_son = fid_tarih[model_adi][-1]
    is_son  = IS_tarih[model_adi][-1]
    ax8.scatter(fid_son, is_son, s=220, color=bilgi["renk"],
                zorder=5, edgecolors="white", linewidth=1.5)
    ax8.annotate(model_adi.replace(" (baseline)", ""),
                 xy=(fid_son, is_son),
                 xytext=(fid_son + 2.5, is_son + 0.02),
                 fontsize=9.5, color=bilgi["renk"], fontweight="bold")
ax8.set_title("FID vs IS — Model Karşılaştırması\n(Sol-üst köşe = İdeal)",
              fontsize=12, fontweight="bold", pad=10)
ax8.set_xlabel("FID ↓ (Düşük = İyi)", fontsize=11)
ax8.set_ylabel("IS ↑ (Yüksek = İyi)", fontsize=11)
ax8.invert_xaxis()
ax8.grid(alpha=0.4)
# Hedef bölge vurgula
ax8.axvspan(ax8.get_xlim()[0] if ax8.get_xlim()[0] < 60 else 20, 60,
            alpha=0.06, color="#22C55E")
ax8.text(0.72, 0.08, "İdeal\nBölge", transform=ax8.transAxes,
         fontsize=9, color="#22C55E", fontweight="bold", ha="center")

# ── GRAFİK 9: Önleme Stratejileri Karşılaştırma ──────────────
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis("off")
STRATEJILER = [
    ("Vanilla GAN",         "—",       "Mode collapse ❌\nVanishing grad. ❌"),
    ("DCGAN",               "Conv arch","Kararlı ✅\nMode coll. riski ⚠️"),
    ("WGAN",                "W mesafesi","Nash dengesi ✅\nYavaş ⚠️"),
    ("WGAN-GP",             "Grad penalty","En kararlı ✅\nYavaş ⚠️"),
    ("Spectral Norm GAN",   "Lip. kısıtı","Hızlı ✅  Kararlı ✅"),
    ("Progressive GAN",     "Kademeli büy.","SOTA kalite ✅\nKarmaşık ⚠️"),
]
sutun_bas = ["Model", "Ana Katkı", "Avantaj / Dezavantaj"]
tablo = ax9.table(
    cellText=STRATEJILER,
    colLabels=sutun_bas,
    loc="center", cellLoc="left",
)
tablo.auto_set_font_size(False)
tablo.set_fontsize(9)
tablo.scale(1.3, 1.92)
STRATEJI_RENKLER = ["#FEE2E2","#ECFDF5","#F0FDF4","#D1FAE5","#A7F3D0","#6EE7B7"]
for (row, col), cell in tablo.get_celld().items():
    if row == 0:
        cell.set_facecolor("#064E3B")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor(STRATEJI_RENKLER[row - 1])
    cell.set_edgecolor("#D1FAE5")
ax9.set_title("GAN Ailesi & Önleme Stratejileri", fontsize=12,
              fontweight="bold", y=0.90)

# Ana başlık
fig.suptitle(
    "HAFTA 4 PAZAR — UYGULAMA 04\n"
    "GAN Değerlendirme: FID · IS · Mode Collapse · PCA · Hiperparametre Ablasyonu",
    fontsize=15, fontweight="bold", color="#064E3B", y=0.98
)

plt.savefig("h4p_04_gan_degerlendirme.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("    ✅ h4p_04_gan_degerlendirme.png kaydedildi")
plt.close()

# ─────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BÖLÜM ÖZETLERİ")
print(f"  Karşılaştırılan model      : {list(MODELLER_FID.keys())}")
print(f"  Hiperparametre ablasyonu   : lr × 4, β₁ × 4, z_dim × 4")
print(f"  Mode collapse senaryosu    : 3 (Sağlıklı / Kısmi / Tam)")
print(f"  PCA görselleştirme         : 3 model (VAE, DCGAN, WGAN-GP)")
print()
en_iyi_fid_model = min(MODELLER_FID, key=lambda m: fid_tarih[m][-1])
en_iyi_is_model  = max(MODELLER_FID, key=lambda m: IS_tarih[m][-1])
print(f"  En iyi FID@50              : {en_iyi_fid_model}"
      f"  ({fid_tarih[en_iyi_fid_model][-1]:.1f})")
print(f"  En iyi IS@50               : {en_iyi_is_model}"
      f"  ({IS_tarih[en_iyi_is_model][-1]:.3f})")
print(f"  Önerilen strateji          : WGAN-GP veya Spectral Norm")
print(f"  Grafik çıktısı             : h4p_04_gan_degerlendirme.png")
print("  ✅ UYGULAMA 04 TAMAMLANDI")
print("=" * 65)
