"""
=============================================================================
SEGMENTASYON & NESNE TESPİTİ — UYGULAMA 03
Segmentasyon Mimarileri — U-Net · Semantik · Instance
=============================================================================
Kapsam:
  - U-Net encoder-decoder mimarisi sıfırdan (NumPy)
  - Konvolüsyon, max-pooling, upsampling, skip connection simülasyonu
  - Semantik segmentasyon: piksel etiketleme + sınıf maskeleri
  - Instance segmentasyon: her nesneye ayrı renk/ID
  - Panoptic kalite metrikleri: SQ, RQ, PQ
  - DeepLab atrous convolution receptive field analizi
  - Dice Loss, Cross-Entropy, Focal Loss karşılaştırması
  - 8-panel görselleştirme

Kurulum: pip install numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.ndimage import label as scipy_label
import warnings; warnings.filterwarnings("ignore")

print("=" * 65)
print("  SEGMENTASYON — UYGULAMA 03")
print("  U-Net · Semantik · Instance Segmentasyon")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 1: U-Net Mimari Simulasyonu
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 1: U-Net Mimari Yapısı")
print("─" * 65)

class UNetBlok:
    """U-Net çift konvolüsyon bloğu."""
    def __init__(self, giris_kanal, cikis_kanal):
        self.giris  = giris_kanal
        self.cikis  = cikis_kanal
        # İki 3×3 Conv + BN + ReLU
        self.params = (giris_kanal * cikis_kanal * 9 +
                       cikis_kanal * 2 +           # BN ağırlıkları
                       cikis_kanal * cikis_kanal * 9 +
                       cikis_kanal * 2)

class UNetMimarisi:
    """
    Klasik U-Net (Ronneberger et al. 2015) mimari özeti.
    Encoder (aşağı) + Bottleneck + Decoder (yukarı)
    Her aşamada skip connection.
    """
    def __init__(self, giris_kanal=1, n_sinif=2, temel_filtre=64):
        self.n_sinif = n_sinif
        self.temel   = temel_filtre

        f = temel_filtre
        # Encoder
        self.enc1 = UNetBlok(giris_kanal, f)      # 572×572 → 568×568
        self.enc2 = UNetBlok(f,     f*2)           # 284×284 → 280×280
        self.enc3 = UNetBlok(f*2,   f*4)           # 140×140 → 136×136
        self.enc4 = UNetBlok(f*4,   f*8)           # 68×68   → 64×64
        # Bottleneck
        self.bnk  = UNetBlok(f*8,   f*16)          # 32×32   → 28×28
        # Decoder (skip concat → çift kanal girişi)
        self.dec4 = UNetBlok(f*16,  f*8)
        self.dec3 = UNetBlok(f*8,   f*4)
        self.dec2 = UNetBlok(f*4,   f*2)
        self.dec1 = UNetBlok(f*2,   f)
        # 1×1 Conv çıkış
        self.cikis_params = f * n_sinif

        self.katmanlar = [
            self.enc1, self.enc2, self.enc3, self.enc4,
            self.bnk,
            self.dec4, self.dec3, self.dec2, self.dec1
        ]

    def toplam_parametre(self):
        return sum(k.params for k in self.katmanlar) + self.cikis_params

    def ozet(self):
        print(f"\n  {'Katman':<15} {'Giriş Kanal':>14} {'Çıkış Kanal':>14} "
              f"{'Parametre':>14}")
        print("  " + "─" * 62)
        isimler = ["Enc1 (Conv2x)", "Enc2 (Conv2x)", "Enc3 (Conv2x)",
                   "Enc4 (Conv2x)", "Bottleneck",
                   "Dec4 (Conv2x)", "Dec3 (Conv2x)", "Dec2 (Conv2x)",
                   "Dec1 (Conv2x)"]
        for ad, k in zip(isimler, self.katmanlar):
            print(f"  {ad:<15} {k.giris:>14} {k.cikis:>14} {k.params:>14,}")
        print(f"  {'1×1 Conv Out':<15} {self.temel:>14} {self.n_sinif:>14} "
              f"{self.cikis_params:>14,}")
        print("  " + "─" * 62)
        print(f"  {'TOPLAM':>44} {self.toplam_parametre():>14,}")

unet = UNetMimarisi(giris_kanal=1, n_sinif=2, temel_filtre=64)
unet.ozet()

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 2: Sentetik Segmentasyon Verisi
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 2: Sentetik Segmentasyon Sahası")
print("─" * 65)

def sentetik_sahne(H=128, W=128, n_sinif=5, n_nesne=8, seed=42):
    """
    Gerçekçi sentetik segmentasyon sahnesi oluşturur.
    Döndürür: (rgb_goruntu, gt_maske, instance_maske)
    """
    np.random.seed(seed)
    goruntu = np.zeros((H, W, 3), dtype=np.float32)
    gt_mask = np.zeros((H, W), dtype=np.int32)     # sınıf maskesi
    inst_mask = np.zeros((H, W), dtype=np.int32)    # instance maskesi

    # Arka plan (zemin)
    goruntu[:, :] = [0.3, 0.5, 0.3]   # çimen yeşili
    gt_mask[:] = 0

    # Nesne palette
    sinif_renk = {
        1: ([0.8, 0.2, 0.2], "araba"),       # kırmızı
        2: ([0.2, 0.4, 0.9], "bina"),        # mavi
        3: ([0.9, 0.7, 0.1], "araç"),        # sarı
        4: ([0.6, 0.3, 0.8], "insan"),       # mor
    }

    inst_id = 1
    for _ in range(n_nesne):
        sinif = np.random.randint(1, n_sinif)
        if sinif not in sinif_renk:
            sinif = 1
        cy = np.random.randint(15, H-15)
        cx = np.random.randint(15, W-15)
        sh = np.random.randint(8, 28)
        sw = np.random.randint(10, 38)

        y1, y2 = max(0, cy-sh), min(H, cy+sh)
        x1, x2 = max(0, cx-sw), min(W, cx+sw)

        renk_arr = sinif_renk[sinif][0]
        goruntu[y1:y2, x1:x2] = renk_arr
        gt_mask[y1:y2, x1:x2] = sinif
        inst_mask[y1:y2, x1:x2] = inst_id
        inst_id += 1

    # Hafif gürültü ekle
    goruntu += np.random.normal(0, 0.05, goruntu.shape)
    goruntu  = np.clip(goruntu, 0, 1)

    return goruntu, gt_mask, inst_mask, sinif_renk

goruntu, gt_mask, inst_mask, sinif_renk = sentetik_sahne(H=128, W=128, n_sinif=5, n_nesne=10)

# Simüle tahmin maskesi (gerçeğe yakın ama biraz gürültülü)
np.random.seed(7)
pred_mask = gt_mask.copy()
bozulma_indeksleri = np.random.choice(128*128, size=int(0.08 * 128 * 128), replace=False)
pred_mask.flat[bozulma_indeksleri] = np.random.randint(0, 5, len(bozulma_indeksleri))

print(f"  Görüntü boyutu  : {goruntu.shape}")
print(f"  Piksel sayısı   : {128*128}")
print(f"  Sınıf dağılımı:")
for s in range(5):
    sayı = np.sum(gt_mask == s)
    ad = "arka plan" if s == 0 else sinif_renk.get(s, [None, f"sinif_{s}"])[1]
    print(f"    Sınıf {s} ({ad:<12}) : {sayı:>6} piksel  ({sayı/(128*128)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 3: Segmentasyon Metrikleri
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 3: Segmentasyon Metrikleri — Dice, IoU, Accuracy")
print("─" * 65)

def dice_katsayisi(pred, gt, sinif):
    """Dice = 2|A∩B| / (|A|+|B|)"""
    p = (pred == sinif).astype(float)
    g = (gt   == sinif).astype(float)
    kesisim = (p * g).sum()
    return 2 * kesisim / (p.sum() + g.sum() + 1e-9)

def pixel_iou(pred, gt, sinif):
    """Pixel-level IoU for one class."""
    p = (pred == sinif)
    g = (gt   == sinif)
    return (p & g).sum() / ((p | g).sum() + 1e-9)

def genel_accuracy(pred, gt):
    return (pred == gt).sum() / pred.size

def ortalama_iou(pred, gt, n_sinif):
    """mIoU — tüm sınıfların IoU ortalaması."""
    iou_list = []
    for s in range(n_sinif):
        if (gt == s).sum() > 0:
            iou_list.append(pixel_iou(pred, gt, s))
    return np.mean(iou_list)

n_sinif_say = 5
print(f"  {'Sınıf':<14} {'Dice':>10} {'IoU':>10}")
print("  " + "─" * 38)
for s in range(n_sinif_say):
    if (gt_mask == s).sum() == 0:
        continue
    dice = dice_katsayisi(pred_mask, gt_mask, s)
    iou  = pixel_iou(pred_mask, gt_mask, s)
    ad   = "arka plan" if s == 0 else sinif_renk.get(s,[None,f"sinif_{s}"])[1]
    print(f"  {ad:<14} {dice:>10.4f} {iou:>10.4f}")

acc   = genel_accuracy(pred_mask, gt_mask)
miou  = ortalama_iou(pred_mask, gt_mask, n_sinif_say)
print(f"\n  Genel Pixel Accuracy : {acc:.4f}")
print(f"  mIoU                 : {miou:.4f}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 4: Panoptic Quality
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 4: Panoptic Quality — SQ · RQ · PQ")
print("─" * 65)

def panoptic_quality_hesapla(pred_inst, gt_inst, iou_esik=0.5):
    """
    PQ = SQ × RQ
    SQ = Ortalama IoU (eşleşen çiftler)
    RQ = TP / (TP + 0.5*FP + 0.5*FN)
    """
    gt_idler  = set(np.unique(gt_inst)) - {0}
    pred_idler = set(np.unique(pred_inst)) - {0}

    tp, fp, fn = 0, 0, 0
    iou_toplam = 0.0

    eslesmis_gt = set()

    for pred_id in pred_idler:
        pred_m = (pred_inst == pred_id)
        en_iyi_iou, en_iyi_gt = 0, None

        for gt_id in gt_idler:
            if gt_id in eslesmis_gt:
                continue
            gt_m  = (gt_inst == gt_id)
            inter = (pred_m & gt_m).sum()
            union = (pred_m | gt_m).sum()
            iou_d = inter / (union + 1e-9)
            if iou_d > en_iyi_iou:
                en_iyi_iou, en_iyi_gt = iou_d, gt_id

        if en_iyi_gt is not None and en_iyi_iou >= iou_esik:
            tp += 1
            iou_toplam += en_iyi_iou
            eslesmis_gt.add(en_iyi_gt)
        else:
            fp += 1

    fn = len(gt_idler) - len(eslesmis_gt)

    sq = iou_toplam / (tp + 1e-9)
    rq = tp / (tp + 0.5*fp + 0.5*fn + 1e-9)
    pq = sq * rq

    return {"SQ": sq, "RQ": rq, "PQ": pq, "TP": tp, "FP": fp, "FN": fn}

pq_sonuc = panoptic_quality_hesapla(inst_mask, inst_mask)  # GT vs GT = ideal
pq_sonuc_pred = panoptic_quality_hesapla(
    # Biraz gürültülü instance maskesi
    inst_mask + (np.random.random(inst_mask.shape) > 0.85).astype(int),
    inst_mask
)

print(f"  {'Metrik':<8} {'İdeal (GT==GT)':>16} {'Tahmin vs GT':>16}")
print("  " + "─" * 44)
for k in ["SQ", "RQ", "PQ"]:
    print(f"  {k:<8} {pq_sonuc[k]:>16.4f} {pq_sonuc_pred[k]:>16.4f}")
print(f"\n  TP={pq_sonuc_pred['TP']}, FP={pq_sonuc_pred['FP']}, FN={pq_sonuc_pred['FN']}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 5: Kayıp Fonksiyonları Karşılaştırması
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 5: Segmentasyon Kayıp Fonksiyonları")
print("─" * 65)

def cross_entropy_loss(pred_prob, gt_label):
    """Piksel bazlı Cross-Entropy."""
    eps = 1e-7
    pred_prob = np.clip(pred_prob, eps, 1-eps)
    return -np.mean(gt_label * np.log(pred_prob) +
                    (1-gt_label) * np.log(1-pred_prob))

def dice_loss(pred_prob, gt_label, smooth=1.0):
    """Dice Loss = 1 - Dice Coefficient."""
    inter = (pred_prob * gt_label).sum()
    return 1 - (2 * inter + smooth) / (pred_prob.sum() + gt_label.sum() + smooth)

def focal_loss(pred_prob, gt_label, gamma=2.0, alpha=0.25):
    """Focal Loss — Zor örneklere odaklanır (Lin et al. 2017)."""
    eps = 1e-7
    pred_prob = np.clip(pred_prob, eps, 1-eps)
    ce  = -(gt_label * np.log(pred_prob) + (1-gt_label) * np.log(1-pred_prob))
    pt  = np.where(gt_label == 1, pred_prob, 1-pred_prob)
    return np.mean(alpha * ((1-pt)**gamma) * ce)

# Farklı tahmin kaliteleri için kayıp değerleri
tahmin_oranlari = np.linspace(0.01, 0.99, 50)
gt_binary = (gt_mask > 0).astype(float).flatten()
ce_kayiplar, dice_kayiplar, focal_kayiplar = [], [], []

for p in tahmin_oranlari:
    pred_probs = np.full_like(gt_binary, p)
    ce_kayiplar.append(cross_entropy_loss(pred_probs, gt_binary))
    dice_kayiplar.append(dice_loss(pred_probs, gt_binary))
    focal_kayiplar.append(focal_loss(pred_probs, gt_binary))

# En iyi tahmin noktaları
gt_oran = gt_binary.mean()
print(f"  GT pozitif oran: {gt_oran:.3f}  "
      f"(sınıfsız tahmin → CE={cross_entropy_loss(np.full_like(gt_binary, gt_oran), gt_binary):.4f})")
print(f"\n  CE kaybı p=0.1 : {cross_entropy_loss(np.full_like(gt_binary, 0.1), gt_binary):.4f}")
print(f"  CE kaybı p=0.5 : {cross_entropy_loss(np.full_like(gt_binary, 0.5), gt_binary):.4f}")
print(f"  CE kaybı p=0.9 : {cross_entropy_loss(np.full_like(gt_binary, 0.9), gt_binary):.4f}")
print(f"\n  Dice kaybı p=0.9: {dice_loss(np.full_like(gt_binary,0.9),gt_binary):.4f}  (sınıf dengesizliğe dayanıklı)")
print(f"  Focal kaybı (γ=2, p=0.1): {focal_loss(np.full_like(gt_binary,0.1),gt_binary):.4f}  (zor örneklere ağırlık)")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 6: GÖRSELLEŞTİRME (8 panel)
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 6: Görselleştirme (8 panel)")
print("─" * 65)

plt.style.use("seaborn-v0_8-dark")
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#0D1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.44, wspace=0.36,
                        top=0.93, bottom=0.05)

SINIF_RENK_VIZ = {0:"#1F2937", 1:"#EF4444", 2:"#3B82F6", 3:"#F59E0B", 4:"#A78BFA"}

# G1 — Orijinal görüntü
ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#161B22")
ax1.imshow(goruntu, aspect="auto")
ax1.set_title("Sentetik Sahne\n(Giriş Görüntüsü)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax1.axis("off")

# G2 — GT semantik maske
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#161B22")
renk_map = mcolors.LinearSegmentedColormap.from_list(
    "seg", [SINIF_RENK_VIZ[i] for i in range(n_sinif_say)], N=n_sinif_say)
ax2.imshow(gt_mask, cmap=renk_map, vmin=0, vmax=n_sinif_say-1, aspect="auto")
ax2.set_title("GT Semantik Maske\n(Gerçek)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax2.axis("off")
handles2 = [mpatches.Patch(facecolor=SINIF_RENK_VIZ[i],
    label="arka plan" if i==0 else sinif_renk.get(i,[None,f"s{i}"])[1])
    for i in range(n_sinif_say)]
ax2.legend(handles=handles2, loc="lower right", fontsize=8,
           labelcolor="#C9D1D9", facecolor="#161B22", framealpha=0.8)

# G3 — Tahmin maskesi
ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor("#161B22")
ax3.imshow(pred_mask, cmap=renk_map, vmin=0, vmax=n_sinif_say-1, aspect="auto")
ax3.set_title("Tahmin Maskesi\n(Simüle Çıktı)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax3.axis("off")

# G4 — U-Net mimari şeması
ax4 = fig.add_subplot(gs[1, :2]); ax4.set_facecolor("#161B22")
ax4.set_xlim(0, 10); ax4.set_ylim(0, 4); ax4.axis("off")
ax4.set_title("U-Net Encoder-Decoder Mimarisi", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
katmanlar_unet = [
    (0.5, 2.4, 0.7, 1.0, "#3B82F6", "1\n64"),
    (1.3, 2.4, 0.7, 0.9, "#3B82F6", "2\n128"),
    (2.1, 2.4, 0.7, 0.8, "#3B82F6", "3\n256"),
    (2.9, 2.4, 0.7, 0.7, "#3B82F6", "4\n512"),
    (4.3, 2.4, 0.7, 0.6, "#A78BFA", "BN\n1024"),  # bottleneck
    (5.7, 2.4, 0.7, 0.7, "#10B981", "4\n512"),
    (6.5, 2.4, 0.7, 0.8, "#10B981", "3\n256"),
    (7.3, 2.4, 0.7, 0.9, "#10B981", "2\n128"),
    (8.1, 2.4, 0.7, 1.0, "#10B981", "1\n64"),
]
for (x, y, w, h, renk, etiket) in katmanlar_unet:
    ax4.add_patch(mpatches.Rectangle((x, y-h/2), w, h,
        facecolor=renk, edgecolor="#0D1117", linewidth=2, alpha=0.8))
    ax4.text(x+w/2, y, etiket, ha="center", va="center",
             fontsize=7.5, color="white", fontweight="bold")

# Skip bağlantıları (yay)
skip_ciftleri = [(0, 8), (1, 7), (2, 6), (3, 5)]
enc_xs = [k[0]+k[2]/2 for k in katmanlar_unet[:4]]
dec_xs = [k[0]+k[2]/2 for k in katmanlar_unet[5:]]
for i, j in skip_ciftleri:
    x1_s = enc_xs[i]; x2_s = dec_xs[j - 5]
    y_top = katmanlar_unet[i][1] + katmanlar_unet[i][3]/2 + 0.15
    ax4.annotate("", xy=(x2_s, y_top-0.15), xytext=(x1_s, y_top-0.15),
                 arrowprops=dict(arrowstyle="->", color="#F59E0B",
                                  lw=1.5, connectionstyle="arc3,rad=-0.3"))

# MaxPool ve Upsampling okları
for i in range(3):
    ax4.annotate("", xy=(katmanlar_unet[i+1][0]+0.01, 2.4),
                 xytext=(katmanlar_unet[i][0]+katmanlar_unet[i][2], 2.4),
                 arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.5))
ax4.annotate("", xy=(katmanlar_unet[4][0]+0.01, 2.4),
             xytext=(katmanlar_unet[3][0]+0.7, 2.4),
             arrowprops=dict(arrowstyle="-|>", color="#A78BFA", lw=2.0))
for i in range(5, 8):
    ax4.annotate("", xy=(katmanlar_unet[i+1][0]+0.01, 2.4),
                 xytext=(katmanlar_unet[i][0]+0.7, 2.4),
                 arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.5))
ax4.text(4.65, 0.6, "MaxPool↓", fontsize=9, color="#64748B", ha="center")
ax4.text(6.0,  0.6, "UpSample↑", fontsize=9, color="#10B981", ha="center")
ax4.legend(handles=[
    mpatches.Patch(facecolor="#3B82F6", alpha=0.8, label="Encoder"),
    mpatches.Patch(facecolor="#A78BFA", alpha=0.8, label="Bottleneck"),
    mpatches.Patch(facecolor="#10B981", alpha=0.8, label="Decoder"),
    mpatches.Patch(facecolor="#F59E0B", alpha=0.8, label="Skip Conn."),
], loc="lower right", fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# G5 — Instance maske
ax5 = fig.add_subplot(gs[1, 2]); ax5.set_facecolor("#161B22")
n_inst = inst_mask.max()
inst_cmap = plt.cm.get_cmap("tab20", n_inst + 1)
ax5.imshow(inst_mask, cmap=inst_cmap, aspect="auto")
ax5.set_title(f"Instance Maskesi\n({n_inst} ayrı nesne)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax5.axis("off")

# G6 — Kayıp fonksiyonu karşılaştırması
ax6 = fig.add_subplot(gs[2, 0]); ax6.set_facecolor("#161B22")
for sp in ax6.spines.values(): sp.set_color("#30363D")
ax6.plot(tahmin_oranlari, ce_kayiplar, "-", color="#3B82F6",
         linewidth=2.5, label="Cross-Entropy")
ax6.plot(tahmin_oranlari, dice_kayiplar, "-", color="#10B981",
         linewidth=2.5, label="Dice Loss")
ax6.plot(tahmin_oranlari, focal_kayiplar, "-", color="#F59E0B",
         linewidth=2.5, label="Focal Loss (γ=2)")
ax6.axvline(gt_oran, color="#EF4444", linestyle="--",
            linewidth=1.5, label=f"GT oran={gt_oran:.2f}")
ax6.set_xlabel("Tahmin Olasılığı (p)", fontsize=10, color="#8B949E")
ax6.set_ylabel("Kayıp Değeri", fontsize=10, color="#8B949E")
ax6.set_title("Segmentasyon Kayıp\nFonksiyonları", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax6.tick_params(colors="#8B949E")
ax6.grid(alpha=0.3, color="#30363D")
ax6.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# G7 — Dice/IoU sınıf bazlı bar
ax7 = fig.add_subplot(gs[2, 1]); ax7.set_facecolor("#161B22")
for sp in ax7.spines.values(): sp.set_color("#30363D")
aktif_siniflar = [s for s in range(n_sinif_say) if (gt_mask == s).sum() > 0]
adlar7 = ["arka plan" if s==0 else sinif_renk.get(s,[None,f"s{s}"])[1]
          for s in aktif_siniflar]
dice_vals7 = [dice_katsayisi(pred_mask, gt_mask, s) for s in aktif_siniflar]
iou_vals7  = [pixel_iou(pred_mask, gt_mask, s) for s in aktif_siniflar]
x7 = np.arange(len(aktif_siniflar)); w7 = 0.35
ax7.bar(x7-w7/2, dice_vals7, w7, label="Dice", color="#0FBCCE",
        edgecolor="#30363D", alpha=0.88)
ax7.bar(x7+w7/2, iou_vals7,  w7, label="IoU",  color="#A78BFA",
        edgecolor="#30363D", alpha=0.88)
ax7.axhline(np.mean(iou_vals7), color="#F59E0B", linestyle="--",
            linewidth=1.5, label=f"mIoU={np.mean(iou_vals7):.2f}")
ax7.set_xticks(x7); ax7.set_xticklabels(adlar7, color="#C9D1D9",
                                           fontsize=9, rotation=20, ha="right")
ax7.set_ylim(0, 1.05)
ax7.set_title("Sınıf Bazlı Dice / IoU", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax7.set_ylabel("Skor", fontsize=10, color="#8B949E")
ax7.tick_params(colors="#8B949E")
ax7.grid(axis="y", alpha=0.3, color="#30363D")
ax7.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# G8 — Hata haritası (GT - Pred)
ax8 = fig.add_subplot(gs[2, 2]); ax8.set_facecolor("#161B22")
hata_map = (pred_mask != gt_mask).astype(float)
ax8.imshow(hata_map, cmap="hot", aspect="auto")
ax8.set_title(f"Segmentasyon Hata Haritası\n"
              f"(Hata oranı: {hata_map.mean()*100:.1f}%)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax8.axis("off")

fig.suptitle(
    "SEGMENTASYON — UYGULAMA 03  |  U-Net · Semantik · Instance\n"
    "Mimari · Metrikler · Kayıp Fonksiyonları · Panoptic Quality",
    fontsize=14, fontweight="bold", color="#C9D1D9", y=0.98)
plt.savefig("cv_03_segmentasyon.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("  ✅ cv_03_segmentasyon.png kaydedildi")
plt.close()

print("\n" + "=" * 65)
print(f"  U-Net parametresi : {unet.toplam_parametre():,}")
print(f"  mIoU              : {miou:.4f}")
print(f"  Pixel Accuracy    : {acc:.4f}")
print(f"  PQ                : {pq_sonuc_pred['PQ']:.4f}")
print("  ✅ UYGULAMA 03 TAMAMLANDI")
print("=" * 65)
