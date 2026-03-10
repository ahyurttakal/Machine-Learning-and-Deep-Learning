"""
=============================================================================
SEGMENTASYON & NESNE TESPİTİ — UYGULAMA 04
Veri Artırma · Eğitim Pipeline · Fine-tuning & Transfer Learning
=============================================================================
Kapsam:
  - 12 temel augmentation: flip, rotate, crop, blur, brightness,
    contrast, noise, cutout, mosaic, mixup, elastic, color jitter
  - Augmentation pipeline zinciri ve olasılık kontrolü
  - Eğitim döngüsü: mini-batch, optimizer, lr scheduler simülasyonu
  - YOLO fine-tuning stratejileri: frozen backbone, layer-wise LR
  - Learning rate schedule: warmup + cosine annealing
  - mAP eğitim geçmişi: train/val karşılaştırması
  - Confusion matrix analizi
  - 8-panel görselleştirme

Kurulum: pip install numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings("ignore")

print("=" * 65)
print("  NESNE TESPİTİ — UYGULAMA 04")
print("  Veri Artırma · Eğitim · Fine-tuning Pipeline")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 1: Augmentation Pipeline
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 1: Augmentation Pipeline")
print("─" * 65)

class Augmentation:
    """Tek bir veri artırma işlemi."""
    def __init__(self, ad, prob, parametre=None, kategori="geometrik"):
        self.ad         = ad
        self.prob       = prob       # Uygulanma olasılığı
        self.parametre  = parametre or {}
        self.kategori   = kategori
        self.uygulandi  = 0
        self.atildi     = 0

    def uygula(self, goruntu, kutular):
        """
        Görüntüye ve bounding box'lara augmentation uygular.
        Burada simüle ediyoruz — gerçekte piksel manipülasyonu yapılır.
        """
        if np.random.random() > self.prob:
            self.atildi += 1
            return goruntu, kutular, False

        self.uygulandi += 1
        goruntu_aug = goruntu.copy()
        kutular_aug = kutular.copy() if kutular is not None else None

        # Simüle transformasyon etkileri
        if self.ad == "flip_horizontal":
            goruntu_aug = goruntu[:, ::-1, :]
            if kutular_aug is not None:
                W = goruntu.shape[1]
                kutular_aug[:, 0] = W - kutular_aug[:, 2]
                kutular_aug[:, 2] = W - kutular_aug[:, 0]

        elif self.ad == "rotate":
            aci = np.random.uniform(*self.parametre.get("limit", [-15, 15]))
            # Piksel kaydırma simülasyonu
            goruntu_aug = np.roll(goruntu, int(aci), axis=1)

        elif self.ad == "brightness":
            delta = np.random.uniform(*self.parametre.get("limit", [-0.2, 0.2]))
            goruntu_aug = np.clip(goruntu + delta, 0, 1)

        elif self.ad == "gaussian_noise":
            std = self.parametre.get("std", 0.03)
            goruntu_aug = np.clip(goruntu + np.random.normal(0, std, goruntu.shape), 0, 1)

        elif self.ad == "cutout":
            H, W = goruntu.shape[:2]
            ch, cw = int(H*0.2), int(W*0.2)
            y1 = np.random.randint(0, H-ch); x1 = np.random.randint(0, W-cw)
            goruntu_aug[y1:y1+ch, x1:x1+cw] = 0.5

        return goruntu_aug, kutular_aug, True


class AugmentationPipeline:
    """
    Sıralı augmentation zinciri.
    Albumentations tarzı Compose yapısı.
    """
    def __init__(self, augmentations: list, bbox_format="xyxy"):
        self.augmentations = augmentations
        self.bbox_format   = bbox_format
        self.toplam_calisma = 0

    def __call__(self, goruntu, kutular=None):
        self.toplam_calisma += 1
        for aug in self.augmentations:
            goruntu, kutular, _ = aug.uygula(goruntu, kutular)
        return goruntu, kutular

    def ozet(self):
        print(f"\n  {'Augmentation':<24} {'Prob':>6} {'Uyg.':>7} {'Atlanan':>8} {'Kat.'}")
        print("  " + "─" * 58)
        for aug in self.augmentations:
            toplam = aug.uygulandi + aug.atildi
            oran   = aug.uygulandi / toplam * 100 if toplam > 0 else 0
            print(f"  {aug.ad:<24} {aug.prob:>6.2f} {aug.uygulandi:>7} "
                  f"({oran:4.0f}%) {aug.atildi:>6}  {aug.kategori}")


# Pipeline tanımı
pipeline_egitim = AugmentationPipeline([
    Augmentation("flip_horizontal",  0.50, kategori="geometrik"),
    Augmentation("rotate",           0.30, {"limit": [-15, 15]},   "geometrik"),
    Augmentation("random_crop",      0.40, {"scale": [0.7, 1.0]},  "geometrik"),
    Augmentation("brightness",       0.40, {"limit": [-0.3, 0.3]}, "fotometrik"),
    Augmentation("contrast",         0.40, {"limit": [0.7, 1.3]},  "fotometrik"),
    Augmentation("hue_saturation",   0.30, {"hue": 20},             "fotometrik"),
    Augmentation("gaussian_blur",    0.20, {"blur_limit": [3, 7]},  "gürültü"),
    Augmentation("gaussian_noise",   0.30, {"std": 0.03},           "gürültü"),
    Augmentation("cutout",           0.25, {"n_holes": 3},          "düzenleme"),
    Augmentation("mosaic",           0.15, {"n": 4},                "düzenleme"),
    Augmentation("mixup",            0.10, {"alpha": 0.2},          "düzenleme"),
    Augmentation("elastic",          0.10, {"sigma": 50},           "geometrik"),
])

# Simüle sentetik görüntü
np.random.seed(42)
H, W = 64, 64
test_goruntu = np.random.rand(H, W, 3).astype(np.float32)
test_kutular = np.array([[10, 15, 40, 45], [30, 5, 55, 30]], dtype=float)

# 200 görüntü simülasyonu
for _ in range(200):
    pipeline_egitim(test_goruntu, test_kutular)

pipeline_egitim.ozet()

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 2: Mosaic & MixUp Augmentation
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 2: Mosaic & MixUp Augmentation")
print("─" * 65)

def mosaic_augmentation(goruntular, kutular_listesi, hedef_boyut=(640, 640)):
    """
    YOLOv4'te tanıtılan Mosaic augmentation.
    4 görüntüyü 2×2 grid'e birleştirir.
    """
    H, W = hedef_boyut
    yc   = int(np.random.uniform(H*0.35, H*0.65))
    xc   = int(np.random.uniform(W*0.35, W*0.65))
    mozaik = np.zeros((H, W, 3), dtype=np.float32)
    tum_kutular = []

    pozisyonlar = [
        (0, 0, xc, yc, "sol-üst"),
        (xc, 0, W, yc, "sağ-üst"),
        (0, yc, xc, H, "sol-alt"),
        (xc, yc, W, H, "sağ-alt"),
    ]

    for idx, (x1, y1, x2, y2, pos) in enumerate(pozisyonlar):
        if idx >= len(goruntular):
            break
        g = goruntular[idx]
        bh = y2 - y1; bw = x2 - x1
        g_resized = g[:bh, :bw] if g.shape[0] >= bh and g.shape[1] >= bw \
                    else np.pad(g, ((0,max(0,bh-g.shape[0])),
                                   (0,max(0,bw-g.shape[1])),
                                   (0,0)), mode="edge")[:bh,:bw]
        mozaik[y1:y2, x1:x2] = g_resized[:bh,:bw]

        # Kutular offset'le
        if idx < len(kutular_listesi) and kutular_listesi[idx] is not None:
            offs_kutular = kutular_listesi[idx].copy()
            offs_kutular[:, [0, 2]] += x1
            offs_kutular[:, [1, 3]] += y1
            tum_kutular.append(offs_kutular)

    return mozaik, np.vstack(tum_kutular) if tum_kutular else np.array([]), xc, yc

def mixup_augmentation(g1, g2, k1, k2, alpha=0.2):
    """
    MixUp: İki görüntüyü lambda ile ağırlıklı karıştırır.
    """
    lam = np.random.beta(alpha, alpha)
    karışık = lam * g1 + (1 - lam) * g2
    tum_k = np.vstack([k1, k2]) if k1 is not None and k2 is not None else k1
    return karışık, tum_k, lam

# Test
np.random.seed(5)
test_goruntular = [np.random.rand(320, 320, 3).astype(np.float32) * 0.4 +
                   np.random.rand(3) * 0.6 for _ in range(4)]
test_kutular_4 = [np.array([[50,50,150,150],[100,80,200,180]], dtype=float)
                  for _ in range(4)]

mozaik_g, mozaik_k, xc, yc = mosaic_augmentation(test_goruntular, test_kutular_4)
mixup_g, mixup_k, lam = mixup_augmentation(
    test_goruntular[0], test_goruntular[1],
    test_kutular_4[0], test_kutular_4[1])

print(f"  Mosaic çıktı: {mozaik_g.shape}, birleşim noktası: ({xc},{yc})")
print(f"  Mosaic kutu sayısı: {len(mozaik_k)}")
print(f"  MixUp lambda: {lam:.3f}  "
      f"({lam*100:.1f}% görüntü1 + {(1-lam)*100:.1f}% görüntü2)")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 3: Eğitim Pipeline & LR Schedule
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 3: Eğitim Pipeline & Learning Rate Schedule")
print("─" * 65)

class CosineLRScheduler:
    """Warmup + Cosine Annealing LR scheduler."""
    def __init__(self, lr_baslangic=0.01, lr_min=0.0001,
                 toplam_epoch=100, warmup_epoch=5):
        self.lr0     = lr_baslangic
        self.lr_min  = lr_min
        self.total   = toplam_epoch
        self.warmup  = warmup_epoch

    def lr_hesapla(self, epoch):
        if epoch < self.warmup:
            return self.lr0 * (epoch + 1) / self.warmup
        progress = (epoch - self.warmup) / (self.total - self.warmup)
        return self.lr_min + 0.5 * (self.lr0 - self.lr_min) * \
               (1 + np.cos(np.pi * progress))

scheduler = CosineLRScheduler(lr_baslangic=0.01, lr_min=1e-5,
                               toplam_epoch=100, warmup_epoch=3)
lr_gecmis = [scheduler.lr_hesapla(e) for e in range(100)]

# Eğitim simülasyonu
np.random.seed(11)
EPOCH = 100
train_loss, val_loss = [], []
train_map,  val_map  = [], []

for e in range(EPOCH):
    prog = e / EPOCH
    lr   = lr_gecmis[e]
    lr_katkisi = 1 - np.exp(-lr * 100)

    # Loss
    tl = 2.8 * np.exp(-4.5 * prog * lr_katkisi) + 0.18 + np.random.normal(0, 0.04)
    vl = 2.9 * np.exp(-4.2 * prog * lr_katkisi) + 0.22 + np.random.normal(0, 0.05)
    train_loss.append(max(0, tl))
    val_loss.append(max(0, vl))

    # mAP
    tm = 0.55 * (1 - np.exp(-5.5 * prog)) + np.random.normal(0, 0.012)
    vm = 0.50 * (1 - np.exp(-5.0 * prog)) + np.random.normal(0, 0.014)
    train_map.append(np.clip(tm, 0, 1))
    val_map.append(np.clip(vm, 0, 1))

en_iyi_val_map = max(val_map)
en_iyi_epoch   = val_map.index(en_iyi_val_map) + 1
print(f"  Toplam epoch       : {EPOCH}")
print(f"  En iyi val mAP     : {en_iyi_val_map:.4f}  (epoch {en_iyi_epoch})")
print(f"  Son train loss     : {train_loss[-1]:.4f}")
print(f"  Son val loss       : {val_loss[-1]:.4f}")
print(f"  LR max             : {max(lr_gecmis):.6f}")
print(f"  LR min (son)       : {lr_gecmis[-1]:.8f}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 4: Fine-tuning Stratejileri
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 4: YOLO Fine-tuning Stratejileri")
print("─" * 65)

KATMAN_GRUPLARI = {
    "Backbone Erken (frozen)": {"donduruldu": True, "lr_carp": 0.0, "epoch_baslangic": 0},
    "Backbone Orta":           {"donduruldu": True, "lr_carp": 0.0, "epoch_baslangic": 0},
    "Backbone Derin":          {"donduruldu": False, "lr_carp": 0.1, "epoch_baslangic": 10},
    "Neck (FPN/PAN)":          {"donduruldu": False, "lr_carp": 0.5, "epoch_baslangic": 5},
    "Head (Detection)":        {"donduruldu": False, "lr_carp": 1.0, "epoch_baslangic": 0},
}

print(f"\n  {'Katman Grubu':<28} {'Donduruldu':>12} {'LR Çarpanı':>12} {'Başlangıç Epoch':>16}")
print("  " + "─" * 72)
for katman, kfg in KATMAN_GRUPLARI.items():
    don = "✓ Frozen" if kfg["donduruldu"] else "✗ Aktif"
    print(f"  {katman:<28} {don:>12} {kfg['lr_carp']:>12.1f} {kfg['epoch_baslangic']:>16}")

print(f"\n  Strateji:")
print(f"  Epoch 0-5   : Sadece Head eğitilir (backbone frozen)")
print(f"  Epoch 5-10  : Neck + Head aktif edilir")
print(f"  Epoch 10+   : Tüm katmanlar aktif (layer-wise LR)")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 5: Confusion Matrix
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 5: Confusion Matrix Analizi")
print("─" * 65)

SINIFLAR = ["araba", "insan", "bisiklet", "kamyon", "motosiklet", "arka plan"]
N = len(SINIFLAR)
np.random.seed(8)

# Gerçekçi confusion matrix (köşegen baskın)
cm = np.zeros((N, N), dtype=int)
for i in range(N-1):
    cm[i, i]    = np.random.randint(80, 150)  # doğru tahmin
    for j in range(N):
        if j != i:
            cm[i, j] = np.random.randint(0, 15)  # yanlış

# Arka plan: false positive sayısı yüksek
cm[N-1, N-1] = 200
for i in range(N-1):
    cm[i, N-1] = np.random.randint(5, 25)
    cm[N-1, i] = np.random.randint(3, 18)

# Normalize
cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

print(f"\n  {'Sınıf':<14} {'Precision':>12} {'Recall':>12} {'F1':>12}")
print("  " + "─" * 52)
for i, sinif in enumerate(SINIFLAR):
    tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
    prec  = tp / (tp + fp + 1e-9)
    rec   = tp / (tp + fn + 1e-9)
    f1    = 2 * prec * rec / (prec + rec + 1e-9)
    print(f"  {sinif:<14} {prec:>12.4f} {rec:>12.4f} {f1:>12.4f}")

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

AUG_RENK = {"geometrik":"#3B82F6","fotometrik":"#10B981",
             "gürültü":"#F59E0B","düzenleme":"#A78BFA"}

# G1 — Mosaic görsel
ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#161B22")
ax1.imshow(mozaik_g[:160,:160], aspect="auto")
ax1.axvline(xc//4, color="#F59E0B", linewidth=2)
ax1.axhline(yc//4, color="#F59E0B", linewidth=2)
ax1.set_title("Mosaic Augmentation\n(4 görüntü birleşimi)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax1.axis("off")

# G2 — MixUp görsel
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#161B22")
ax2.imshow(mixup_g[:100,:100], aspect="auto")
ax2.set_title(f"MixUp Augmentation\nλ={lam:.2f}", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax2.axis("off")

# G3 — Augmentation uygulama oranları
ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor("#161B22")
for sp in ax3.spines.values(): sp.set_color("#30363D")
aug_adlari    = [a.ad.replace("_","\n") for a in pipeline_egitim.augmentations]
uyg_oranlari  = [a.uygulandi / max(a.uygulandi+a.atildi,1) * 100
                 for a in pipeline_egitim.augmentations]
renkler3 = [AUG_RENK.get(a.kategori,"#64748B")
            for a in pipeline_egitim.augmentations]
y3 = np.arange(len(aug_adlari))
ax3.barh(y3, uyg_oranlari, color=renkler3, edgecolor="#30363D", alpha=0.88)
ax3.set_yticks(y3)
ax3.set_yticklabels(aug_adlari, fontsize=7.5, color="#C9D1D9")
ax3.set_xlabel("Uygulama Oranı (%)", fontsize=10, color="#8B949E")
ax3.set_title("Augmentation Uygulama\nOranları", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax3.tick_params(colors="#8B949E")
ax3.grid(axis="x", alpha=0.3, color="#30363D")
handles3 = [mpatches.Patch(facecolor=v, label=k) for k,v in AUG_RENK.items()]
ax3.legend(handles=handles3, fontsize=8, labelcolor="#C9D1D9",
           facecolor="#161B22", loc="lower right")

# G4 — Eğitim kayıp eğrileri
ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor("#161B22")
for sp in ax4.spines.values(): sp.set_color("#30363D")
epok_listesi = range(EPOCH)
ax4.plot(epok_listesi, train_loss, "-", color="#3B82F6", linewidth=2,
         label="Train Loss", alpha=0.9)
ax4.plot(epok_listesi, val_loss,   "-", color="#F59E0B", linewidth=2,
         label="Val Loss",   alpha=0.9)
ax4.fill_between(epok_listesi, train_loss, val_loss, alpha=0.08,
                  color="#A78BFA")
ax4.set_title("Eğitim / Val Kayıp\nEğrisi", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax4.set_xlabel("Epoch", fontsize=10, color="#8B949E")
ax4.set_ylabel("Loss", fontsize=10, color="#8B949E")
ax4.tick_params(colors="#8B949E")
ax4.grid(alpha=0.3, color="#30363D")
ax4.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# G5 — mAP eğitim geçmişi
ax5 = fig.add_subplot(gs[1, 1]); ax5.set_facecolor("#161B22")
for sp in ax5.spines.values(): sp.set_color("#30363D")
ax5.plot(epok_listesi, train_map, "-", color="#10B981", linewidth=2,
         label="Train mAP")
ax5.plot(epok_listesi, val_map,   "-", color="#0FBCCE", linewidth=2,
         label="Val mAP")
ax5.axvline(en_iyi_epoch, color="#F59E0B", linestyle="--",
            linewidth=1.8, label=f"En iyi: epoch {en_iyi_epoch}")
ax5.scatter([en_iyi_epoch], [en_iyi_val_map], s=120, c="#F59E0B", zorder=5)
ax5.set_title("mAP Eğitim Geçmişi", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax5.set_xlabel("Epoch", fontsize=10, color="#8B949E")
ax5.set_ylabel("mAP@0.5", fontsize=10, color="#8B949E")
ax5.tick_params(colors="#8B949E")
ax5.grid(alpha=0.3, color="#30363D")
ax5.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# G6 — LR schedule
ax6 = fig.add_subplot(gs[1, 2]); ax6.set_facecolor("#161B22")
for sp in ax6.spines.values(): sp.set_color("#30363D")
ax6.plot(range(EPOCH), lr_gecmis, "-", color="#A78BFA", linewidth=2.5)
ax6.fill_between(range(EPOCH), lr_gecmis, alpha=0.20, color="#A78BFA")
ax6.axvspan(0, 3, alpha=0.15, color="#F59E0B", label="Warmup (3 epoch)")
ax6.set_title("Cosine LR Scheduler\n(Warmup + Annealing)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax6.set_xlabel("Epoch", fontsize=10, color="#8B949E")
ax6.set_ylabel("Öğrenme Hızı", fontsize=10, color="#8B949E")
ax6.tick_params(colors="#8B949E")
ax6.grid(alpha=0.3, color="#30363D")
ax6.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# G7 — Confusion matrix
ax7 = fig.add_subplot(gs[2, :2]); ax7.set_facecolor("#161B22")
im7 = ax7.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
plt.colorbar(im7, ax=ax7, shrink=0.85)
ax7.set_xticks(range(N)); ax7.set_xticklabels(SINIFLAR, rotation=30,
    ha="right", fontsize=9, color="#C9D1D9")
ax7.set_yticks(range(N)); ax7.set_yticklabels(SINIFLAR, fontsize=9, color="#C9D1D9")
for i in range(N):
    for j in range(N):
        renk_text = "black" if cm_norm[i,j] > 0.5 else "white"
        ax7.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                 fontsize=9, color=renk_text)
ax7.set_title("Normalized Confusion Matrix\n(Sınıflandırma Hatası Analizi)",
              fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax7.set_xlabel("Tahmin Edilen", fontsize=10, color="#8B949E")
ax7.set_ylabel("Gerçek", fontsize=10, color="#8B949E")

# G8 — Fine-tuning layer freeze stratejisi
ax8 = fig.add_subplot(gs[2, 2]); ax8.set_facecolor("#161B22")
ax8.set_xlim(0, 10); ax8.set_ylim(-0.5, 5.5); ax8.axis("off")
ax8.set_title("Fine-tuning Katman\nDondurma Stratejisi", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
katman_adlari = list(KATMAN_GRUPLARI.keys())
for i, (katman, kfg) in enumerate(KATMAN_GRUPLARI.items()):
    y = 4.5 - i
    renk = "#64748B" if kfg["donduruldu"] else "#0FBCCE"
    alpha = 0.35 if kfg["donduruldu"] else 0.85
    ax8.add_patch(mpatches.FancyBboxPatch(
        (0.5, y-0.35), 9.0, 0.65,
        boxstyle="round,pad=0.05",
        facecolor=renk, edgecolor="#0D1117", linewidth=2, alpha=alpha))
    durum = "🔒 Frozen" if kfg["donduruldu"] else f"✓ LR×{kfg['lr_carp']}"
    ax8.text(5, y, f"{katman}   {durum}",
             ha="center", va="center", fontsize=10,
             color="white" if not kfg["donduruldu"] else "#94A3B8",
             fontweight="bold" if not kfg["donduruldu"] else "normal")

fig.suptitle(
    "NESNE TESPİTİ — UYGULAMA 04  |  Augmentation · Eğitim · Fine-tuning\n"
    "Mosaic · MixUp · LR Schedule · Confusion Matrix · Layer Freeze",
    fontsize=14, fontweight="bold", color="#C9D1D9", y=0.98)
plt.savefig("cv_04_egitim_pipeline.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("  ✅ cv_04_egitim_pipeline.png kaydedildi")
plt.close()

print("\n" + "=" * 65)
print(f"  Augmentation sayısı   : {len(pipeline_egitim.augmentations)}")
print(f"  Toplam simülasyon     : 200 görüntü")
print(f"  En iyi val mAP        : {en_iyi_val_map:.4f}  (epoch {en_iyi_epoch})")
print(f"  LR warmup             : 3 epoch")
print("  ✅ UYGULAMA 04 TAMAMLANDI")
print("=" * 65)
