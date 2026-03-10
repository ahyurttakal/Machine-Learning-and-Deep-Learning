"""
=============================================================================
SEGMENTASYON & NESNE TESPİTİ — UYGULAMA 02
YOLO Mimarisi & Tespit Pipeline — Sıfırdan İmplementasyon
=============================================================================
Kapsam:
  - YOLO grid mekanizması: S×S hücre, B kutu, C sınıf
  - Bounding box encoding/decoding (tx, ty, tw, th)
  - Anchor box tasarımı ve k-means kümeleme
  - Çok ölçekli özellik haritası (P3/P4/P5)
  - YOLOv8 kayıp fonksiyonu: CIoU + BCE + DFL
  - Tam tespit pipeline: ön işleme → çıkarım → NMS → sonuç
  - Model karşılaştırma: YOLOv5n vs YOLOv8n vs YOLOv10n
  - 8-panel görselleştirme

Kurulum: pip install numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from collections import defaultdict
import warnings; warnings.filterwarnings("ignore")

print("=" * 65)
print("  NESNE TESPİTİ — UYGULAMA 02")
print("  YOLO Mimarisi & Tespit Pipeline")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 1: YOLO Grid Mekanizması
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 1: YOLO Grid Mekanizması")
print("─" * 65)

class YOLOGrid:
    """
    YOLO v1-v3 tarzı grid bazlı tespit.
    Görüntü → S×S hücre → Her hücrede B kutu + C sınıf tahmini.
    """
    def __init__(self, S=7, B=2, C=20, goruntu_boyutu=448):
        self.S = S          # Grid boyutu
        self.B = B          # Hücre başına kutu sayısı
        self.C = C          # Sınıf sayısı
        self.img_w = goruntu_boyutu
        self.img_h = goruntu_boyutu
        self.hucre_w = goruntu_boyutu / S
        self.hucre_h = goruntu_boyutu / S

    def encode_box(self, x_merkez, y_merkez, w, h):
        """
        Piksel koordinatlarını → YOLO normalize formatına çevir.
        tx, ty: hücreye göre offset [0,1]
        tw, th: görüntüye göre normalize boyut [0,1]
        """
        hucre_col = int(x_merkez / self.hucre_w)
        hucre_row = int(y_merkez / self.hucre_h)
        hucre_col = min(hucre_col, self.S - 1)
        hucre_row = min(hucre_row, self.S - 1)

        tx = (x_merkez / self.hucre_w) - hucre_col
        ty = (y_merkez / self.hucre_h) - hucre_row
        tw = w / self.img_w
        th = h / self.img_h

        return tx, ty, tw, th, hucre_col, hucre_row

    def decode_box(self, tx, ty, tw, th, hucre_col, hucre_row):
        """YOLO formatından → piksel koordinatlarına geri çevir."""
        x_merkez = (tx + hucre_col) * self.hucre_w
        y_merkez = (ty + hucre_row) * self.hucre_h
        w = tw * self.img_w
        h = th * self.img_h
        x1 = x_merkez - w / 2
        y1 = y_merkez - h / 2
        return x1, y1, x1 + w, y1 + h

    def cikti_tensoru_boyutu(self):
        return (self.S, self.S, self.B * 5 + self.C)

grid = YOLOGrid(S=7, B=2, C=20)
print(f"  Grid boyutu   : {grid.S}×{grid.S}")
print(f"  Kutu / hücre  : {grid.B}")
print(f"  Sınıf sayısı  : {grid.C}")
print(f"  Çıktı tensörü : {grid.cikti_tensoru_boyutu()}  → {np.prod(grid.cikti_tensoru_boyutu())} değer")

# Encoding testi
test_nesneler = [
    (224, 112, 120, 80,  "araba"),
    (100, 300, 60,  90,  "insan"),
    (350, 200, 200, 140, "kamyon"),
]
print(f"\n  {'Nesne':<12} {'tx':>8} {'ty':>8} {'tw':>8} {'th':>8} {'Hücre'}")
print("  " + "─" * 62)
for cx, cy, w, h, ad in test_nesneler:
    tx, ty, tw, th, col, row = grid.encode_box(cx, cy, w, h)
    print(f"  {ad:<12} {tx:>8.4f} {ty:>8.4f} {tw:>8.4f} {th:>8.4f} ({col},{row})")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 2: Anchor Box Tasarımı (K-Means)
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 2: Anchor Box — K-Means Kümeleme")
print("─" * 65)

def iou_wh(box, merkez):
    """Yalnızca w,h bazlı IoU (konum sıfırda varsayılır)."""
    inter_w = min(box[0], merkez[0])
    inter_h = min(box[1], merkez[1])
    inter   = inter_w * inter_h
    union   = box[0]*box[1] + merkez[0]*merkez[1] - inter
    return inter / (union + 1e-9)

def kmeans_anchor(kutular, k=9, iterasyon=100):
    """
    IoU tabanlı K-Means ile optimal anchor boyutları bulur.
    YOLO'nun anchor tasarımı bu yöntemi kullanır.
    """
    n = len(kutular)
    merkezler = kutular[np.random.choice(n, k, replace=False)].copy()

    for _ in range(iterasyon):
        uzakliklar = np.zeros((n, k))
        for j, m in enumerate(merkezler):
            for i, b in enumerate(kutular):
                uzakliklar[i, j] = 1 - iou_wh(b, m)

        etiketler = np.argmin(uzakliklar, axis=1)

        yeni_merkezler = np.array([
            kutular[etiketler == j].mean(axis=0)
            if np.any(etiketler == j)
            else merkezler[j]
            for j in range(k)
        ])

        if np.allclose(merkezler, yeni_merkezler, atol=0.01):
            break
        merkezler = yeni_merkezler

    # Alana göre sırala
    merkezler = merkezler[np.argsort(merkezler[:, 0] * merkezler[:, 1])]
    return merkezler

# Simüle COCO benzeri kutu dağılımı
np.random.seed(42)
N_kutu = 500
wh_kutulari = np.column_stack([
    np.concatenate([np.random.exponential(30, 200),   # küçük nesneler
                    np.random.exponential(80, 200),    # orta nesneler
                    np.random.exponential(150, 100)]), # büyük nesneler
    np.concatenate([np.random.exponential(40, 200),
                    np.random.exponential(100, 200),
                    np.random.exponential(180, 100)]),
])
wh_kutulari = np.clip(wh_kutulari, 5, 600)

anchor_9  = kmeans_anchor(wh_kutulari, k=9)
anchor_3s = anchor_9[:3]    # Küçük nesneler (P3 80×80)
anchor_3m = anchor_9[3:6]   # Orta nesneler  (P4 40×40)
anchor_3l = anchor_9[6:]    # Büyük nesneler (P5 20×20)

print(f"  K=9 anchor (w×h piksel):")
print(f"  {'Katman':<10} {'Anchor 1':>16} {'Anchor 2':>16} {'Anchor 3':>16}")
print("  " + "─" * 62)
for katman, ankor_seti in [("P3 (küçük)", anchor_3s),
                             ("P4 (orta)",  anchor_3m),
                             ("P5 (büyük)", anchor_3l)]:
    a = [f"{int(a[0])}×{int(a[1])}" for a in ankor_seti]
    print(f"  {katman:<10} {a[0]:>16} {a[1]:>16} {a[2]:>16}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 3: Çok Ölçekli Özellik Haritası
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 3: Çok Ölçekli Özellik Haritası (FPN)")
print("─" * 65)

class FPNOzellikHaritasi:
    """YOLOv3/v8 tarzı Feature Pyramid Network simülasyonu."""

    def __init__(self, goruntu_boyutu=640, n_sinif=80):
        self.img = goruntu_boyutu
        self.C   = n_sinif
        self.katmanlar = {
            "P3": {"stride": 8,  "boyut": goruntu_boyutu // 8,
                   "nesne_boyutu": "küçük  (<32px)"},
            "P4": {"stride": 16, "boyut": goruntu_boyutu // 16,
                   "nesne_boyutu": "orta   (32-96px)"},
            "P5": {"stride": 32, "boyut": goruntu_boyutu // 32,
                   "nesne_boyutu": "büyük  (>96px)"},
        }

    def cikti_boyutlari(self, n_anchor=3):
        """Her katman için çıktı tensörü boyutu."""
        sonuclar = {}
        for ad, k in self.katmanlar.items():
            G = k["boyut"]
            sonuclar[ad] = {
                "harita": (G, G),
                "tensor": (G, G, n_anchor * (5 + self.C)),
                "toplam": G * G * n_anchor * (5 + self.C),
            }
        return sonuclar

fpn = FPNOzellikHaritasi(640, 80)
boyutlar = fpn.cikti_boyutlari(3)
toplam_tahmin = sum(v["toplam"] for v in boyutlar.values())

print(f"\n  {'Katman':<6} {'Stride':>8} {'Harita':>10} {'Nesne Boyutu':<18} {'Tensör boyutu'}")
print("  " + "─" * 70)
for (ad, k), (_, b) in zip(fpn.katmanlar.items(), boyutlar.items()):
    print(f"  {ad:<6} {k['stride']:>8}   {k['boyut']}×{k['boyut']:<8}"
          f" {k['nesne_boyutu']:<18} {b['tensor']}")
print(f"\n  Toplam tahmin sayısı (640×640): "
      f"{sum(v['harita'][0]**2 * 3 for v in boyutlar.values())} bounding box")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 4: Kayıp Fonksiyonu
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 4: YOLOv8 Kayıp Fonksiyonu")
print("─" * 65)

def ciou_loss(pred_box, gt_box):
    """
    CIoU Loss (Complete IoU) — DIoU + aspect ratio cezası.
    pred_box, gt_box: [x1, y1, x2, y2]
    """
    # IoU hesapla
    ix1 = max(pred_box[0], gt_box[0]); iy1 = max(pred_box[1], gt_box[1])
    ix2 = min(pred_box[2], gt_box[2]); iy2 = min(pred_box[3], gt_box[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    a1 = (pred_box[2]-pred_box[0]) * (pred_box[3]-pred_box[1])
    a2 = (gt_box[2]-gt_box[0])     * (gt_box[3]-gt_box[1])
    union = a1 + a2 - inter
    iou_d = inter / (union + 1e-9)

    # Merkez mesafesi
    px = (pred_box[0]+pred_box[2])/2; py = (pred_box[1]+pred_box[3])/2
    gx = (gt_box[0]+gt_box[2])/2;    gy = (gt_box[1]+gt_box[3])/2
    rho2 = (px-gx)**2 + (py-gy)**2

    # Kapsayan kutu köşegeni
    enc_x1 = min(pred_box[0],gt_box[0]); enc_y1 = min(pred_box[1],gt_box[1])
    enc_x2 = max(pred_box[2],gt_box[2]); enc_y2 = max(pred_box[3],gt_box[3])
    c2 = (enc_x2-enc_x1)**2 + (enc_y2-enc_y1)**2 + 1e-9

    # Aspect ratio cezası
    pw = pred_box[2]-pred_box[0]; ph = pred_box[3]-pred_box[1]
    gw = gt_box[2]-gt_box[0];    gh = gt_box[3]-gt_box[1]
    v = (4 / (np.pi**2)) * (np.arctan(gw/(gh+1e-9)) - np.arctan(pw/(ph+1e-9)))**2
    alpha = v / (1 - iou_d + v + 1e-9)

    ciou = iou_d - rho2/c2 - alpha*v
    return 1 - ciou, iou_d

def bce_loss(pred, gt, eps=1e-7):
    """Binary Cross Entropy for class/confidence."""
    pred = np.clip(pred, eps, 1-eps)
    return -(gt * np.log(pred) + (1-gt) * np.log(1-pred))

# Test eğitim adımı simülasyonu
np.random.seed(0)
N_iter = 80
kayip_gecmisi = {"toplam": [], "ciou": [], "cls": [], "conf": []}

for it in range(N_iter):
    ilerleme = it / N_iter
    # Eğitim ilerledikçe azalan kayıplar
    kayip_ciou = 1.8 * np.exp(-3.5 * ilerleme) + 0.15 + np.random.normal(0, 0.04)
    kayip_cls  = 2.2 * np.exp(-4.0 * ilerleme) + 0.08 + np.random.normal(0, 0.05)
    kayip_conf = 1.5 * np.exp(-3.0 * ilerleme) + 0.05 + np.random.normal(0, 0.03)
    toplam     = 0.5*kayip_ciou + 0.3*kayip_cls + 0.2*kayip_conf

    kayip_gecmisi["ciou"].append(max(0, kayip_ciou))
    kayip_gecmisi["cls"].append(max(0, kayip_cls))
    kayip_gecmisi["conf"].append(max(0, kayip_conf))
    kayip_gecmisi["toplam"].append(max(0, toplam))

print(f"  80 iterasyon simülasyonu:")
print(f"  Başlangıç toplam kayıp : {kayip_gecmisi['toplam'][0]:.4f}")
print(f"  Son toplam kayıp       : {kayip_gecmisi['toplam'][-1]:.4f}")
print(f"  CIoU kaybı (son)       : {kayip_gecmisi['ciou'][-1]:.4f}")
print(f"  Sınıf kaybı (son)      : {kayip_gecmisi['cls'][-1]:.4f}")
print(f"  Confidence (son)       : {kayip_gecmisi['conf'][-1]:.4f}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 5: Tespit Pipeline
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 5: Tam Tespit Pipeline Simülasyonu")
print("─" * 65)

def iou_box(b1, b2):
    ix1,iy1,ix2,iy2 = max(b1[0],b2[0]),max(b1[1],b2[1]),\
                       min(b1[2],b2[2]),min(b1[3],b2[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter+1e-9)

def nms_pipeline(boxes, skorlar, siniflar, iou_esik=0.45, conf_esik=0.25):
    """Tam NMS pipeline: confidence filtresi + sınıf bazlı NMS."""
    # Confidence filtresi
    maske = np.array(skorlar) >= conf_esik
    boxes   = [b for b, m in zip(boxes, maske) if m]
    skorlar = [s for s, m in zip(skorlar, maske) if m]
    siniflar = [c for c, m in zip(siniflar, maske) if m]

    if not boxes:
        return [], [], []

    # Sınıf bazlı NMS
    sonuc_box, sonuc_skor, sonuc_sinif = [], [], []
    for sinif in set(siniflar):
        idx = [i for i, c in enumerate(siniflar) if c == sinif]
        s_boxes   = [boxes[i] for i in idx]
        s_skorlar = [skorlar[i] for i in idx]

        # Greedy NMS
        sirali = sorted(range(len(s_skorlar)), key=lambda x: -s_skorlar[x])
        secilen = []
        while sirali:
            en_iyi = sirali[0]; secilen.append(en_iyi)
            sirali = [j for j in sirali[1:]
                      if iou_box(s_boxes[en_iyi], s_boxes[j]) < iou_esik]

        for i in secilen:
            sonuc_box.append(s_boxes[i])
            sonuc_skor.append(s_skorlar[i])
            sonuc_sinif.append(sinif)

    return sonuc_box, sonuc_skor, sonuc_sinif

# Sahte tespit çıktısı
np.random.seed(13)
SINIF_ADLARI = ["araba", "insan", "bisiklet", "kamyon", "motosiklet"]

ham_boxes, ham_skorlar, ham_siniflar = [], [], []
# 3 gerçek nesne, her biri için 3-5 çakışan tahmin
gercek_nesneler = [
    ([80, 60, 220, 180], "araba"),
    ([280, 100, 380, 300], "insan"),
    ([400, 150, 550, 260], "kamyon"),
]
for gt_box, sinif in gercek_nesneler:
    for _ in range(np.random.randint(3, 6)):
        noise = np.random.randint(-20, 20, 4)
        pred  = [max(0, gt_box[i] + noise[i]) for i in range(4)]
        ham_boxes.append(pred)
        ham_skorlar.append(np.random.uniform(0.3, 0.99))
        ham_siniflar.append(sinif)

nms_boxes, nms_skorlar, nms_siniflar = nms_pipeline(
    ham_boxes, ham_skorlar, ham_siniflar)

print(f"  Ham tespit sayısı  : {len(ham_boxes)}")
print(f"  NMS sonrası        : {len(nms_boxes)}")
print(f"\n  {'Sınıf':<14} {'Güven':>8} {'Konum (x1,y1,x2,y2)'}")
print("  " + "─" * 55)
for box, skor, sinif in sorted(zip(nms_boxes, nms_skorlar, nms_siniflar),
                                 key=lambda x: -x[1]):
    print(f"  {sinif:<14} {skor:>8.3f}  {[int(b) for b in box]}")

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

SINIF_RENK = {
    "araba":"#3B82F6","insan":"#22C55E","bisiklet":"#F59E0B",
    "kamyon":"#A78BFA","motosiklet":"#F472B6"
}

# G1 — YOLO grid görselleştirme
ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#161B22")
S = 7; ax1.set_xlim(0, S); ax1.set_ylim(0, S)
ax1.set_aspect("equal")
for sp in ax1.spines.values(): sp.set_color("#30363D")
# Grid çiz
for i in range(S+1):
    ax1.axhline(i, color="#30363D", linewidth=0.8)
    ax1.axvline(i, color="#30363D", linewidth=0.8)
# Test nesnelerini hücrelerde göster
for cx, cy, w, h, ad in test_nesneler:
    tx, ty, tw, th, col, row = grid.encode_box(cx, cy, w, h)
    # Sınırlayıcı kutu (normalize)
    nw = tw * S; nh = th * S
    nx1 = col + tx - nw/2; ny1 = (S-1-row) + ty - nh/2
    renk = {"araba":"#3B82F6","insan":"#22C55E","kamyon":"#A78BFA"}[ad]
    ax1.add_patch(mpatches.Rectangle((nx1, ny1), nw, nh,
        linewidth=2, edgecolor=renk, facecolor=renk, alpha=0.25))
    ax1.scatter([col + tx], [S-1-row + ty], s=80, c=renk, zorder=5)
    ax1.text(col + tx + 0.1, S-1-row + ty + 0.1, ad[:3],
             color="white", fontsize=8, fontweight="bold")
ax1.set_title(f"YOLO Grid ({S}×{S})\nNesne Hücre Ataması", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax1.tick_params(colors="#8B949E")
ax1.invert_yaxis()

# G2 — Anchor box dağılımı
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#161B22")
for sp in ax2.spines.values(): sp.set_color("#30363D")
ax2.scatter(wh_kutulari[::5, 0], wh_kutulari[::5, 1],
            s=8, c="#64748B", alpha=0.4, label="GT kutular")
renk_ankor = ["#22C55E","#3B82F6","#A78BFA"]
for (ankor_set, katman, renk) in zip([anchor_3s, anchor_3m, anchor_3l],
                                      ["P3 küçük","P4 orta","P5 büyük"],
                                      renk_ankor):
    for a in ankor_set:
        ax2.add_patch(mpatches.Rectangle((-a[0]/2, -a[1]/2), a[0], a[1],
            linewidth=2.5, edgecolor=renk, facecolor=renk, alpha=0.15,
            transform=ax2.transData))
        ax2.scatter(a[0], a[1], s=120, c=renk, zorder=5, marker="*")
ax2.set_xlim(0, 500); ax2.set_ylim(0, 600)
ax2.set_title("K-Means Anchor Tasarımı\n(k=9 anchor)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax2.set_xlabel("Genişlik (px)", fontsize=10, color="#8B949E")
ax2.set_ylabel("Yükseklik (px)", fontsize=10, color="#8B949E")
ax2.tick_params(colors="#8B949E")
ax2.grid(alpha=0.3, color="#30363D")

# G3 — FPN özellik haritası piramidi
ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor("#161B22")
ax3.set_xlim(0, 10); ax3.set_ylim(0, 6); ax3.axis("off")
ax3.set_title("FPN Özellik Piramidi\n(640×640 giriş)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
fpn_katmanlar = [
    (4.2, 4.2, 1.6, 1.6, "#3B82F6", "P3: 80×80", "Küçük\nnesneler"),
    (3.4, 2.8, 3.2, 3.2, "#0FBCCE", "P4: 40×40", "Orta\nnesneler"),
    (2.5, 1.2, 5.0, 5.0, "#A78BFA", "P5: 20×20", "Büyük\nnesneler"),
]
for (x, y, w, h, renk, etiket, tip) in fpn_katmanlar:
    ax3.add_patch(mpatches.Rectangle((5-w/2, y-h/2+0.5), w, h,
        linewidth=2, edgecolor=renk, facecolor=renk, alpha=0.22))
    ax3.text(5, y+0.5, etiket, ha="center", va="center",
             fontsize=10, fontweight="bold", color=renk)
    ax3.text(5, y-0.25, tip, ha="center", va="center",
             fontsize=8, color="#8B949E")

# G4 — Kayıp eğrileri
ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor("#161B22")
for sp in ax4.spines.values(): sp.set_color("#30363D")
iterler = range(N_iter)
for ad, renk in [("toplam","#0FBCCE"), ("ciou","#3B82F6"),
                  ("cls","#F59E0B"), ("conf","#22C55E")]:
    ls = "-" if ad == "toplam" else "--"
    lw = 2.8 if ad == "toplam" else 1.8
    ax4.plot(iterler, kayip_gecmisi[ad], ls, color=renk,
             linewidth=lw, label=ad.upper() + " Loss")
ax4.set_title("Eğitim Kayıp Eğrileri", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax4.set_xlabel("İterasyon", fontsize=10, color="#8B949E")
ax4.set_ylabel("Kayıp", fontsize=10, color="#8B949E")
ax4.tick_params(colors="#8B949E")
ax4.grid(alpha=0.3, color="#30363D")
ax4.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# G5 — Ham tespit + NMS sonucu
ax5 = fig.add_subplot(gs[1, 1]); ax5.set_facecolor("#161B22")
ax5.set_xlim(0, 640); ax5.set_ylim(0, 480)
for sp in ax5.spines.values(): sp.set_color("#30363D")
ax5.tick_params(colors="#8B949E")
# Ham kutular
for b, s, c in zip(ham_boxes, ham_skorlar, ham_siniflar):
    renk = SINIF_RENK.get(c, "#64748B")
    ax5.add_patch(mpatches.Rectangle(
        (b[0],b[1]), b[2]-b[0], b[3]-b[1],
        linewidth=1, edgecolor=renk, facecolor=renk, alpha=0.12))
# NMS sonucu
for b, s, c in zip(nms_boxes, nms_skorlar, nms_siniflar):
    renk = SINIF_RENK.get(c, "#64748B")
    ax5.add_patch(mpatches.Rectangle(
        (b[0],b[1]), b[2]-b[0], b[3]-b[1],
        linewidth=3, edgecolor=renk, facecolor="none"))
    ax5.text(b[0]+2, b[1]-8, f"{c[:4]} {s:.2f}",
             fontsize=9, color=renk, fontweight="bold",
             path_effects=[pe.withStroke(linewidth=2, foreground="#0D1117")])
ax5.set_title("Tespit Pipeline\n(soluk=ham, çerçeve=NMS)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax5.invert_yaxis()

# G6 — Model karşılaştırma (mAP vs FPS)
ax6 = fig.add_subplot(gs[1, 2]); ax6.set_facecolor("#161B22")
for sp in ax6.spines.values(): sp.set_color("#30363D")
modeller = {
    "YOLOv5n": (28.0, 450, "#64748B", "^"),
    "YOLOv5s": (37.4, 280, "#3B82F6", "^"),
    "YOLOv5m": (45.4, 180, "#3B82F6", "s"),
    "YOLOv8n": (37.3, 400, "#0FBCCE", "o"),
    "YOLOv8s": (44.9, 260, "#0FBCCE", "o"),
    "YOLOv8m": (50.2, 130, "#0FBCCE", "s"),
    "YOLOv8l": (52.9,  90, "#0FBCCE", "D"),
    "YOLOv9s": (46.8, 200, "#F59E0B", "P"),
    "YOLOv10n":(38.5, 420, "#22C55E", "*"),
    "YOLOv10s":(46.3, 280, "#22C55E", "*"),
}
for ad, (mAP, fps, renk, marker) in modeller.items():
    ax6.scatter(fps, mAP, s=120, c=renk, marker=marker, zorder=4,
                edgecolors="#0D1117", linewidth=1.2)
    ax6.annotate(ad, (fps, mAP), textcoords="offset points",
                 xytext=(5, 4), fontsize=8, color="#C9D1D9")
ax6.set_xlabel("FPS (GPU T4)", fontsize=10, color="#8B949E")
ax6.set_ylabel("mAP50-95 (COCO)", fontsize=10, color="#8B949E")
ax6.set_title("YOLO Ailesi\nmAP vs Hız", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax6.tick_params(colors="#8B949E")
ax6.grid(alpha=0.3, color="#30363D")
handles = [mpatches.Patch(color="#64748B",label="v5"),
           mpatches.Patch(color="#0FBCCE",label="v8"),
           mpatches.Patch(color="#F59E0B",label="v9"),
           mpatches.Patch(color="#22C55E",label="v10")]
ax6.legend(handles=handles, fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# G7 — CIoU vs GIoU vs IoU karşılaştırma
ax7 = fig.add_subplot(gs[2, 0]); ax7.set_facecolor("#161B22")
for sp in ax7.spines.values(): sp.set_color("#30363D")
np.random.seed(3)
gt_test  = [100, 100, 200, 200]
pred_ofsetler = np.linspace(-80, 80, 40)
iou_vals, diou_vals, ciou_vals = [], [], []
for ofs in pred_ofsetler:
    pred = [gt_test[0]+ofs, gt_test[1]+ofs, gt_test[2]+ofs, gt_test[3]+ofs]
    ciou_l, iou_d = ciou_loss(pred, gt_test)
    iou_vals.append(iou_d)
    ciou_vals.append(1 - ciou_l)
ax7.plot(pred_ofsetler, iou_vals,  "-", color="#64748B", linewidth=2, label="IoU")
ax7.plot(pred_ofsetler, ciou_vals, "-", color="#0FBCCE", linewidth=2, label="CIoU benzerliği")
ax7.axvline(0, color="#F59E0B", linestyle="--", linewidth=1.5, label="GT merkezi")
ax7.set_title("CIoU vs IoU\n(Offset Analizi)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax7.set_xlabel("Merkez Offseti (px)", fontsize=10, color="#8B949E")
ax7.set_ylabel("Benzerlik", fontsize=10, color="#8B949E")
ax7.tick_params(colors="#8B949E")
ax7.grid(alpha=0.3, color="#30363D")
ax7.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# G8 — Sınıf başına tespit dağılımı pasta
ax8 = fig.add_subplot(gs[2, 1:]); ax8.set_facecolor("#161B22")
sinif_sayaci = defaultdict(int)
for c in ham_siniflar: sinif_sayaci[c] += 1
renkler8 = [SINIF_RENK.get(c,"#64748B") for c in sinif_sayaci]
ax8.pie(list(sinif_sayaci.values()),
        labels=[f"{c}\n({v} tespit)" for c,v in sinif_sayaci.items()],
        colors=renkler8, autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 10, "color": "#C9D1D9"},
        wedgeprops={"edgecolor": "#0D1117", "linewidth": 2})
ax8.set_title("Ham Tespit Sınıf Dağılımı\n(NMS öncesi)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax8.set_facecolor("#161B22")

fig.suptitle(
    "NESNE TESPİTİ — UYGULAMA 02  |  YOLO Mimarisi & Pipeline\n"
    "Grid · Anchor · FPN · Kayıp · NMS · Model Karşılaştırma",
    fontsize=14, fontweight="bold", color="#C9D1D9", y=0.98)
plt.savefig("cv_02_yolo_mimari.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("  ✅ cv_02_yolo_mimari.png kaydedildi")
plt.close()

print("\n" + "=" * 65)
print(f"  Grid boyutu   : {S}×{S} = {S*S} hücre")
print(f"  Toplam anchor : 9 (3 katman × 3)")
print(f"  Ham tespit    : {len(ham_boxes)}")
print(f"  NMS sonrası   : {len(nms_boxes)}")
print("  ✅ UYGULAMA 02 TAMAMLANDI")
print("=" * 65)
