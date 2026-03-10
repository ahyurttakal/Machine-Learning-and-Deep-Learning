"""
=============================================================================
SEGMENTASYON & NESNE TESPİTİ — UYGULAMA 01
Temel Metrikler: IoU · NMS · Precision · Recall · mAP
=============================================================================
Kapsam:
  - IoU (Intersection over Union) sıfırdan hesaplama
  - Non-Maximum Suppression (NMS) algoritması
  - Precision, Recall, F1-Score
  - Average Precision (AP) — PR eğrisi altındaki alan
  - mean Average Precision (mAP) — çok sınıflı
  - PASCAL VOC ve COCO metrik karşılaştırması
  - Tam simülasyon: gerçek tespit senaryoları
  - 8-panel görselleştirme

Kurulum: pip install numpy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import defaultdict
import warnings; warnings.filterwarnings("ignore")

print("=" * 65)
print("  NESNE TESPİTİ — UYGULAMA 01")
print("  Temel Metrikler: IoU · NMS · Precision · Recall · mAP")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 1: IoU (Intersection over Union)
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 1: IoU — Intersection over Union")
print("─" * 65)

def iou(box1, box2):
    """
    İki bounding box arasındaki IoU değerini hesaplar.
    Format: [x1, y1, x2, y2] (sol-üst, sağ-alt köşe)
    """
    # Kesişim dikdörtgeninin koordinatları
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    # Kesişim alanı (negatif olursa 0)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_alan = inter_w * inter_h

    # Her iki kutunun alanı
    alan1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    alan2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Birleşim alanı
    birlesim = alan1 + alan2 - inter_alan

    return inter_alan / birlesim if birlesim > 0 else 0.0

def iou_vektorel(boxes, query_box):
    """Tek bir kutuyu N kutuya karşı vektörize IoU hesabı."""
    inter_x1 = np.maximum(boxes[:, 0], query_box[0])
    inter_y1 = np.maximum(boxes[:, 1], query_box[1])
    inter_x2 = np.minimum(boxes[:, 2], query_box[2])
    inter_y2 = np.minimum(boxes[:, 3], query_box[3])
    inter    = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    alan1    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    alan2    = (query_box[2] - query_box[0]) * (query_box[3] - query_box[1])
    union    = alan1 + alan2 - inter
    return np.where(union > 0, inter / union, 0.0)

# Test senaryoları
test_ciftleri = [
    ([10, 10, 50, 50], [10, 10, 50, 50], "Mükemmel eşleşme"),
    ([10, 10, 50, 50], [30, 30, 70, 70], "Kısmi örtüşme"),
    ([10, 10, 50, 50], [60, 60, 90, 90], "Örtüşme yok"),
    ([10, 10, 80, 80], [20, 20, 60, 60], "Tam içinde"),
    ([10, 10, 50, 50], [40, 10, 80, 50], "Yatay örtüşme"),
]

print(f"\n  {'Senaryo':<28} {'IoU':>8}  {'Yorum'}")
print("  " + "─" * 60)
for b1, b2, ad in test_ciftleri:
    deger = iou(b1, b2)
    yorum = ("Zayıf" if deger < 0.3 else "Orta" if deger < 0.6 else
             "İyi" if deger < 0.9 else "Mükemmel")
    print(f"  {ad:<28} {deger:>8.4f}  {yorum}")

# IoU eşik analizi
print("\n  IoU eşik değerleri ve yorumları:")
esikler = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
for e in esikler:
    print(f"  IoU@{e:.2f} → {'PASCAL VOC standardı' if e==0.50 else 'COCO ağırlıklı' if e==0.75 else ''}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 2: Non-Maximum Suppression (NMS)
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 2: Non-Maximum Suppression (NMS)")
print("─" * 65)

def nms(boxes, skorlar, iou_esik=0.5):
    """
    Klasik Greedy NMS implementasyonu.
    boxes  : [N, 4] — [x1, y1, x2, y2]
    skorlar: [N]    — confidence değerleri
    """
    if len(boxes) == 0:
        return []

    boxes   = np.array(boxes, dtype=float)
    skorlar = np.array(skorlar, dtype=float)

    # Skora göre azalan sırala
    sirali_idx = np.argsort(-skorlar)
    secilen    = []

    while len(sirali_idx) > 0:
        en_iyi = sirali_idx[0]
        secilen.append(int(en_iyi))

        if len(sirali_idx) == 1:
            break

        # Kalan kutularla IoU hesapla
        kalan_boxes = boxes[sirali_idx[1:]]
        iou_degerleri = iou_vektorel(kalan_boxes, boxes[en_iyi])

        # IoU eşiğinin altındakileri tut
        tut = iou_degerleri < iou_esik
        sirali_idx = sirali_idx[1:][tut]

    return secilen

def soft_nms(boxes, skorlar, sigma=0.5, iou_esik=0.3, skor_esik=0.01):
    """
    Soft-NMS: Bastırmak yerine skoru düşür (Bodla et al. 2017).
    """
    boxes   = np.array(boxes, dtype=float)
    skorlar = np.array(skorlar, dtype=float).copy()
    N       = len(boxes)
    secilen = []

    for i in range(N):
        en_iyi_idx = np.argmax(skorlar)
        if skorlar[en_iyi_idx] < skor_esik:
            break
        secilen.append(int(en_iyi_idx))
        skorlar[en_iyi_idx] = -1  # İşaretle

        for j in range(N):
            if j == en_iyi_idx or skorlar[j] < 0:
                continue
            iou_d = iou(boxes[en_iyi_idx], boxes[j])
            # Gaussian ceza
            skorlar[j] *= np.exp(-(iou_d ** 2) / sigma)

    return secilen

# NMS testi: çakışan tahminler
np.random.seed(42)
N_kutu = 12
# Bir nesne etrafında kümelenmiş kutular oluştur
merkez_x, merkez_y = 100, 100
w, h = 60, 60
test_boxes = []
test_skorlar = []
for _ in range(N_kutu):
    cx = merkez_x + np.random.randint(-15, 15)
    cy = merkez_y + np.random.randint(-15, 15)
    bw = w + np.random.randint(-10, 10)
    bh = h + np.random.randint(-10, 10)
    test_boxes.append([cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2])
    test_skorlar.append(round(np.random.uniform(0.3, 0.99), 3))

nms_sonuc     = nms(test_boxes, test_skorlar, iou_esik=0.5)
soft_nms_sonuc = soft_nms(test_boxes, test_skorlar)

print(f"  Başlangıç kutu sayısı : {N_kutu}")
print(f"  NMS sonrası           : {len(nms_sonuc)} kutu  → {nms_sonuc}")
print(f"  Soft-NMS sonrası      : {len(soft_nms_sonuc)} kutu  → {soft_nms_sonuc}")
print(f"\n  NMS IoU eşiği 0.5 → yakın kutular: bastırıldı")
print(f"  Soft-NMS: çakışan kutular yok edilmez, skoru düşürülür")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 3: Precision, Recall, F1
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 3: Precision · Recall · F1-Score")
print("─" * 65)

def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0
    return precision, recall, f1

def eslestir_tespitler(gt_boxes, pred_boxes, pred_skorlar,
                        iou_esik=0.5):
    """
    GT kutularını tahmin kutularına eşleştir.
    Döndürür: (tp_listesi, fp_listesi, fn_sayisi)
    """
    if not pred_boxes:
        return [], [], len(gt_boxes)

    # Skora göre sırala
    sirali = sorted(zip(pred_skorlar, pred_boxes), reverse=True)
    pred_skorlar_s = [s for s, _ in sirali]
    pred_boxes_s   = [b for _, b in sirali]

    eslesmis_gt = set()
    tp_list, fp_list = [], []
    tp_sayac = 0

    for i, (skor, pred) in enumerate(zip(pred_skorlar_s, pred_boxes_s)):
        en_iyi_iou, en_iyi_gt = 0, -1
        for j, gt in enumerate(gt_boxes):
            if j in eslesmis_gt:
                continue
            iou_d = iou(pred, gt)
            if iou_d > en_iyi_iou:
                en_iyi_iou, en_iyi_gt = iou_d, j

        if en_iyi_iou >= iou_esik and en_iyi_gt >= 0:
            eslesmis_gt.add(en_iyi_gt)
            tp_list.append((skor, 1))
            tp_sayac += 1
        else:
            tp_list.append((skor, 0))
            fp_list.append(skor)

    fn = len(gt_boxes) - len(eslesmis_gt)
    return tp_list, fp_list, fn

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 4: Average Precision (AP) & mAP
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 4: Average Precision (AP) & mAP")
print("─" * 65)

def average_precision(tp_list, n_gt, interp=True):
    """
    PR eğrisinin altındaki alanı hesaplar.
    11-nokta interpolasyon (PASCAL VOC) veya tam alan.
    """
    if not tp_list or n_gt == 0:
        return 0.0

    tp_list = sorted(tp_list, reverse=True, key=lambda x: x[0])
    tp_cum = np.cumsum([x[1] for x in tp_list])
    fp_cum = np.cumsum([1 - x[1] for x in tp_list])

    precisions = tp_cum / (tp_cum + fp_cum)
    recalls    = tp_cum / n_gt

    # Uç noktalara 0 ve 1 ekle
    precisions = np.concatenate([[1.0], precisions, [0.0]])
    recalls    = np.concatenate([[0.0], recalls,    [1.0]])

    if interp:
        # 11-nokta interpolasyon (VOC)
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            mask = recalls >= t
            ap += np.max(precisions[mask]) if np.any(mask) else 0
        ap /= 11
    else:
        # Alan altı (COCO)
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        idx = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])

    return float(ap)

# Simüle senaryolar (6 sınıf)
SINIFLAR = ["araba", "insan", "bisiklet", "kamyon", "motosiklet", "otobüs"]
np.random.seed(7)

sinif_ap_voc  = {}
sinif_ap_coco = {}
sinif_tp_listesi = {}
sinif_ngt = {}

for sinif in SINIFLAR:
    n_gt   = np.random.randint(20, 60)
    n_pred = int(n_gt * np.random.uniform(0.8, 1.4))
    tp_l = []
    tp_sayisi = 0
    for _ in range(n_pred):
        skor = np.random.beta(3, 2)
        if tp_sayisi < n_gt and np.random.random() < 0.72:
            tp_l.append((skor, 1))
            tp_sayisi += 1
        else:
            tp_l.append((skor, 0))

    sinif_tp_listesi[sinif] = tp_l
    sinif_ngt[sinif] = n_gt
    sinif_ap_voc[sinif]  = average_precision(tp_l, n_gt, interp=True)
    sinif_ap_coco[sinif] = average_precision(tp_l, n_gt, interp=False)

map_voc  = np.mean(list(sinif_ap_voc.values()))
map_coco = np.mean(list(sinif_ap_coco.values()))

print(f"  {'Sınıf':<14} {'AP@0.5 (VOC)':>14} {'AP (COCO)':>12} {'GT':>6}")
print("  " + "─" * 52)
for s in SINIFLAR:
    print(f"  {s:<14} {sinif_ap_voc[s]:>14.4f} {sinif_ap_coco[s]:>12.4f} {sinif_ngt[s]:>6}")
print("  " + "─" * 52)
print(f"  {'mAP':<14} {map_voc:>14.4f} {map_coco:>12.4f}")

# COCO mAP@[.5:.95]
iou_esikleri = np.arange(0.50, 1.00, 0.05)
coco_map_list = []
for esik in iou_esikleri:
    # Farklı IoU eşiğinde AP düşer
    decay = (esik - 0.50) * 0.8
    coco_map_list.append(map_coco * (1 - decay))
coco_map_50_95 = np.mean(coco_map_list)
print(f"\n  COCO mAP@[.50:.95] = {coco_map_50_95:.4f}  (10 IoU eşiğinin ortalaması)")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 5: GÖRSELLEŞTİRME (8 panel)
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 5: Görselleştirme (8 panel)")
print("─" * 65)

plt.style.use("seaborn-v0_8-dark")
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#0D1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.44, wspace=0.36,
                        top=0.93, bottom=0.05)

# G1 — IoU görsel
ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#161B22")
ax1.set_xlim(0, 200); ax1.set_ylim(0, 200); ax1.set_aspect("equal")
for sp in ax1.spines.values(): sp.set_color("#30363D")
ax1.tick_params(colors="#8B949E")
gt_box   = [30, 40, 130, 140]
pred_box = [60, 60, 160, 160]
iou_d    = iou(gt_box, pred_box)

rect_gt   = mpatches.Rectangle((gt_box[0], gt_box[1]),
    gt_box[2]-gt_box[0], gt_box[3]-gt_box[1],
    linewidth=2, edgecolor="#22C55E", facecolor="#22C55E", alpha=0.25)
rect_pred = mpatches.Rectangle((pred_box[0], pred_box[1]),
    pred_box[2]-pred_box[0], pred_box[3]-pred_box[1],
    linewidth=2, edgecolor="#3B82F6", facecolor="#3B82F6", alpha=0.25)
ix1,iy1,ix2,iy2 = max(gt_box[0],pred_box[0]),max(gt_box[1],pred_box[1]),\
                   min(gt_box[2],pred_box[2]),min(gt_box[3],pred_box[3])
rect_inter = mpatches.Rectangle((ix1,iy1), ix2-ix1, iy2-iy1,
    linewidth=2, edgecolor="#F59E0B", facecolor="#F59E0B", alpha=0.55)
ax1.add_patch(rect_gt); ax1.add_patch(rect_pred); ax1.add_patch(rect_inter)
ax1.text(80, 90, f"IoU={iou_d:.3f}", ha="center", va="center",
         fontsize=14, fontweight="bold", color="#F59E0B")
ax1.legend(handles=[
    mpatches.Patch(facecolor="#22C55E", alpha=0.5, label="GT Box"),
    mpatches.Patch(facecolor="#3B82F6", alpha=0.5, label="Pred Box"),
    mpatches.Patch(facecolor="#F59E0B", alpha=0.7, label="Kesişim"),
], fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22", loc="upper left")
ax1.set_title("IoU Görselleştirme", fontsize=12, fontweight="bold",
              color="#C9D1D9", pad=8)

# G2 — NMS görselleştirme
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#161B22")
ax2.set_xlim(20, 180); ax2.set_ylim(20, 180); ax2.set_aspect("equal")
for sp in ax2.spines.values(): sp.set_color("#30363D")
ax2.tick_params(colors="#8B949E")
for i, (box, skor) in enumerate(zip(test_boxes, test_skorlar)):
    renk = "#22C55E" if i in nms_sonuc else "#EF4444"
    alpha = 0.85 if i in nms_sonuc else 0.20
    ax2.add_patch(mpatches.Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        linewidth=2 if i in nms_sonuc else 1,
        edgecolor=renk, facecolor=renk, alpha=alpha))
    if i in nms_sonuc:
        ax2.text(box[0]+2, box[1]+5, f"{skor:.2f}",
                 fontsize=8, color="white", fontweight="bold")
ax2.set_title("NMS Sonucu", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax2.legend(handles=[
    mpatches.Patch(facecolor="#22C55E", alpha=0.7, label="Tutulan"),
    mpatches.Patch(facecolor="#EF4444", alpha=0.4, label="Bastırılan"),
], fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# G3 — PR Eğrisi (ilk 3 sınıf)
ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor("#161B22")
for sp in ax3.spines.values(): sp.set_color("#30363D")
renkler3 = ["#3B82F6","#10B981","#F59E0B"]
for sinif, renk in zip(SINIFLAR[:3], renkler3):
    tp_l = sinif_tp_listesi[sinif]
    ngt  = sinif_ngt[sinif]
    tp_l_sorted = sorted(tp_l, reverse=True, key=lambda x: x[0])
    tp_c = np.cumsum([x[1] for x in tp_l_sorted])
    fp_c = np.cumsum([1-x[1] for x in tp_l_sorted])
    prec = tp_c / (tp_c + fp_c + 1e-9)
    rec  = tp_c / ngt
    ax3.plot(rec, prec, color=renk, linewidth=2,
             label=f"{sinif} AP={sinif_ap_voc[sinif]:.2f}")
ax3.set_xlabel("Recall", fontsize=10, color="#8B949E")
ax3.set_ylabel("Precision", fontsize=10, color="#8B949E")
ax3.set_title("Precision-Recall Eğrisi", fontsize=12, fontweight="bold",
              color="#C9D1D9", pad=8)
ax3.tick_params(colors="#8B949E")
ax3.grid(alpha=0.3, color="#30363D")
ax3.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")
ax3.set_xlim(0,1); ax3.set_ylim(0,1)

# G4 — AP sınıf karşılaştırması
ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor("#161B22")
for sp in ax4.spines.values(): sp.set_color("#30363D")
x4 = np.arange(len(SINIFLAR)); w4 = 0.35
renkler4 = ["#3B82F6","#F59E0B"]
ax4.bar(x4-w4/2, list(sinif_ap_voc.values()),  w4, label="AP@0.5 (VOC)",
        color="#3B82F6", edgecolor="#30363D", alpha=0.88)
ax4.bar(x4+w4/2, list(sinif_ap_coco.values()), w4, label="AP (COCO)",
        color="#F59E0B", edgecolor="#30363D", alpha=0.88)
ax4.axhline(map_voc,  color="#3B82F6", linestyle="--", linewidth=1.5, alpha=0.7)
ax4.axhline(map_coco, color="#F59E0B", linestyle="--", linewidth=1.5, alpha=0.7)
ax4.set_xticks(x4); ax4.set_xticklabels(SINIFLAR, color="#C9D1D9", fontsize=9)
ax4.set_title("AP Sınıf Karşılaştırması", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax4.set_ylabel("AP", fontsize=10, color="#8B949E")
ax4.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")
ax4.tick_params(colors="#8B949E")
ax4.grid(axis="y", alpha=0.3, color="#30363D")

# G5 — mAP@[.5:.95]
ax5 = fig.add_subplot(gs[1, 1]); ax5.set_facecolor("#161B22")
for sp in ax5.spines.values(): sp.set_color("#30363D")
ax5.plot(iou_esikleri, coco_map_list, "o-", color="#0FBCCE",
         linewidth=2.5, markersize=8, markerfacecolor="#fff",
         markeredgecolor="#0FBCCE", markeredgewidth=2)
ax5.fill_between(iou_esikleri, coco_map_list, alpha=0.15, color="#0FBCCE")
ax5.axhline(coco_map_50_95, color="#F59E0B", linestyle="--",
            linewidth=1.8, label=f"mAP@[.5:.95]={coco_map_50_95:.3f}")
ax5.set_xlabel("IoU Eşiği", fontsize=10, color="#8B949E")
ax5.set_ylabel("mAP", fontsize=10, color="#8B949E")
ax5.set_title("COCO mAP@[.5:.95] Eğrisi", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax5.tick_params(colors="#8B949E")
ax5.grid(alpha=0.3, color="#30363D")
ax5.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# G6 — IoU ısı haritası
ax6 = fig.add_subplot(gs[1, 2]); ax6.set_facecolor("#161B22")
n = 8
iou_mat = np.zeros((n, n))
np.random.seed(5)
boxes_test = [[np.random.randint(0,80), np.random.randint(0,80),
               np.random.randint(50,120), np.random.randint(50,120)]
              for _ in range(n)]
for i in range(n):
    for j in range(n):
        iou_mat[i, j] = iou(boxes_test[i], boxes_test[j])
im6 = ax6.imshow(iou_mat, cmap="hot", vmin=0, vmax=1, aspect="auto")
plt.colorbar(im6, ax=ax6, shrink=0.85)
ax6.set_title("IoU Matrisi (8×8 Kutu)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax6.tick_params(colors="#8B949E")
ax6.set_xlabel("Kutu İndeksi", fontsize=10, color="#8B949E")
ax6.set_ylabel("Kutu İndeksi", fontsize=10, color="#8B949E")

# G7 — Confidence eşiği vs F1
ax7 = fig.add_subplot(gs[2, 0]); ax7.set_facecolor("#161B22")
for sp in ax7.spines.values(): sp.set_color("#30363D")
esikler7 = np.arange(0.1, 1.0, 0.05)
f1_listesi = []
for esik in esikler7:
    tp_e = sum(1 for s,t in sinif_tp_listesi["araba"] if s >= esik and t == 1)
    fp_e = sum(1 for s,t in sinif_tp_listesi["araba"] if s >= esik and t == 0)
    fn_e = sinif_ngt["araba"] - tp_e
    _, _, f1 = precision_recall_f1(tp_e, fp_e, fn_e)
    f1_listesi.append(f1)
en_iyi_esik = esikler7[np.argmax(f1_listesi)]
ax7.plot(esikler7, f1_listesi, "-", color="#A78BFA", linewidth=2.5)
ax7.axvline(en_iyi_esik, color="#F59E0B", linestyle="--",
            linewidth=1.8, label=f"En iyi eşik={en_iyi_esik:.2f}")
ax7.set_xlabel("Confidence Eşiği", fontsize=10, color="#8B949E")
ax7.set_ylabel("F1 Score", fontsize=10, color="#8B949E")
ax7.set_title("F1 vs Confidence Eşiği\n(araba sınıfı)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ax7.tick_params(colors="#8B949E")
ax7.grid(alpha=0.3, color="#30363D")
ax7.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# G8 — VOC vs COCO karşılaştırma radar
ax8 = fig.add_subplot(gs[2, 1:], projection="polar")
ax8.set_facecolor("#161B22")
cats = SINIFLAR
N8   = len(cats)
ang8 = [n/float(N8)*2*np.pi for n in range(N8)] + [0]
voc_vals  = list(sinif_ap_voc.values())  + [list(sinif_ap_voc.values())[0]]
coco_vals = list(sinif_ap_coco.values()) + [list(sinif_ap_coco.values())[0]]
ax8.plot(ang8, voc_vals,  "o-", color="#3B82F6", linewidth=2.5,
         markersize=7, label="AP@0.5 VOC")
ax8.fill(ang8, voc_vals,  alpha=0.15, color="#3B82F6")
ax8.plot(ang8, coco_vals, "s-", color="#F59E0B", linewidth=2.5,
         markersize=7, label="AP COCO")
ax8.fill(ang8, coco_vals, alpha=0.15, color="#F59E0B")
ax8.set_xticks(ang8[:-1])
ax8.set_xticklabels(cats, fontsize=9, color="#C9D1D9")
ax8.set_ylim(0, 1); ax8.set_facecolor("#161B22")
ax8.tick_params(colors="#8B949E")
ax8.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
           fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")
ax8.set_title("VOC vs COCO AP Radar\n(Tüm Sınıflar)", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=20)

fig.suptitle(
    "NESNE TESPİTİ — Temel Metrikler\n"
    "IoU · NMS · Precision-Recall · AP · mAP",
    fontsize=14, fontweight="bold", color="#C9D1D9", y=0.98)

plt.savefig("cv_01_metrikler.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("  ✅ cv_01_metrikler.png kaydedildi")
plt.close()

print("\n" + "=" * 65)
print(f"  mAP@0.5 (VOC)      : {map_voc:.4f}")
print(f"  mAP (COCO)         : {map_coco:.4f}")
print(f"  mAP@[.5:.95]       : {coco_map_50_95:.4f}")
print(f"  NMS: {N_kutu} → {len(nms_sonuc)} kutu")
print("  ✅ UYGULAMA 01 TAMAMLANDI")
print("=" * 65)
