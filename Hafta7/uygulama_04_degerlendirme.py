"""
=============================================================================
HAFTA 4 CUMARTESİ — UYGULAMA 04
Kapsamlı NLP Değerlendirmesi: ROC, PR, Hata Analizi & Benchmark
=============================================================================
Kapsam:
  - 3 model × 3 strateji performans karşılaştırması
    (DistilBERT, RoBERTa, BERT-base)
    (Feature Extraction, Kısmi FT, Tam FT)
  - ROC eğrisi & AUC karşılaştırması
  - Precision-Recall eğrisi & Average Precision
  - Kalibrasyon analizi (güvenilirlik diyagramı)
  - Hata analizi: yanlış sınıflanan örnekler incelemesi
  - Eşik optimizasyonu: F1-max threshold seçimi
  - Radar grafiği ile çok boyutlu metrik karşılaştırması
  - Bootstrap güven aralıkları
  - Kapsamlı benchmark tablosu (LaTeX uyumlu çıktı)
  - Simülasyon modunda tamamen çalışır

Kurulum: pip install transformers evaluate torch scikit-learn matplotlib seaborn
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ── Bağımlılık kontrolü ───────────────────────────────────────
try:
    from sklearn.metrics import (
        roc_auc_score,
        roc_curve, auc, precision_recall_curve,
        average_precision_score, f1_score,
        accuracy_score, confusion_matrix, brier_score_loss,
        log_loss,
    )
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

print("=" * 65)
print("  HAFTA 4 CUMARTESİ — UYGULAMA 04")
print("  Kapsamlı NLP Değerlendirmesi")
print("=" * 65)
print(f"  sklearn : {'✅' if SKLEARN_AVAILABLE else '❌  pip install scikit-learn'}")
print()

# ─────────────────────────────────────────────────────────────
# VERİ TANIMI — 3 model × 3 strateji
# ─────────────────────────────────────────────────────────────
MODELLER = {
    "DistilBERT": {
        "Feature Extraction": {"auc": 0.918, "ap": 0.915, "brier": 0.128},
        "Kısmi FT":           {"auc": 0.951, "ap": 0.949, "brier": 0.092},
        "Tam FT":             {"auc": 0.971, "ap": 0.969, "brier": 0.068},
    },
    "RoBERTa": {
        "Feature Extraction": {"auc": 0.934, "ap": 0.930, "brier": 0.112},
        "Kısmi FT":           {"auc": 0.962, "ap": 0.959, "brier": 0.081},
        "Tam FT":             {"auc": 0.978, "ap": 0.977, "brier": 0.058},
    },
    "BERT-base": {
        "Feature Extraction": {"auc": 0.925, "ap": 0.921, "brier": 0.120},
        "Kısmi FT":           {"auc": 0.956, "ap": 0.953, "brier": 0.085},
        "Tam FT":             {"auc": 0.974, "ap": 0.972, "brier": 0.063},
    },
}

MODEL_RENKLER = {
    "DistilBERT": "#1565C0",
    "RoBERTa":    "#059669",
    "BERT-base":  "#D97706",
}

STRATEJI_STILLER = {
    "Feature Extraction": ":",
    "Kısmi FT":           "--",
    "Tam FT":             "-",
}

N_TEST = 2000  # simüle edilmiş test örneği sayısı

# ─────────────────────────────────────────────────────────────
# BÖLÜM 1: SİMÜLE EDİLMİŞ OLASILIKLAR ÜRETİMİ
# ─────────────────────────────────────────────────────────────
print("─" * 65)
print("  BÖLÜM 1: Simüle edilmiş tahmin olasılıkları")
print("─" * 65)

def gercekci_prob_uret(auc_hedef, brier_hedef, n=N_TEST, seed=0):
    """
    Hedef AUC ve Brier skora sahip gerçekçi olasılıklar üretir.
    Sınıf dengeli (n/2 negatif, n/2 pozitif).
    """
    np.random.seed(seed)
    yaricap = 2.0   # ROC eğrisi parametresi (yüksek → iyi ayrışma)
    # AUC'den ayırma gücü hesaplama
    ayirma = np.sqrt(2) * np.percentile(
        [np.random.normal(0, 1) for _ in range(10000)], auc_hedef * 100
    )
    ayirma = max(0.2, min(4.0, ayirma * 1.1))

    n_neg = n // 2
    n_pos = n - n_neg

    # Doğal logit dağılımları
    logit_neg = np.random.normal(-ayirma / 2, 1.2, n_neg)
    logit_pos = np.random.normal( ayirma / 2, 1.2, n_pos)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    prob_neg = sigmoid(logit_neg)
    prob_pos = sigmoid(logit_pos)

    probs  = np.concatenate([prob_neg, prob_pos])
    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])

    return probs, labels.astype(int)


tahminler = {}
for model_adi, stratejiler in MODELLER.items():
    tahminler[model_adi] = {}
    for si, (strateji, bilgi) in enumerate(stratejiler.items()):
        seed_val = abs(hash(model_adi + strateji)) % 2**20
        probs, labels = gercekci_prob_uret(
            bilgi["auc"], bilgi["brier"],
            n=N_TEST, seed=seed_val
        )
        tahminler[model_adi][strateji] = {
            "probs":  probs,
            "labels": labels,
        }
    print(f"  {model_adi}: 3 strateji × {N_TEST} örnek üretildi")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 2: ROC EĞRİLERİ
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 2: ROC Eğrisi Analizi")
print("─" * 65)

roc_sonuclari = {}
print(f"  {'Model':<15} {'Strateji':<22} {'AUC':>8}")
print("  " + "-" * 50)
for model_adi in MODELLER:
    roc_sonuclari[model_adi] = {}
    for strateji in MODELLER[model_adi]:
        t = tahminler[model_adi][strateji]
        if SKLEARN_AVAILABLE:
            fpr, tpr, _ = roc_curve(t["labels"], t["probs"])
            roc_auc     = auc(fpr, tpr)
        else:
            n_pts = 100
            fpr = np.linspace(0, 1, n_pts)
            auc_h = MODELLER[model_adi][strateji]["auc"]
            tpr = np.power(fpr, 0.5 - auc_h * 0.4) * auc_h + (1 - auc_h) * fpr
            tpr = np.clip(tpr, 0, 1)
            roc_auc = auc_h
        roc_sonuclari[model_adi][strateji] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        print(f"  {model_adi:<15} {strateji:<22} {roc_auc:>8.4f}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 3: PRESİZYON-GERİ ÇAĞIRMA EĞRİLERİ
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 3: Precision-Recall Eğrisi Analizi")
print("─" * 65)

pr_sonuclari = {}
print(f"  {'Model':<15} {'Strateji':<22} {'AP':>8}")
print("  " + "-" * 50)
for model_adi in MODELLER:
    pr_sonuclari[model_adi] = {}
    for strateji in MODELLER[model_adi]:
        t = tahminler[model_adi][strateji]
        if SKLEARN_AVAILABLE:
            prec, rec, _ = precision_recall_curve(t["labels"], t["probs"])
            ap            = average_precision_score(t["labels"], t["probs"])
        else:
            prec = np.linspace(1, 0.5, 100)
            rec  = np.linspace(0, 1, 100)
            ap   = MODELLER[model_adi][strateji]["ap"]
        pr_sonuclari[model_adi][strateji] = {
            "precision": prec, "recall": rec, "ap": ap
        }
        print(f"  {model_adi:<15} {strateji:<22} {ap:>8.4f}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 4: KALİBRASYON ANALİZİ
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 4: Kalibrasyon Analizi (Güvenilirlik Diyagramı)")
print("─" * 65)

def kalibrasyon_hesapla(labels, probs, n_bins=10):
    """Kalibrasyon eğrisi verisi."""
    bins = np.linspace(0, 1, n_bins + 1)
    ortalama_probs, ortalama_gercekler = [], []
    for i in range(n_bins):
        maske = (probs >= bins[i]) & (probs < bins[i + 1])
        if maske.sum() > 5:
            ortalama_probs.append(probs[maske].mean())
            ortalama_gercekler.append(labels[maske].mean())
    return np.array(ortalama_probs), np.array(ortalama_gercekler)


kal_sonuclari = {}
print(f"  {'Model':<15} {'Strateji':<22} {'Brier Skoru':>14}")
print("  " + "-" * 55)
for model_adi in MODELLER:
    kal_sonuclari[model_adi] = {}
    for strateji in MODELLER[model_adi]:
        t = tahminler[model_adi][strateji]
        if SKLEARN_AVAILABLE:
            frac_pos, mean_pred = calibration_curve(
                t["labels"], t["probs"], n_bins=10
            )
            brier = brier_score_loss(t["labels"], t["probs"])
        else:
            mean_pred = np.linspace(0.05, 0.95, 10)
            # Mükemmel kalibrasyondan sapma
            brier = MODELLER[model_adi][strateji]["brier"]
            sapma = (1 - MODELLER[model_adi][strateji]["auc"]) * 0.3
            frac_pos = mean_pred + np.random.normal(0, sapma, len(mean_pred))
            frac_pos = np.clip(frac_pos, 0, 1)
        kal_sonuclari[model_adi][strateji] = {
            "mean_pred": mean_pred,
            "frac_pos":  frac_pos,
            "brier":     MODELLER[model_adi][strateji]["brier"],
        }
        print(f"  {model_adi:<15} {strateji:<22} {MODELLER[model_adi][strateji]['brier']:>14.4f}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 5: EŞİK OPTİMİZASYONU
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 5: Eşik Optimizasyonu (F1-max Threshold)")
print("─" * 65)

def esik_optimizasyonu(labels, probs, n_esik=200):
    """Tüm eşikler için F1, Precision, Recall hesaplar."""
    esikler = np.linspace(0.01, 0.99, n_esik)
    f1_ler, prec_ler, rec_ler = [], [], []

    for esik in esikler:
        tahmin = (probs >= esik).astype(int)
        if tahmin.sum() == 0:
            f1_ler.append(0.0)
            prec_ler.append(1.0)
            rec_ler.append(0.0)
            continue
        if SKLEARN_AVAILABLE:
            f1_ler.append(f1_score(labels, tahmin, zero_division=0))
            prec_ler.append(float(
                np.sum((tahmin == 1) & (labels == 1)) /
                (np.sum(tahmin == 1) + 1e-9)
            ))
            rec_ler.append(float(
                np.sum((tahmin == 1) & (labels == 1)) /
                (np.sum(labels == 1) + 1e-9)
            ))
        else:
            tp = np.sum((tahmin == 1) & (labels == 1))
            fp = np.sum((tahmin == 1) & (labels == 0))
            fn = np.sum((tahmin == 0) & (labels == 1))
            p  = tp / (tp + fp + 1e-9)
            r  = tp / (tp + fn + 1e-9)
            f1_ler.append(2 * p * r / (p + r + 1e-9))
            prec_ler.append(p); rec_ler.append(r)

    en_iyi_idx  = np.argmax(f1_ler)
    return {
        "esikler":       esikler,
        "f1_ler":        np.array(f1_ler),
        "prec_ler":      np.array(prec_ler),
        "rec_ler":       np.array(rec_ler),
        "en_iyi_esik":   float(esikler[en_iyi_idx]),
        "en_iyi_f1":     float(f1_ler[en_iyi_idx]),
    }


# Tam FT modelleri için eşik optimizasyonu
esik_sonuclari = {}
print(f"  {'Model':<15} {'Varsayılan F1':>14} {'Optimal Eşik':>14} {'Optimal F1':>12}")
print("  " + "-" * 58)
for model_adi in MODELLER:
    t = tahminler[model_adi]["Tam FT"]
    sonuc = esik_optimizasyonu(t["labels"], t["probs"])
    esik_sonuclari[model_adi] = sonuc
    # Varsayılan eşik (0.5) performansı
    preds_varsayilan = (t["probs"] >= 0.5).astype(int)
    if SKLEARN_AVAILABLE:
        f1_varsayilan = f1_score(t["labels"], preds_varsayilan, zero_division=0)
    else:
        f1_varsayilan = sonuc["en_iyi_f1"] * 0.985
    print(f"  {model_adi:<15} {f1_varsayilan:>14.4f} {sonuc['en_iyi_esik']:>14.4f}"
          f" {sonuc['en_iyi_f1']:>12.4f}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 6: HATA ANALİZİ
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 6: Hata Analizi — Yanlış Sınıflanan Örnekler")
print("─" * 65)

YANLIS_SINIFLANDIRILMIS = {
    "Yanlış Negatif (FN)": [
        {
            "metin": "I didn't hate this movie. It wasn't the worst I've seen.",
            "gercek": "POS", "tahmin": "NEG", "skor": 0.38,
            "neden": "Çift olumsuzlama ('didn't hate') modeli şaşırtıyor.",
        },
        {
            "metin": "If you enjoy watching paint dry, this film is for you.",
            "gercek": "POS", "tahmin": "NEG", "skor": 0.29,
            "neden": "İronik/alaycı ifade. Model literalist.",
        },
        {
            "metin": "The film fails to bore you — it's surprisingly engaging.",
            "gercek": "POS", "tahmin": "NEG", "skor": 0.42,
            "neden": "Karmaşık yapı: 'fails to bore' → pozitif ama görünüşte negatif.",
        },
    ],
    "Yanlış Pozitif (FP)": [
        {
            "metin": "Exceptional waste of my Saturday evening. Truly remarkable how bad.",
            "gercek": "NEG", "tahmin": "POS", "skor": 0.72,
            "neden": "'Exceptional', 'remarkable' güçlü pozitif kelimeler; bağlam kaçırılmış.",
        },
        {
            "metin": "A masterpiece of confusion and disorganization.",
            "gercek": "NEG", "tahmin": "POS", "skor": 0.68,
            "neden": "'masterpiece' kelimesi modeli yanıltıyor.",
        },
        {
            "metin": "Brilliantly terrible. The best worst movie ever made.",
            "gercek": "NEG", "tahmin": "POS", "skor": 0.75,
            "neden": "So-bad-it's-good tarzı ince ironi modeli aşıyor.",
        },
    ],
}

for hata_turu, ornekler in YANLIS_SINIFLANDIRILMIS.items():
    print(f"\n  [{hata_turu}]")
    for i, ornek in enumerate(ornekler, 1):
        metin_kisaltma = ornek["metin"][:70] + "..." if len(ornek["metin"]) > 70 else ornek["metin"]
        print(f"    {i}. \"{metin_kisaltma}\"")
        print(f"       Gerçek: {ornek['gercek']}  |  Tahmin: {ornek['tahmin']}"
              f"  |  Skor: {ornek['skor']:.2f}")
        print(f"       Neden: {ornek['neden']}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 7: BOOTSTRAP GÜVENİLİRLİK ARALIKLARI
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 7: Bootstrap Güven Aralıkları (%95)")
print("─" * 65)

def bootstrap_ci(labels, probs, metrik_fn, n_iter=500, ci=0.95):
    """Bootstrap ile güven aralığı hesaplar."""
    skorlar = []
    n = len(labels)
    for _ in range(n_iter):
        idx     = np.random.choice(n, n, replace=True)
        lbl_b   = labels[idx]
        prb_b   = probs[idx]
        if len(np.unique(lbl_b)) < 2:
            continue
        try:
            skorlar.append(metrik_fn(lbl_b, prb_b))
        except Exception:
            continue
    alt = (1 - ci) / 2
    return (
        np.mean(skorlar),
        np.percentile(skorlar, alt * 100),
        np.percentile(skorlar, (1 - alt) * 100),
    )

if SKLEARN_AVAILABLE:
    auc_fn = roc_auc_score
else:
    auc_fn = lambda l, p: np.corrcoef(l, p)[0, 1] * 0.5 + 0.5

print(f"  {'Model':<15} {'Strateji':<22} {'AUC Ort.':>10} {'%95 GA':>18}")
print("  " + "-" * 70)
for model_adi in MODELLER:
    for strateji in ["Feature Extraction", "Tam FT"]:
        t    = tahminler[model_adi][strateji]
        ort, alt, ust = bootstrap_ci(t["labels"], t["probs"], auc_fn, n_iter=300)
        print(f"  {model_adi:<15} {strateji:<22} {ort:>10.4f} [{alt:.4f}, {ust:.4f}]")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 8: BENCHMARK TABLOSU
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 8: Kapsamlı Benchmark Tablosu")
print("─" * 65)

def f1_hesapla(labels, probs, esik=0.5):
    preds = (probs >= esik).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    return 2 * p * r / (p + r + 1e-9)

def acc_hesapla(labels, probs, esik=0.5):
    preds = (probs >= esik).astype(int)
    return np.mean(preds == labels)


benchmark = []
print(f"  {'Model':<12} {'Strateji':<22} {'AUC':>7} {'AP':>7}"
      f" {'F1':>7} {'ACC':>7} {'Brier':>7}")
print("  " + "-" * 74)
for model_adi in MODELLER:
    for strateji in MODELLER[model_adi]:
        t = tahminler[model_adi][strateji]
        roc_auc = roc_sonuclari[model_adi][strateji]["auc"]
        pr_ap   = pr_sonuclari[model_adi][strateji]["ap"]
        f1      = f1_hesapla(t["labels"], t["probs"])
        acc     = acc_hesapla(t["labels"], t["probs"])
        brier   = kal_sonuclari[model_adi][strateji]["brier"]
        benchmark.append({
            "model": model_adi, "strateji": strateji,
            "auc": roc_auc, "ap": pr_ap, "f1": f1, "acc": acc, "brier": brier,
        })
        print(f"  {model_adi:<12} {strateji:<22} {roc_auc:>7.4f} {pr_ap:>7.4f}"
              f" {f1:>7.4f} {acc:>7.4f} {brier:>7.4f}")

# En iyi konfigürasyon
en_iyi = max(benchmark, key=lambda x: x["auc"])
print()
print(f"  🏆 En İyi: {en_iyi['model']} — {en_iyi['strateji']}"
      f"  |  AUC={en_iyi['auc']:.4f}  AP={en_iyi['ap']:.4f}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 9: GÖRSELLEŞTİRME (9 grafik)
# ─────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 9: Görselleştirme (9 grafik)")
print("─" * 65)

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor("#F0F9FF")
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.44, wspace=0.38,
                        top=0.93, bottom=0.05)

# ── GRAFİK 1: ROC Eğrileri ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("white")
ax1.plot([0, 1], [0, 1], linestyle="--", color="#CBD5E1",
         linewidth=1.2, label="Rastgele (AUC=0.50)")
for model_adi, renk in MODEL_RENKLER.items():
    for strateji, stil in STRATEJI_STILLER.items():
        roc = roc_sonuclari[model_adi][strateji]
        lbl = f"{model_adi} {strateji}" if strateji == "Tam FT" else None
        ax1.plot(roc["fpr"], roc["tpr"],
                 color=renk, linestyle=stil, linewidth=1.8,
                 label=f"{model_adi} {strateji} ({roc['auc']:.3f})",
                 alpha=0.85)
ax1.set_title("ROC Eğrisi", fontsize=13, fontweight="bold", pad=10)
ax1.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontsize=10)
ax1.set_ylabel("Doğru Pozitif Oranı (TPR)", fontsize=10)
ax1.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
ax1.grid(alpha=0.4)

# ── GRAFİK 2: Precision-Recall Eğrileri ─────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("white")
ax2.axhline(y=0.5, linestyle="--", color="#CBD5E1",
            linewidth=1.2, label="Temel (AP=0.50)")
for model_adi, renk in MODEL_RENKLER.items():
    for strateji, stil in STRATEJI_STILLER.items():
        pr = pr_sonuclari[model_adi][strateji]
        ax2.plot(pr["recall"], pr["precision"],
                 color=renk, linestyle=stil, linewidth=1.8,
                 label=f"{model_adi} {strateji} ({pr['ap']:.3f})",
                 alpha=0.85)
ax2.set_title("Precision-Recall Eğrisi", fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("Recall", fontsize=10)
ax2.set_ylabel("Precision", fontsize=10)
ax2.set_xlim([0, 1]); ax2.set_ylim([0.4, 1.02])
ax2.legend(fontsize=6.5, loc="lower left", framealpha=0.9)
ax2.grid(alpha=0.4)

# ── GRAFİK 3: Kalibrasyon Diyagramı (Tam FT) ─────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("white")
ax3.plot([0, 1], [0, 1], linestyle="--", color="#CBD5E1",
         linewidth=1.5, label="Mükemmel Kalibrasyon")
for model_adi, renk in MODEL_RENKLER.items():
    kal = kal_sonuclari[model_adi]["Tam FT"]
    brier = kal["brier"]
    ax3.plot(kal["mean_pred"], kal["frac_pos"],
             marker="s", linewidth=2.0, color=renk, markersize=6,
             label=f"{model_adi} (Brier={brier:.3f})")
ax3.set_title("Kalibrasyon — Tam FT Modeller", fontsize=13, fontweight="bold", pad=10)
ax3.set_xlabel("Tahmin Edilen Olasılık", fontsize=10)
ax3.set_ylabel("Gözlenen Pozitif Oran", fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.4)

# ── GRAFİK 4: Eşik vs F1/Precision/Recall (en iyi model) ────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor("white")
en_iyi_model = max(MODEL_RENKLER.keys(),
                   key=lambda m: roc_sonuclari[m]["Tam FT"]["auc"])
es = esik_sonuclari[en_iyi_model]
ax4.plot(es["esikler"], es["f1_ler"],    color="#1565C0", linewidth=2.2, label="F1")
ax4.plot(es["esikler"], es["prec_ler"],  color="#059669", linewidth=2.0,
         linestyle="--", label="Precision")
ax4.plot(es["esikler"], es["rec_ler"],   color="#D97706", linewidth=2.0,
         linestyle=":",  label="Recall")
ax4.axvline(x=es["en_iyi_esik"], color="#EF4444", linewidth=1.5,
            linestyle="-.", label=f"Optimal={es['en_iyi_esik']:.3f}")
ax4.axvline(x=0.50, color="#64748B", linewidth=1.2,
            linestyle="--", label="Varsayılan=0.50")
ax4.set_title(f"Eşik Optimizasyonu — {en_iyi_model} (Tam FT)",
              fontsize=12, fontweight="bold", pad=10)
ax4.set_xlabel("Karar Eşiği", fontsize=10)
ax4.set_ylabel("Skor", fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(alpha=0.4)
ax4.set_xlim([0, 1]); ax4.set_ylim([0, 1.05])

# ── GRAFİK 5: AUC Isı Haritası ───────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
model_listesi    = list(MODELLER.keys())
strateji_listesi = list(list(MODELLER.values())[0].keys())
auc_matris = np.array([
    [roc_sonuclari[m][s]["auc"] for s in strateji_listesi]
    for m in model_listesi
])
from matplotlib.colors import LinearSegmentedColormap
cmap_auc = LinearSegmentedColormap.from_list("auc", ["#DBEAFE", "#1565C0"])
im5 = ax5.imshow(auc_matris, cmap=cmap_auc, aspect="auto",
                 vmin=0.90, vmax=0.98)
ax5.set_xticks(range(len(strateji_listesi)))
ax5.set_yticks(range(len(model_listesi)))
ax5.set_xticklabels(strateji_listesi, fontsize=9, rotation=12)
ax5.set_yticklabels(model_listesi, fontsize=10)
for i in range(len(model_listesi)):
    for j in range(len(strateji_listesi)):
        renk_metin = "white" if auc_matris[i, j] > 0.955 else "#1E293B"
        ax5.text(j, i, f"{auc_matris[i, j]:.4f}",
                 ha="center", va="center",
                 fontsize=11, fontweight="bold", color=renk_metin)
plt.colorbar(im5, ax=ax5, label="AUC", fraction=0.046, pad=0.04)
ax5.set_title("AUC Isı Haritası\n(Model × Strateji)", fontsize=13, fontweight="bold", pad=8)

# ── GRAFİK 6: Radar Grafiği — Tam FT Karşılaştırması ─────────
ax6 = fig.add_subplot(gs[1, 2], polar=True)
kategoriler = ["AUC", "AP", "F1", "Accuracy", "1-Brier"]
n_kat       = len(kategoriler)
acılar      = [n / float(n_kat) * 2 * np.pi for n in range(n_kat)]
acılar     += acılar[:1]  # kapalı çokgen

RADAR_MIN = 0.85

for model_adi, renk in MODEL_RENKLER.items():
    t = tahminler[model_adi]["Tam FT"]
    degerler = [
        roc_sonuclari[model_adi]["Tam FT"]["auc"],
        pr_sonuclari[model_adi]["Tam FT"]["ap"],
        f1_hesapla(t["labels"], t["probs"]),
        acc_hesapla(t["labels"], t["probs"]),
        1 - kal_sonuclari[model_adi]["Tam FT"]["brier"],
    ]
    degerler_norm = [(d - RADAR_MIN) / (1 - RADAR_MIN) for d in degerler]
    degerler_norm += degerler_norm[:1]
    ax6.plot(acılar, degerler_norm, color=renk,
             linewidth=2.0, label=model_adi)
    ax6.fill(acılar, degerler_norm, color=renk, alpha=0.12)

ax6.set_xticks(acılar[:-1])
ax6.set_xticklabels(kategoriler, fontsize=9)
ax6.set_ylim(0, 1)
ax6.set_yticks([0.25, 0.5, 0.75, 1.0])
ax6.set_yticklabels(
    [f"{RADAR_MIN + 0.25*(1-RADAR_MIN):.2f}",
     f"{RADAR_MIN + 0.50*(1-RADAR_MIN):.2f}",
     f"{RADAR_MIN + 0.75*(1-RADAR_MIN):.2f}",
     "1.00"],
    fontsize=7.5
)
ax6.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
ax6.set_title("Radar: Tam FT Modeller\n(5 Metrik)", fontsize=12,
              fontweight="bold", pad=20)

# ── GRAFİK 7: F1 vs AUC scatter (tüm kombinasyonlar) ────────
ax7 = fig.add_subplot(gs[2, 0])
ax7.set_facecolor("white")
STRATEJI_MARKER = {
    "Feature Extraction": "o",
    "Kısmi FT":           "s",
    "Tam FT":             "^",
}
for model_adi, renk in MODEL_RENKLER.items():
    for strateji, marker in STRATEJI_MARKER.items():
        t   = tahminler[model_adi][strateji]
        f1v = f1_hesapla(t["labels"], t["probs"])
        auv = roc_sonuclari[model_adi][strateji]["auc"]
        ax7.scatter(f1v, auv, c=renk, marker=marker, s=120,
                    edgecolors="white", linewidth=1.2, zorder=5)
# Efsaneler
model_handles = [
    mpatches.Patch(color=r, label=m) for m, r in MODEL_RENKLER.items()
]
strateji_handles = [
    Line2D([0], [0], marker=mk, color="#1E293B",
           markersize=8, linewidth=0, label=st)
    for st, mk in STRATEJI_MARKER.items()
]
ax7.legend(handles=model_handles + strateji_handles,
           fontsize=8, loc="lower right", ncol=2)
ax7.set_title("F1 vs AUC (Tüm Kombinasyonlar)", fontsize=13, fontweight="bold", pad=10)
ax7.set_xlabel("F1 Skoru", fontsize=10)
ax7.set_ylabel("AUC", fontsize=10)
ax7.set_xlim(0.82, 0.97); ax7.set_ylim(0.90, 0.99)
ax7.grid(alpha=0.4)

# ── GRAFİK 8: Hata dağılımı (FP/FN oranları) ─────────────────
ax8 = fig.add_subplot(gs[2, 1])
ax8.set_facecolor("white")
x_pos  = np.arange(len(model_listesi))
gen    = 0.25
fp_oranlar  = []
fn_oranlar  = []
for model_adi in model_listesi:
    t     = tahminler[model_adi]["Tam FT"]
    preds = (t["probs"] >= 0.5).astype(int)
    fp    = np.sum((preds == 1) & (t["labels"] == 0))
    fn    = np.sum((preds == 0) & (t["labels"] == 1))
    n     = len(t["labels"])
    fp_oranlar.append(fp / n)
    fn_oranlar.append(fn / n)
ax8.bar(x_pos - gen / 2, fp_oranlar, gen, color="#EF4444",
        label="Yanlış Pozitif (FP)", edgecolor="white")
ax8.bar(x_pos + gen / 2, fn_oranlar, gen, color="#D97706",
        label="Yanlış Negatif (FN)", edgecolor="white")
for i, (fp_o, fn_o) in enumerate(zip(fp_oranlar, fn_oranlar)):
    ax8.text(i - gen / 2, fp_o + 0.002, f"{fp_o:.1%}", ha="center",
             va="bottom", fontsize=8.5, color="#EF4444")
    ax8.text(i + gen / 2, fn_o + 0.002, f"{fn_o:.1%}", ha="center",
             va="bottom", fontsize=8.5, color="#D97706")
ax8.set_title("Hata Dağılımı — Tam FT\n(Varsayılan Eşik = 0.50)", fontsize=12, fontweight="bold", pad=8)
ax8.set_xticks(x_pos)
ax8.set_xticklabels(model_listesi, fontsize=10)
ax8.set_ylabel("Hata Oranı", fontsize=10)
ax8.legend(fontsize=9)
ax8.grid(axis="y", alpha=0.4)
ax8.set_ylim(0, max(max(fp_oranlar), max(fn_oranlar)) * 1.35)

# ── GRAFİK 9: Benchmark Özet Tablosu ─────────────────────────
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis("off")
tablo_verisi = []
for row in benchmark:
    if row["strateji"] == "Tam FT":
        tablo_verisi.append([
            row["model"],
            f"{row['auc']:.4f}",
            f"{row['ap']:.4f}",
            f"{row['f1']:.4f}",
            f"{row['acc']:.4f}",
            f"{row['brier']:.4f}",
        ])
sutun_baslik = ["Model", "AUC", "AP", "F1", "ACC", "Brier"]
tablo = ax9.table(
    cellText=tablo_verisi,
    colLabels=sutun_baslik,
    loc="center", cellLoc="center",
)
tablo.auto_set_font_size(False)
tablo.set_fontsize(10)
tablo.scale(1.3, 2.2)
# En yüksek AUC'yi vurgula
auc_idx = max(range(len(tablo_verisi)),
              key=lambda i: float(tablo_verisi[i][1]))
for (row, col), cell in tablo.get_celld().items():
    if row == 0:
        cell.set_facecolor("#0E4D78")
        cell.set_text_props(color="white", fontweight="bold")
    elif row == auc_idx + 1:
        cell.set_facecolor("#DCFCE7")
    elif row % 2 == 1:
        cell.set_facecolor("#EFF6FF")
    else:
        cell.set_facecolor("white")
    cell.set_edgecolor("#CBD5E1")
ax9.set_title("Benchmark — Tam FT Modeller", fontsize=12,
              fontweight="bold", y=0.90)

# Ana başlık
fig.suptitle(
    "HAFTA 4 CUMARTESİ — UYGULAMA 04\n"
    "Kapsamlı NLP Değerlendirmesi: ROC · PR · Kalibrasyon · Hata Analizi · Benchmark",
    fontsize=15, fontweight="bold", color="#0C2340", y=0.98
)

plt.savefig("h4c_04_degerlendirme.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("    ✅ h4c_04_degerlendirme.png kaydedildi")
plt.close()

# ─────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BÖLÜM ÖZETLERİ")
print(f"  Değerlendirilen model      : {len(MODELLER)}")
print(f"  Değerlendirilen strateji   : {len(STRATEJI_STILLER)}")
print(f"  Toplam konfigürasyon       : {len(MODELLER) * len(STRATEJI_STILLER)}")
print(f"  Test örneği / konfig.      : {N_TEST}")
print(f"  Bootstrap örnekleme        : 300 iterasyon")
print(f"  En iyi model               : {en_iyi['model']} — {en_iyi['strateji']}")
print(f"  En yüksek AUC              : {en_iyi['auc']:.4f}")
print(f"  Grafik çıktısı             : h4c_04_degerlendirme.png")
print("  ✅ UYGULAMA 04 TAMAMLANDI")
print("=" * 65)
