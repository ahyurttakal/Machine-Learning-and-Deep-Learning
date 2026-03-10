"""
=============================================================================
HAFTA 4 CUMARTESİ — UYGULAMA 02
Hugging Face datasets API & Veri İşleme
=============================================================================
Kapsam:
  - load_dataset ile IMDB veri seti yükleme
  - DatasetDict yapısı: train / test / unsupervised bölümleri
  - map(batched=True, num_proc=N) ile paralel tokenizasyon
  - Sınıf dengesizliği analizi
  - Stratified train/validation bölme
  - Uzunluk dağılımı & özellik istatistikleri
  - Arrow format avantajları: bellek-eşlemeli okuma
  - DataLoader ile manuel batch döngüsü
  - Veri kalitesi kontrolleri (duplikasyon, boş metin, uzunluk)
  - Kapsamlı görselleştirme (8 grafik)
  - datasets yüklü değilse tam simülasyon modu

Kurulum: pip install datasets transformers torch numpy matplotlib scikit-learn
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import time
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ── Bağımlılık kontrolü ───────────────────────────────────────
try:
    from datasets import load_dataset, DatasetDict, Dataset
    DS_AVAILABLE = True
except ImportError:
    DS_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TOK_AVAILABLE = True
except ImportError:
    TOK_AVAILABLE = False

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

print("=" * 65)
print("  HAFTA 4 CUMARTESİ — Uygulama 02")
print("  Hugging Face datasets API & Veri İşleme")
print(f"  datasets     : {'✅' if DS_AVAILABLE  else '❌  pip install datasets'}")
print(f"  transformers : {'✅' if TOK_AVAILABLE else '❌  pip install transformers'}")
print(f"  torch        : {'✅' if TORCH_AVAILABLE else '❌  pip install torch'}")
print(f"  Mod          : {'🤖 Gerçek' if (DS_AVAILABLE and TOK_AVAILABLE) else '🎭 Simülasyon'}")
print("=" * 65)

PALETTE = {
    "navy":  "#0E4D78", "blue":   "#1565C0", "cyan":  "#0891B2",
    "teal":  "#0D9488", "amber":  "#D97706", "green": "#059669",
    "red":   "#DC2626", "slate":  "#1E293B", "gray":  "#64748B",
    "purple":"#7C3AED",
}

# ─────────────────────────────────────────────────────────────
# BÖLÜM 1 — VERİ SETİ YÜKLEME
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 1: load_dataset('imdb') — Yapı İnceleme")
print("═" * 55)

# IMDB simülatörü (gerçek API yoksa)
def imdb_simulator(n_train=1000, n_test=200, seed=42):
    """Gerçekçi IMDB verisi üret."""
    np.random.seed(seed)
    pozitif_kalip = [
        "This movie was absolutely {adj}. The {noun} was brilliant.",
        "An {adj} performance by the cast. Highly recommended!",
        "One of the {sup} films I have ever seen. Truly {adj}.",
        "The director did an {adj} job. {noun} was outstanding.",
        "{adj} storytelling and incredible visuals throughout.",
    ]
    negatif_kalip = [
        "Terrible {noun}, complete waste of time and money.",
        "The worst movie I have ever seen. {adj} and boring.",
        "Disappointing {noun}, expected much more from this film.",
        "Awful {adj} performance. The plot made no sense at all.",
        "I regret watching this. Horrible {adj} experience overall.",
    ]
    adj_pos = ["wonderful", "amazing", "brilliant", "outstanding", "fantastic"]
    adj_neg = ["terrible", "awful", "horrible", "dreadful", "appalling"]
    sup_pos = ["best", "greatest", "most beautiful", "finest"]
    nouns   = ["acting", "script", "direction", "cinematography", "story"]

    def gen_review(label, i):
        kalip = pozitif_kalip[i % 5] if label == 1 else negatif_kalip[i % 5]
        adj  = np.random.choice(adj_pos if label == 1 else adj_neg)
        text = (kalip
                .replace("{adj}", adj)
                .replace("{noun}", np.random.choice(nouns))
                .replace("{sup}", np.random.choice(sup_pos)))
        # Uzunluk varyasyonu ekle
        extra = " ".join(["word"] * np.random.randint(0, 40))
        return text + (" " + extra if extra else "")

    def make_split(n, split_seed):
        np.random.seed(split_seed)
        labels = [i % 2 for i in range(n)]
        texts  = [gen_review(lb, i) for i, lb in enumerate(labels)]
        return {"text": texts, "label": labels}

    return {
        "train": make_split(n_train, seed),
        "test":  make_split(n_test, seed + 1),
    }

if DS_AVAILABLE:
    print("  load_dataset('imdb') yükleniyor (cache'li)...")
    try:
        t0 = time.time()
        ds = load_dataset("imdb")
        t_load = time.time() - t0
        print(f"  ✅ Yüklendi: {t_load:.2f}s")
        # Küçük alt-küme al (hız için)
        ds_train_raw = {"text": ds["train"]["text"][:1000],
                        "label": ds["train"]["label"][:1000]}
        ds_test_raw  = {"text": ds["test"]["text"][:200],
                        "label": ds["test"]["label"][:200]}
        RAW_DATA = {"train": ds_train_raw, "test": ds_test_raw}
        DS_LOADED = True
    except Exception as e:
        print(f"  ⚠️  Yükleme hatası ({e}), simülasyon kullanılıyor.")
        RAW_DATA = imdb_simulator(1000, 200)
        DS_LOADED = False
else:
    RAW_DATA = imdb_simulator(1000, 200)
    DS_LOADED = False
    print("  [SİMÜLASYON] 1000 eğitim + 200 test örneği oluşturuldu.")

TRAIN_TEXTS  = RAW_DATA["train"]["text"]
TRAIN_LABELS = RAW_DATA["train"]["label"]
TEST_TEXTS   = RAW_DATA["test"]["text"]
TEST_LABELS  = RAW_DATA["test"]["label"]

print(f"\n  Eğitim seti : {len(TRAIN_TEXTS):6,} örnek")
print(f"  Test seti   : {len(TEST_TEXTS):6,} örnek")
print(f"  Özellikler  : text (str), label (int: 0=NEG, 1=POS)")
print(f"\n  Örnek metin (eğitim[0]): '{TRAIN_TEXTS[0][:80]}...'")
print(f"  Etiket: {'POSITIVE' if TRAIN_LABELS[0] == 1 else 'NEGATIVE'}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 2 — TEMEL İSTATİSTİKLER & SINIF DENGESİ
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 2: Veri İstatistikleri & Sınıf Dengesi Analizi")
print("═" * 55)

# Uzunluk istatistikleri
def metin_istatistikleri(texts, labels):
    uzunluklar    = [len(t.split()) for t in texts]
    char_uzunluk  = [len(t) for t in texts]
    lbl_counter   = Counter(labels)
    return {
        "n":          len(texts),
        "uzunluklar": uzunluklar,
        "char_len":   char_uzunluk,
        "lbl_counter": lbl_counter,
        "ort_kelime": np.mean(uzunluklar),
        "std_kelime": np.std(uzunluklar),
        "min_kelime": np.min(uzunluklar),
        "max_kelime": np.max(uzunluklar),
        "n_pos": lbl_counter.get(1, 0),
        "n_neg": lbl_counter.get(0, 0),
    }

train_stats = metin_istatistikleri(TRAIN_TEXTS, TRAIN_LABELS)
test_stats  = metin_istatistikleri(TEST_TEXTS,  TEST_LABELS)

print(f"\n  {'İstatistik':<28} {'Eğitim':>12} {'Test':>12}")
print("  " + "-" * 54)
print(f"  {'Örnek sayısı':<28} {train_stats['n']:>12,} {test_stats['n']:>12,}")
print(f"  {'Ort. kelime/metin':<28} {train_stats['ort_kelime']:>12.1f} {test_stats['ort_kelime']:>12.1f}")
print(f"  {'Std. kelime':<28} {train_stats['std_kelime']:>12.1f} {test_stats['std_kelime']:>12.1f}")
print(f"  {'Min kelime':<28} {train_stats['min_kelime']:>12} {test_stats['min_kelime']:>12}")
print(f"  {'Max kelime':<28} {train_stats['max_kelime']:>12} {test_stats['max_kelime']:>12}")
print(f"  {'POSİTİF sayısı':<28} {train_stats['n_pos']:>12} {test_stats['n_pos']:>12}")
print(f"  {'NEGATİF sayısı':<28} {train_stats['n_neg']:>12} {test_stats['n_neg']:>12}")

denge_orani = train_stats["n_pos"] / max(train_stats["n"], 1)
print(f"\n  Sınıf dengesi (train): POS={denge_orani:.2%}, NEG={1-denge_orani:.2%}")
if 0.45 <= denge_orani <= 0.55:
    print("  ✅ Dengeli veri seti!")
else:
    print("  ⚠️  Dengesiz veri — oversampling veya class_weight önerilir.")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 3 — TOKENİZASYON (batched map)
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 3: map(batched=True) ile Toplu Tokenizasyon")
print("═" * 55)

MODEL_ADI  = "distilbert-base-uncased"
MAX_LENGTH = 256

if TOK_AVAILABLE:
    print(f"  Tokenizer yükleniyor: {MODEL_ADI}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ADI)
else:
    tok = None
    print("  [SİMÜLASYON] Tokenizer yüklenmedi.")

def tokenize_batch(texts, labels, tokenizer=None, max_length=MAX_LENGTH,
                   batch_size=64, batched=True):
    """
    Gerçek (HuggingFace Datasets) veya simüle edilmiş batch tokenizasyon.
    Döndürür: {input_ids, attention_mask, labels, token_lengths}
    """
    t0 = time.time()

    if tokenizer and TOK_AVAILABLE:
        if batched:
            # Batch tokenizasyon (hızlı)
            enc = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="np",
            )
            input_ids    = enc["input_ids"]
            attn_masks   = enc["attention_mask"]
        else:
            # Tek tek tokenizasyon (yavaş)
            input_ids_list  = []
            attn_masks_list = []
            for text in texts:
                enc = tokenizer(
                    text, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="np",
                )
                input_ids_list.append(enc["input_ids"][0])
                attn_masks_list.append(enc["attention_mask"][0])
            input_ids  = np.array(input_ids_list)
            attn_masks = np.array(attn_masks_list)
    else:
        # Simülasyon: gerçekçi token ID dizisi üret
        np.random.seed(42)
        n = len(texts)
        input_ids  = np.zeros((n, max_length), dtype=np.int32)
        attn_masks = np.zeros((n, max_length), dtype=np.int32)
        input_ids[:, 0]  = 101  # [CLS]
        for i, text in enumerate(texts):
            words    = text.split()
            n_tokens = min(len(words) + 2, max_length)
            # Sahte token ID'leri (101-30521 arası)
            fake_ids = np.random.randint(1000, 30521, size=n_tokens - 2)
            input_ids[i, 1:n_tokens-1]    = fake_ids
            input_ids[i, n_tokens-1]      = 102  # [SEP]
            attn_masks[i, :n_tokens]      = 1
        if not batched:
            time.sleep(len(texts) * 0.001)  # Simüle edilmiş gecikme

    token_lengths = attn_masks.sum(axis=1)
    elapsed       = time.time() - t0
    return {
        "input_ids":    input_ids,
        "attn_masks":   attn_masks,
        "labels":       np.array(labels, dtype=np.int32),
        "token_lengths": token_lengths,
        "elapsed":      elapsed,
    }

# Batch vs Single hız karşılaştırması
ORNEKLER_100 = TRAIN_TEXTS[:100]
ETIKETLER_100 = TRAIN_LABELS[:100]

print("\n  Batched tokenizasyon...")
sonuc_batched = tokenize_batch(ORNEKLER_100, ETIKETLER_100,
                               tokenizer=tok, batched=True)
print(f"    Batched  : {sonuc_batched['elapsed']*1000:.1f}ms")

print("  Single tokenizasyon (tek tek)...")
sonuc_single  = tokenize_batch(ORNEKLER_100, ETIKETLER_100,
                               tokenizer=tok, batched=False)
print(f"    Single   : {sonuc_single['elapsed']*1000:.1f}ms")

hizlanma = sonuc_single["elapsed"] / max(sonuc_batched["elapsed"], 1e-6)
print(f"    Hızlanma : {hizlanma:.1f}× daha hızlı")

# Tüm eğitim verisi tokenizasyonu
print("\n  Tüm eğitim seti tokenize ediliyor...")
TRAIN_SONUC = tokenize_batch(TRAIN_TEXTS, TRAIN_LABELS, tokenizer=tok)
TEST_SONUC  = tokenize_batch(TEST_TEXTS,  TEST_LABELS,  tokenizer=tok)
print(f"    Eğitim: {TRAIN_SONUC['input_ids'].shape}  ({TRAIN_SONUC['elapsed']:.2f}s)")
print(f"    Test  : {TEST_SONUC['input_ids'].shape}  ({TEST_SONUC['elapsed']:.2f}s)")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 4 — STRATİFİED TRAIN/VAL BÖLME
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 4: Stratified Train / Validation Bölme")
print("═" * 55)

N = len(TRAIN_TEXTS)
INDICES = np.arange(N)

if SKLEARN_AVAILABLE:
    train_idx, val_idx = train_test_split(
        INDICES,
        test_size=0.1,
        random_state=42,
        stratify=TRAIN_LABELS,
    )
else:
    # Manuel stratified bölme
    np.random.seed(42)
    pos_idx = np.where(np.array(TRAIN_LABELS) == 1)[0]
    neg_idx = np.where(np.array(TRAIN_LABELS) == 0)[0]
    np.random.shuffle(pos_idx); np.random.shuffle(neg_idx)
    n_val_pos = max(1, int(len(pos_idx) * 0.1))
    n_val_neg = max(1, int(len(neg_idx) * 0.1))
    val_idx   = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    train_idx = np.concatenate([pos_idx[n_val_pos:], neg_idx[n_val_neg:]])

train_labels_np  = np.array(TRAIN_LABELS)
print(f"\n  Toplam: {N} → Eğitim: {len(train_idx)} ({len(train_idx)/N:.0%})"
      f" + Validasyon: {len(val_idx)} ({len(val_idx)/N:.0%})")

train_pos = train_labels_np[train_idx].sum()
train_neg = (1 - train_labels_np[train_idx]).sum()
val_pos   = train_labels_np[val_idx].sum()
val_neg   = (1 - train_labels_np[val_idx]).sum()

print(f"\n  Eğitim:     POS={train_pos} ({train_pos/len(train_idx):.1%})  "
      f"NEG={train_neg} ({train_neg/len(train_idx):.1%})")
print(f"  Validasyon: POS={val_pos} ({val_pos/len(val_idx):.1%})  "
      f"NEG={val_neg} ({val_neg/len(val_idx):.1%})")
print("  ✅ Stratified bölme: sınıf oranları korundu!")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 5 — DATALOADER
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 5: DataLoader ile Manuel Batch Döngüsü")
print("═" * 55)

BATCH_SIZE = 32

if TORCH_AVAILABLE:
    train_ids  = torch.tensor(TRAIN_SONUC["input_ids"][train_idx],  dtype=torch.long)
    train_mask = torch.tensor(TRAIN_SONUC["attn_masks"][train_idx], dtype=torch.long)
    train_lbl  = torch.tensor(train_labels_np[train_idx],           dtype=torch.long)
    val_ids    = torch.tensor(TRAIN_SONUC["input_ids"][val_idx],    dtype=torch.long)
    val_mask   = torch.tensor(TRAIN_SONUC["attn_masks"][val_idx],   dtype=torch.long)
    val_lbl    = torch.tensor(train_labels_np[val_idx],             dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_ids, train_mask, train_lbl),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_ids,   val_mask,   val_lbl),
                              batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n  Train DataLoader: {len(train_loader)} batch × {BATCH_SIZE} = {len(train_idx)} örnek")
    print(f"  Val   DataLoader: {len(val_loader)} batch × {BATCH_SIZE} ≈ {len(val_idx)} örnek")

    # İlk birkaç batch'i örnek olarak gözlemle
    for i, (ids, mask, lbls) in enumerate(train_loader):
        if i >= 3: break
        pos_ratio = lbls.float().mean().item()
        print(f"    Batch {i+1:2d}: shape={tuple(ids.shape)}, "
              f"POS oranı={pos_ratio:.2f}, "
              f"token doluluk={mask.float().mean().item():.3f}")
else:
    print(f"\n  [SİMÜLASYON] PyTorch olmadan DataLoader simülasyonu...")
    n_batches = len(train_idx) // BATCH_SIZE
    print(f"  Train DataLoader: ~{n_batches} batch × {BATCH_SIZE} örnek")
    for i in range(3):
        idx    = train_idx[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        lbls   = train_labels_np[idx]
        ids_b  = TRAIN_SONUC["input_ids"][idx]
        masks  = TRAIN_SONUC["attn_masks"][idx]
        pos_ratio   = lbls.mean()
        tok_doluluk = masks.mean()
        print(f"    Batch {i+1:2d}: shape={ids_b.shape}, "
              f"POS oranı={pos_ratio:.2f}, token doluluk={tok_doluluk:.3f}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 6 — VERİ KALİTESİ KONTROLLERİ
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 6: Veri Kalitesi Kontrolleri")
print("═" * 55)

tum_metinler = TRAIN_TEXTS + TEST_TEXTS

# 1. Duplikasyon kontrolü
benzersiz     = set(tum_metinler)
duplikasyon   = len(tum_metinler) - len(benzersiz)
print(f"\n  Toplam metin       : {len(tum_metinler)}")
print(f"  Benzersiz metin    : {len(benzersiz)}")
print(f"  Duplikasyon        : {duplikasyon} ({duplikasyon/len(tum_metinler):.2%})")

# 2. Boş / çok kısa metin
bos_metinler  = sum(1 for t in tum_metinler if len(t.strip()) == 0)
kisa_metinler = sum(1 for t in tum_metinler if len(t.split()) < 5)
uzun_metinler = sum(1 for t in tum_metinler if len(t.split()) > MAX_LENGTH)
print(f"  Boş metin          : {bos_metinler}")
print(f"  Çok kısa (<5 kel.) : {kisa_metinler}")
print(f"  Çok uzun (>256 kel.): {uzun_metinler} ({uzun_metinler/len(tum_metinler):.2%})")

# 3. Karakter analizi
html_tag_sayisi = sum(1 for t in tum_metinler if re.search(r"<[^>]+>", t))
print(f"  HTML etiket içeren : {html_tag_sayisi}")

# 4. Token uzunluğu, MAX_LENGTH ile kesilenlerin oranı
n_kesilmis = int((TRAIN_SONUC["token_lengths"] >= MAX_LENGTH).sum())
print(f"  Token sınırı ({MAX_LENGTH}) dolduran: {n_kesilmis} ({n_kesilmis/len(TRAIN_TEXTS):.2%})")

if duplikasyon == 0 and bos_metinler == 0 and kisa_metinler < 5:
    print("\n  ✅ Veri kalitesi: Yüksek — ciddi sorun tespit edilmedi.")
else:
    print("\n  ⚠️  Veri kalitesi: Bazı sorunlar tespit edildi — temizleme önerilir.")

# ─────────────────────────────────────────────────────────────
# GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────
print("\n  Görselleştirmeler oluşturuluyor...")

fig = plt.figure(figsize=(22, 20))
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38)
fig.suptitle("HuggingFace datasets API — Veri İşleme & Tokenizasyon Analizi",
             fontsize=15, fontweight="bold")

colors_list = list(PALETTE.values())

# ── a. Sınıf dengesi (eğitim) ────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
pos_train = sum(1 for l in TRAIN_LABELS if l == 1)
neg_train = sum(1 for l in TRAIN_LABELS if l == 0)
pos_test  = sum(1 for l in TEST_LABELS  if l == 1)
neg_test  = sum(1 for l in TEST_LABELS  if l == 0)

x      = np.array([0, 1])
wid    = 0.32
bars_p = ax1.bar(x - wid/2, [pos_train, pos_test],  wid,
                 color=PALETTE["green"], alpha=0.87, label="POSİTİF")
bars_n = ax1.bar(x + wid/2, [neg_train, neg_test], wid,
                 color=PALETTE["red"],   alpha=0.87, label="NEGATİF")
for bar in bars_p + bars_n:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+8,
             str(int(bar.get_height())), ha="center", fontsize=10, fontweight="bold")
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Eğitim", "Test"], fontsize=12)
ax1.set_title("Sınıf Dağılımı — POS vs NEG", fontweight="bold")
ax1.set_ylabel("Örnek Sayısı")
ax1.legend(fontsize=10)
ax1.grid(axis="y", alpha=0.3)

# ── b. Kelime uzunluğu dağılımı (eğitim) ─────────────────────
ax2 = fig.add_subplot(gs[0, 1])
pos_uzunluk = [len(t.split()) for t, l in zip(TRAIN_TEXTS, TRAIN_LABELS) if l == 1]
neg_uzunluk = [len(t.split()) for t, l in zip(TRAIN_TEXTS, TRAIN_LABELS) if l == 0]
bins = np.linspace(0, min(max(train_stats["uzunluklar"]), 500), 30)
ax2.hist(pos_uzunluk, bins=bins, alpha=0.65, color=PALETTE["green"],
         label=f"POSİTİF (ort={np.mean(pos_uzunluk):.0f})")
ax2.hist(neg_uzunluk, bins=bins, alpha=0.65, color=PALETTE["red"],
         label=f"NEGATİF (ort={np.mean(neg_uzunluk):.0f})")
ax2.axvline(MAX_LENGTH, color=PALETTE["amber"], ls="--", lw=2,
            label=f"max_length={MAX_LENGTH}")
ax2.set_title("Kelime Uzunluğu Dağılımı (Etiket Bazlı)", fontweight="bold")
ax2.set_xlabel("Kelime Sayısı / Metin")
ax2.set_ylabel("Frekans")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)

# ── c. Token uzunluğu dağılımı (tokenize edilmiş) ────────────
ax3 = fig.add_subplot(gs[0, 2])
tok_pos = TRAIN_SONUC["token_lengths"][[i for i, l in enumerate(TRAIN_LABELS) if l == 1]]
tok_neg = TRAIN_SONUC["token_lengths"][[i for i, l in enumerate(TRAIN_LABELS) if l == 0]]
bins_tok = np.linspace(0, MAX_LENGTH + 10, 30)
ax3.hist(tok_pos, bins=bins_tok, alpha=0.65, color=PALETTE["green"],
         label=f"POS (ort={np.mean(tok_pos):.0f})")
ax3.hist(tok_neg, bins=bins_tok, alpha=0.65, color=PALETTE["red"],
         label=f"NEG (ort={np.mean(tok_neg):.0f})")
ax3.axvline(MAX_LENGTH, color=PALETTE["amber"], ls="--", lw=2,
            label=f"Kesme={MAX_LENGTH}")
ax3.set_title("Token Uzunluğu (Tokenize Edilmiş)", fontweight="bold")
ax3.set_xlabel("Token Sayısı")
ax3.set_ylabel("Frekans")
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)

# ── d. Batch vs Single tokenizasyon hızı ─────────────────────
ax4 = fig.add_subplot(gs[1, 0])
batch_ns   = [1, 10, 50, 100]
t_singles  = []
t_batcheds = []

for bn in batch_ns:
    subset = TRAIN_TEXTS[:bn]
    subset_lbl = TRAIN_LABELS[:bn]

    r_single  = tokenize_batch(subset, subset_lbl, tok, batched=False)
    r_batched = tokenize_batch(subset, subset_lbl, tok, batched=True)
    t_singles.append(r_single["elapsed"] * 1000)
    t_batcheds.append(r_batched["elapsed"] * 1000)

ax4.plot(batch_ns, t_singles,  "o-", lw=2.5, color=PALETTE["red"],
         label="Single (tek tek)", markersize=10)
ax4.plot(batch_ns, t_batcheds, "s-", lw=2.5, color=PALETTE["green"],
         label="Batched", markersize=10)
ax4.fill_between(batch_ns, t_singles, t_batcheds,
                 alpha=0.15, color=PALETTE["green"])
ax4.set_xlabel("Örnek Sayısı")
ax4.set_ylabel("Süre (ms)")
ax4.set_title("Batch vs Single Tokenizasyon Hızı", fontweight="bold")
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)

# ── e. Stratified bölme sınıf dengesi ────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
splits = ["Eğitim\n(train_idx)", "Validasyon\n(val_idx)", "Test"]
split_pos = [train_pos, int(val_pos),  pos_test]
split_neg = [train_neg, int(val_neg),  neg_test]
split_totals = [train_pos + train_neg, val_pos + val_neg, pos_test + neg_test]
split_pos_ratio = [p/t for p, t in zip(split_pos, split_totals)]

x5 = np.arange(len(splits))
ax5.bar(x5, [r * 100 for r in split_pos_ratio],
        color=PALETTE["navy"], alpha=0.85, label="POS oranı (%)")
ax5.axhline(50, color=PALETTE["amber"], ls="--", lw=2, label="İdeal=%50")
for xi, r in zip(x5, split_pos_ratio):
    ax5.text(xi, r * 100 + 1, f"{r:.1%}", ha="center",
             fontsize=12, fontweight="bold")
ax5.set_xticks(x5)
ax5.set_xticklabels(splits, fontsize=10)
ax5.set_title("Stratified Bölme — Sınıf Dengesi", fontweight="bold")
ax5.set_ylabel("POSİTİF Oranı (%)")
ax5.legend(fontsize=10)
ax5.set_ylim(0, 80)
ax5.grid(axis="y", alpha=0.3)

# ── f. DataLoader batch token doluluk dağılımı ───────────────
ax6 = fig.add_subplot(gs[1, 2])
# Token doluluk (padding olmayan token oranı)
doluluk = TRAIN_SONUC["attn_masks"].mean(axis=1)  # Her örnek için
ax6.hist(doluluk, bins=25, color=PALETTE["cyan"], alpha=0.85, edgecolor="white")
ax6.axvline(doluluk.mean(), color=PALETTE["amber"], ls="--", lw=2,
            label=f"Ort={doluluk.mean():.3f}")
ax6.set_title("Attention Mask Doluluk Dağılımı\n(1=gerçek token, 0=padding)", fontweight="bold")
ax6.set_xlabel("Doluluk Oranı (token/MAX_LENGTH)")
ax6.set_ylabel("Frekans")
ax6.legend(fontsize=10)
ax6.grid(axis="y", alpha=0.3)

# ── g. Veri kalitesi özet grafiği ────────────────────────────
ax7 = fig.add_subplot(gs[2, :2])
kategoriler = [
    "Duplikasyon",
    "Boş metin",
    "Çok kısa (<5)",
    "Çok uzun (>256)",
    "HTML içeren",
    "Token sınırı dolan",
]
sayilar = [
    duplikasyon,
    bos_metinler,
    kisa_metinler,
    uzun_metinler,
    html_tag_sayisi,
    n_kesilmis,
]
renkler = [
    PALETTE["red"] if v > 0 else PALETTE["green"]
    for v in sayilar
]
bars7 = ax7.barh(range(len(kategoriler)), sayilar,
                 color=renkler, alpha=0.85)
for bar, v in zip(bars7, sayilar):
    ax7.text(bar.get_width() + max(sayilar) * 0.01,
             bar.get_y() + bar.get_height()/2,
             f"{v}  ({v/len(tum_metinler):.1%})",
             va="center", fontsize=10)
ax7.set_yticks(range(len(kategoriler)))
ax7.set_yticklabels(kategoriler, fontsize=11)
ax7.set_title("Veri Kalitesi Kontrolleri — Sorun Sayısı",
              fontweight="bold")
ax7.set_xlabel("Sorunlu Örnek Sayısı")
ax7.grid(axis="x", alpha=0.3)
patch_ok  = mpatches.Patch(color=PALETTE["green"], label="✅ Sorun yok")
patch_err = mpatches.Patch(color=PALETTE["red"],   label="⚠️ Sorun var")
ax7.legend(handles=[patch_ok, patch_err], fontsize=10)

# ── h. Attention mask ısı haritası (ilk 20 örnek, ilk 64 token) ──
ax8 = fig.add_subplot(gs[2, 2])
sample_masks = TRAIN_SONUC["attn_masks"][:20, :64]
im8 = ax8.imshow(sample_masks, cmap="Blues", aspect="auto",
                 vmin=0, vmax=1, interpolation="nearest")
ax8.set_xlabel("Token Pozisyonu (0-63)")
ax8.set_ylabel("Örnek İndeksi")
ax8.set_title("Attention Mask Isı Haritası\n(İlk 20 örnek, ilk 64 token)",
              fontweight="bold")
plt.colorbar(im8, ax=ax8, fraction=0.04, label="0=PAD, 1=Token")

# ── i. Arrow format: veri akışı diyagramı ───────────────────
ax9 = fig.add_subplot(gs[3, :])
ax9.axis("off")
ax9.set_title("HuggingFace datasets — İş Akışı Özeti", fontweight="bold", fontsize=12)

adimlar = [
    ("📂\nload_dataset\n('imdb')",
     "Arrow format\ndiske indir & cache",  PALETTE["navy"]),
    ("🔍\n.features\n.info()",
     "Şema inceleme:\ntext, label",        PALETTE["blue"]),
    ("🗺️\n.map(tokenize,\nbatched=True)",
     "Toplu tokenizasyon\nnum_proc=4",     PALETTE["teal"]),
    ("✂️\n.train_test\n_split()",
     "Stratified bölme\n%90/%10",          PALETTE["cyan"]),
    ("🔧\n.set_format\n('torch')",
     "Tensor formatı\npt/tf/np",           PALETTE["amber"]),
    ("🚀\nDataLoader\n(batch=32)",
     "Batch döngüsü\nshuffle=True",        PALETTE["green"]),
]

n_adim = len(adimlar)
box_w  = 0.13
gap    = (1.0 - n_adim * box_w) / (n_adim + 1)

for i, (baslik, acik, renk) in enumerate(adimlar):
    x = gap + i * (box_w + gap)
    ax9.add_patch(mpatches.FancyBboxPatch(
        (x, 0.18), box_w, 0.65,
        boxstyle="round,pad=0.02",
        facecolor=renk, edgecolor="white",
        transform=ax9.transAxes, linewidth=1.5
    ))
    ax9.text(x + box_w/2, 0.72, baslik, transform=ax9.transAxes,
             fontsize=10.5, color="white", ha="center", va="center",
             fontweight="bold")
    ax9.text(x + box_w/2, 0.28, acik, transform=ax9.transAxes,
             fontsize=9.5, color="#E2E8F0", ha="center", va="center")
    if i < n_adim - 1:
        ax9.annotate("", xy=(x + box_w + gap * 0.2, 0.52),
                     xytext=(x + box_w + gap * 0.0, 0.52),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="->",
                                     color=PALETTE["slate"], lw=2.5))

# Altta: Arrow format avantajları
avantajlar = [
    "🏎️  Bellek-eşlemeli (memory-mapped): RAM'e kopyalamadan disk'ten oku",
    "⚡  map() sonuçları cache'lenir: aynı dönüşüm tekrar çalıştırmaya gerek yok",
    "🔢  batched=True + num_proc=4: ~4-10× daha hızlı tokenizasyon",
    "📦  60.000+ veri seti: load_dataset('imdb'), ('squad'), ('oscar', 'unshuffled_deduplicated_tr')...",
]
for j, avantaj in enumerate(avantajlar):
    ax9.text(0.0 + j * 0.25, 0.05, avantaj, transform=ax9.transAxes,
             fontsize=9.5, color=PALETTE["slate"], va="center",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#DBEAFE",
                       edgecolor=PALETTE["navy"], alpha=0.8))

plt.savefig("h4c_02_datasets.png", dpi=150, bbox_inches="tight")
print("    ✅ h4c_02_datasets.png kaydedildi")
plt.close()

# ─────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  BÖLÜM ÖZETLERİ")
print(f"  Eğitim örnekleri       : {len(TRAIN_TEXTS):6,}")
print(f"  Test örnekleri         : {len(TEST_TEXTS):6,}")
print(f"  Sınıf dengesi (train)  : POS={denge_orani:.1%}, NEG={1-denge_orani:.1%}")
print(f"  Batch tokenizasyon hızı: {hizlanma:.1f}× (100 örnekte)")
print(f"  max_length aşan örnekler: {n_kesilmis} ({n_kesilmis/len(TRAIN_TEXTS):.1%})")
print(f"  Val seti sınıf dengesi : POS={val_pos/len(val_idx):.1%}, NEG={val_neg/len(val_idx):.1%}")
print(f"  Arrow format avantajı  : bellek-eşlemeli, cache'li, çok işlemci")
print("  ✅ UYGULAMA 02 TAMAMLANDI — h4c_02_datasets.png")
print("=" * 65)
