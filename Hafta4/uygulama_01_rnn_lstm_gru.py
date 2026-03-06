"""
=============================================================================
UYGULAMA 01 — Vanilla RNN, LSTM, GRU: Temel Karşılaştırma
=============================================================================
Kapsam:
  - Vanishing Gradient demosu (gradyan normu katman bazında izleme)
  - SimpleRNN vs LSTM vs GRU ablasyon (IMDB duygu analizi)
  - Dizi uzunluğu etkisi analizi (50 / 100 / 200 / 400 token)
  - Gizli durum boyutu ablasyonu (32 / 64 / 128 / 256)
  - return_sequences=True/False davranışı görselleştirme
  - Eğitim süresi ve parametre karşılaştırması

Veri: IMDB Duygu Analizi (25K eğitim, 25K test)
Kurulum: pip install tensorflow numpy matplotlib seaborn
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)
plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

print("=" * 65)
print("  UYGULAMA 01 — Vanilla RNN, LSTM, GRU: Temel Karşılaştırma")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
# 1. VERİ HAZIRLAMA — IMDB
# ═════════════════════════════════════════════════════════════
print("\n[1] IMDB veri seti yükleniyor...")

VOCAB_SIZE = 20000
MAX_LEN    = 200

(X_train_raw, y_train), (X_test_raw, y_test) = keras.datasets.imdb.load_data(
    num_words=VOCAB_SIZE
)

print(f"    Eğitim örnekleri : {len(X_train_raw):,}")
print(f"    Test örnekleri   : {len(X_test_raw):,}")
print(f"    Sınıf dağılımı   : {np.bincount(y_train)} (0=negatif, 1=pozitif)")
print(f"    Dizi uzunlukları : min={min(map(len,X_train_raw))}, "
      f"max={max(map(len,X_train_raw))}, "
      f"ort={np.mean(list(map(len,X_train_raw))):.0f}")

def prepare_data(maxlen=MAX_LEN):
    X_tr = keras.preprocessing.sequence.pad_sequences(
        X_train_raw, maxlen=maxlen, padding="post", truncating="post"
    )
    X_te = keras.preprocessing.sequence.pad_sequences(
        X_test_raw, maxlen=maxlen, padding="post", truncating="post"
    )
    return X_tr, X_te

X_train, X_test = prepare_data(MAX_LEN)

# Küçük validation seti
val_split  = 5000
X_val, X_train_ = X_train[:val_split], X_train[val_split:]
y_val, y_train_ = y_train[:val_split], y_train[val_split:]
print(f"    Eğitim: {len(X_train_):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ═════════════════════════════════════════════════════════════
# 2. VANİSHİNG GRADİENT DEMOSU
# ═════════════════════════════════════════════════════════════
print("\n[2] Vanishing Gradient demosu...")

def build_deep_rnn(rnn_type="simple", n_layers=6, units=64, name="deep_rnn"):
    """Çok katmanlı RNN — gradyan normlarını izlemek için."""
    inp = keras.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(VOCAB_SIZE, 64, mask_zero=True)(inp)

    RNNCell = {"simple": layers.SimpleRNN, "lstm": layers.LSTM, "gru": layers.GRU}[rnn_type]
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        x = RNNCell(units, return_sequences=return_seq,
                    name=f"{rnn_type}_layer_{i+1}")(x)

    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Her RNN türü için gradyan normlarını hesapla
def measure_gradient_norms(model, X_sample, y_sample, n_layers):
    """Bir batch üzerinde her katmanın gradyan normunu hesapla."""
    X_t = tf.constant(X_sample[:128], dtype=tf.int32)
    y_t = tf.constant(y_sample[:128], dtype=tf.float32)

    with tf.GradientTape() as tape:
        preds = model(X_t, training=True)
        loss  = tf.keras.losses.binary_crossentropy(y_t, tf.squeeze(preds))

    grads = tape.gradient(loss, model.trainable_variables)
    # Katman bazında ortalama norm
    layer_norms = {}
    for var, grad in zip(model.trainable_variables, grads):
        if grad is None: continue
        layer_name = var.name.split("/")[0]
        norm_val   = float(tf.norm(grad).numpy())
        if layer_name not in layer_norms:
            layer_norms[layer_name] = []
        layer_norms[layer_name].append(norm_val)

    # Ortalama norm per layer
    return {k: np.mean(v) for k, v in layer_norms.items()}

N_LAYERS  = 5
grad_norms_all = {}

for rnn_type in ["simple", "lstm", "gru"]:
    print(f"    {rnn_type.upper()} için gradyan normları hesaplanıyor...")
    tf.random.set_seed(42)
    m      = build_deep_rnn(rnn_type, n_layers=N_LAYERS, units=64)
    norms  = measure_gradient_norms(m, X_train_, y_train_)
    # Sadece RNN katmanlarını filtrele
    rnn_norms = {k: v for k, v in norms.items() if rnn_type in k.lower()}
    grad_norms_all[rnn_type] = rnn_norms
    for layer, norm in rnn_norms.items():
        status = "⚠️ Vanishing!" if norm < 1e-4 and rnn_type=="simple" else "✅"
        print(f"      {layer:<30}: norm={norm:.2e}  {status}")

# ═════════════════════════════════════════════════════════════
# 3. RNN TİPİ ABLASYONU
# ═════════════════════════════════════════════════════════════
print("\n[3] RNN tipi ablasyonu (SimpleRNN vs LSTM vs GRU)...")

EPOCHS_ABLATION = 8
BATCH_SIZE      = 128

def build_rnn_model(rnn_type="lstm", units=64, embed_dim=64, name="model"):
    inp = keras.Input(shape=(MAX_LEN,))
    x   = layers.Embedding(VOCAB_SIZE, embed_dim, mask_zero=True)(inp)
    RNNCell = {"simple": layers.SimpleRNN, "lstm": layers.LSTM, "gru": layers.GRU}[rnn_type]
    x   = RNNCell(units, dropout=0.2, recurrent_dropout=0.1)(x)
    x   = layers.Dense(32, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m   = keras.Model(inp, out, name=name)
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return m

def quick_train(model, epochs=EPOCHS_ABLATION):
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=4,
                                       restore_best_weights=True, mode="max"),
    ]
    t0 = time.time()
    h  = model.fit(
        X_train_, y_train_,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=BATCH_SIZE,
        callbacks=cbs, verbose=0
    )
    elapsed  = time.time() - t0
    test_res = model.evaluate(X_test, y_test, verbose=0)
    return h.history, test_res, elapsed

rnn_ablation = {}
for rnn_type in ["simple", "lstm", "gru"]:
    print(f"    {rnn_type.upper()} eğitiliyor...")
    tf.random.set_seed(42)
    m      = build_rnn_model(rnn_type=rnn_type, name=f"ablation_{rnn_type}")
    hist, test_res, elapsed = quick_train(m)
    params = m.count_params()
    rnn_ablation[rnn_type.upper()] = {
        "history":  hist,
        "test_acc": test_res[1],
        "test_auc": test_res[2],
        "params":   params,
        "time":     elapsed,
    }
    print(f"      test_acc={test_res[1]:.4f}  test_auc={test_res[2]:.4f}  "
          f"params={params:,}  t={elapsed:.0f}s")

# ═════════════════════════════════════════════════════════════
# 4. DİZİ UZUNLUĞU ETKİSİ
# ═════════════════════════════════════════════════════════════
print("\n[4] Dizi uzunluğu etkisi analizi...")

len_ablation = {}
for maxlen in [50, 100, 200, 400]:
    print(f"    maxlen={maxlen}...")
    X_tr_l, X_te_l = prepare_data(maxlen)
    X_v_l, X_tr_l_ = X_tr_l[:val_split], X_tr_l[val_split:]

    tf.random.set_seed(42)
    inp = keras.Input(shape=(maxlen,))
    x   = layers.Embedding(VOCAB_SIZE, 64, mask_zero=True)(inp)
    x   = layers.LSTM(64, dropout=0.2)(x)
    x   = layers.Dense(1, activation="sigmoid")(x)
    m   = keras.Model(inp, x)
    m.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])

    h = m.fit(
        X_tr_l_, y_train_,
        validation_data=(X_v_l, y_val),
        epochs=EPOCHS_ABLATION, batch_size=BATCH_SIZE, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_auc", patience=4,
                                                  restore_best_weights=True, mode="max")]
    )
    test_res = m.evaluate(X_te_l, y_test, verbose=0)
    len_ablation[maxlen] = {"test_auc": test_res[2], "test_acc": test_res[1], "history": h.history}
    print(f"      test_acc={test_res[1]:.4f}  test_auc={test_res[2]:.4f}")

# ═════════════════════════════════════════════════════════════
# 5. GİZLİ DURUM BOYUTU ABLASYONU
# ═════════════════════════════════════════════════════════════
print("\n[5] Gizli durum boyutu ablasyonu...")

units_ablation = {}
for units in [32, 64, 128, 256]:
    print(f"    units={units}...")
    tf.random.set_seed(42)
    m = build_rnn_model(rnn_type="lstm", units=units, name=f"units_{units}")
    hist, test_res, elapsed = quick_train(m)
    units_ablation[units] = {
        "test_acc": test_res[1],
        "test_auc": test_res[2],
        "params":   m.count_params(),
        "time":     elapsed,
        "history":  hist,
    }
    print(f"      test_acc={test_res[1]:.4f}  params={m.count_params():,}  t={elapsed:.0f}s")

# ═════════════════════════════════════════════════════════════
# 6. RETURN_SEQUENCES DAVRANIŞI
# ═════════════════════════════════════════════════════════════
print("\n[6] return_sequences davranışı inceleniyor...")

tf.random.set_seed(42)
# return_sequences=True modeli
inp      = keras.Input(shape=(20,))
emb      = layers.Embedding(100, 8)(inp)
lstm_seq = layers.LSTM(16, return_sequences=True)(emb)   # (batch, 20, 16)
lstm_fin = layers.LSTM(8, return_sequences=False)(lstm_seq)  # (batch, 8)
out_demo = layers.Dense(1, activation="sigmoid")(lstm_fin)
demo_model = keras.Model(inp, out_demo)

# Ara çıktıları göster
demo_inp      = keras.Input(shape=(20,))
demo_emb      = layers.Embedding(100, 8)(demo_inp)
rs_true_out   = layers.LSTM(16, return_sequences=True, name="rs_true")(demo_emb)
rs_false_out  = layers.LSTM(16, return_sequences=False, name="rs_false")(demo_emb)

m_seq  = keras.Model(demo_inp, rs_true_out)
m_fin  = keras.Model(demo_inp, rs_false_out)

sample_seq = np.random.randint(1, 100, size=(2, 20))
out_seq    = m_seq.predict(sample_seq, verbose=0)
out_fin    = m_fin.predict(sample_seq, verbose=0)

print(f"    return_sequences=True  çıktı şekli : {out_seq.shape}  (batch, timesteps, units)")
print(f"    return_sequences=False çıktı şekli : {out_fin.shape}   (batch, units)")
print("    → True: Her adım çıktısı (yığılmış LSTM için gerekli)")
print("    → False: Sadece son adım (sınıflandırma için yeterli)")

# ═════════════════════════════════════════════════════════════
# 7. GÖRSELLEŞTİRME
# ═════════════════════════════════════════════════════════════
print("\n[7] Görselleştirmeler hazırlanıyor...")

PALETTE = {
    "SIMPLE": "#DC2626",
    "LSTM":   "#2563EB",
    "GRU":    "#059669",
}

fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.35)
fig.suptitle("RNN / LSTM / GRU — Temel Karşılaştırma (IMDB)", fontsize=15, fontweight="bold")

# ── 7a. Vanishing Gradient — RNN ────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for rnn_type, norms in grad_norms_all.items():
    if not norms: continue
    names_g = list(norms.keys())
    vals_g  = list(norms.values())
    color   = {"simple": "#DC2626", "lstm": "#2563EB", "gru": "#059669"}[rnn_type]
    ax1.semilogy(range(len(names_g)), vals_g, "o-", lw=2, color=color,
                 markersize=7, label=rnn_type.upper())
ax1.axhline(1e-4, color="#D97706", ls="--", lw=1.5, label="Vanishing eşiği (1e-4)")
ax1.set_title("Vanishing Gradient — Katman Bazında Gradyan Normu", fontweight="bold")
ax1.set_xlabel("Katman (geri→ileri)"); ax1.set_ylabel("Gradyan Normu (log)")
ax1.legend(); ax1.grid(alpha=0.3)

# ── 7b. RNN tipi ablasyon — val_accuracy ────────────────────
ax2 = fig.add_subplot(gs[0, 2:])
for name, res in rnn_ablation.items():
    ax2.plot(res["history"]["val_accuracy"], lw=2,
             color=PALETTE[name], label=f"{name} (test={res['test_acc']:.4f})")
ax2.set_title("Val Accuracy — RNN Tipi Karşılaştırması", fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy")
ax2.legend(); ax2.grid(alpha=0.3)

# ── 7c. Test AUC karşılaştırma bar ──────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
names_b = list(rnn_ablation.keys())
aucs_b  = [r["test_auc"] for r in rnn_ablation.values()]
bars = ax3.bar(names_b, aucs_b,
               color=[PALETTE[n] for n in names_b], alpha=0.85, edgecolor="white")
for bar, v in zip(bars, aucs_b):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
             f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
ax3.set_ylim(0.85, 0.97); ax3.set_title("Test AUC Karşılaştırması", fontweight="bold")
ax3.grid(axis="y", alpha=0.3)

# ── 7d. Eğitim süresi ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
times_b  = [r["time"] for r in rnn_ablation.values()]
params_b = [r["params"]/1e6 for r in rnn_ablation.values()]
ax4b = ax4.twinx()
bars2 = ax4.bar(names_b, times_b, color=[PALETTE[n] for n in names_b], alpha=0.6)
ax4b.plot(names_b, params_b, "D-", color="#7C3AED", lw=2, markersize=8, label="Parametre (M)")
ax4.set_title("Eğitim Süresi & Parametre Sayısı", fontweight="bold")
ax4.set_ylabel("Süre (sn)", color="#DC2626")
ax4b.set_ylabel("Parametre (M)", color="#7C3AED")
ax4b.legend(loc="upper right"); ax4.grid(axis="y", alpha=0.3)

# ── 7e. Dizi uzunluğu etkisi ─────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
maxlens   = list(len_ablation.keys())
auc_lens  = [r["test_auc"] for r in len_ablation.values()]
acc_lens  = [r["test_acc"] for r in len_ablation.values()]
ax5.plot(maxlens, auc_lens, "o-", lw=2.5, color="#2563EB", markersize=9, label="Test AUC")
ax5.plot(maxlens, acc_lens, "s--", lw=2, color="#059669", markersize=8, label="Test Acc")
for x_l, y_l in zip(maxlens, auc_lens):
    ax5.annotate(f"{y_l:.4f}", (x_l, y_l), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=9)
ax5.set_title("Dizi Uzunluğu Etkisi (LSTM)", fontweight="bold")
ax5.set_xlabel("maxlen (token)"); ax5.legend(); ax5.grid(alpha=0.3)

# ── 7f. Gizli durum boyutu etkisi ────────────────────────────
ax6 = fig.add_subplot(gs[1, 3])
units_list   = list(units_ablation.keys())
auc_units    = [r["test_auc"] for r in units_ablation.values()]
params_units = [r["params"]/1e6 for r in units_ablation.values()]
ax6b = ax6.twinx()
ax6.plot(units_list, auc_units, "o-", lw=2.5, color="#2563EB", markersize=9, label="Test AUC")
ax6b.plot(units_list, params_units, "s--", lw=2, color="#D97706", markersize=8, label="Parametre (M)")
ax6.set_title("Gizli Durum Boyutu Etkisi", fontweight="bold")
ax6.set_xlabel("units"); ax6.set_ylabel("AUC", color="#2563EB")
ax6b.set_ylabel("Parametre (M)", color="#D97706")
ax6.legend(loc="lower right"); ax6b.legend(loc="upper left")
ax6.grid(alpha=0.3)

# ── 7g. Val AUC karşılaştırma ────────────────────────────────
ax7 = fig.add_subplot(gs[2, :2])
for name, res in rnn_ablation.items():
    ax7.plot(res["history"].get("val_auc", []),
             lw=2, color=PALETTE[name], label=name)
ax7.set_title("Val AUC — Tüm RNN Türleri", fontweight="bold")
ax7.set_xlabel("Epoch"); ax7.set_ylabel("Val AUC")
ax7.legend(); ax7.grid(alpha=0.3)

# ── 7h. return_sequences görsel açıklama ─────────────────────
ax8 = fig.add_subplot(gs[2, 2:])
ax8.axis("off")
info_txt = (
    "return_sequences Davranışı:\n\n"
    "return_sequences=True  → şekil: (batch, timesteps, units)\n"
    f"  Örnek çıktı: {out_seq.shape}  — Her adımda gizli durum\n\n"
    "return_sequences=False → şekil: (batch, units)\n"
    f"  Örnek çıktı: {out_fin.shape}    — Sadece son adım\n\n"
    "Kullanım Kuralı:\n"
    "  • Yığılmış LSTM arası: return_sequences=True\n"
    "  • Son LSTM / sınıflandırma: return_sequences=False\n"
    "  • BiLSTM: merge_mode='concat' → 2×units boyut\n"
)
ax8.text(0.05, 0.95, info_txt, transform=ax8.transAxes,
         fontsize=12, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#EFF6FF",
                   edgecolor="#2563EB", linewidth=1.5))
ax8.set_title("return_sequences Açıklaması", fontweight="bold")

plt.savefig("01_rnn_lstm_gru_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 01_rnn_lstm_gru_analiz.png")
plt.close()

# ─── Özet ────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("  RNN TİPİ ABLASYON SONUÇLARI")
print("─" * 55)
print(f"  {'Tür':<10} {'Test Acc':>10} {'Test AUC':>10} {'Parametre':>12} {'Süre':>8}")
print("  " + "─" * 50)
for name, r in rnn_ablation.items():
    print(f"  {name:<10} {r['test_acc']:>10.4f} {r['test_auc']:>10.4f} "
          f"{r['params']:>12,} {r['time']:>7.0f}s")

print("\n" + "=" * 65)
print("  ✅ UYGULAMA 01 TAMAMLANDI")
print("  Çıktı: 01_rnn_lstm_gru_analiz.png")
print("=" * 65)
