"""
=============================================================================
UYGULAMA 04 — Zaman Serisi Tahmini: LSTM + Conv1D Hibrit
=============================================================================
Kapsam:
  - Sentetik çoklu değişkenli zaman serisi (trend + mevsimsellik + gürültü)
  - Pencereli (windowed) veri hazırlama — veri sızıntısına dikkat!
  - Naive baseline (son değeri kopyala) vs LSTM vs Conv1D vs Hibrit
  - Tek adım vs çok adım tahmin karşılaştırması
  - Causal padding ile Conv1D (gelecek bilgisi sızmasını engelle)
  - 30 günlük gelecek tahmini görselleştirmesi
  - MAE, RMSE, MAPE metrik karşılaştırması

Kurulum: pip install tensorflow numpy matplotlib scikit-learn pandas
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)
plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

print("=" * 65)
print("  UYGULAMA 04 — Zaman Serisi Tahmini: LSTM + Conv1D Hibrit")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
# 1. SENTETİK VERİ ÜRETME
# ═════════════════════════════════════════════════════════════
print("\n[1] Çoklu değişkenli sentetik zaman serisi üretiliyor...")

N_DAYS   = 1500
dates    = pd.date_range("2019-01-01", periods=N_DAYS, freq="D")
t        = np.arange(N_DAYS)

# Ana seri: trend + mevsimsellik + haftalık döngü + gürültü
trend      = 0.05 * t
seasonal   = 15 * np.sin(2 * np.pi * t / 365.25)      # yıllık
weekly     = 4  * np.sin(2 * np.pi * t / 7)            # haftalık
noise      = np.random.normal(0, 2.5, N_DAYS)
price      = 50 + trend + seasonal + weekly + noise

# Kovaryant özellikler
volume     = 1000 + 200 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 50, N_DAYS)
sentiment  = np.random.normal(0, 1, N_DAYS)             # duygu endeksi
weekday    = np.array([dates[i].weekday() for i in range(N_DAYS)]) / 6.0

df = pd.DataFrame({
    "date":      dates,
    "price":     price,
    "volume":    volume,
    "sentiment": sentiment,
    "weekday":   weekday,
})
df.set_index("date", inplace=True)

print(f"    Toplam gün    : {N_DAYS}")
print(f"    Fiyat aralığı : {price.min():.1f} — {price.max():.1f}")
print(f"    Özellik sayısı: 4 (price, volume, sentiment, weekday)")

# ─── Temporal Split (ASLA rastgele bölme!) ───────────────────
train_sz  = int(N_DAYS * 0.70)
val_sz    = int(N_DAYS * 0.15)
test_sz   = N_DAYS - train_sz - val_sz

df_train = df.iloc[:train_sz]
df_val   = df.iloc[train_sz:train_sz + val_sz]
df_test  = df.iloc[train_sz + val_sz:]

print(f"    Eğitim : {len(df_train)} gün  |  Val: {len(df_val)} gün  |  Test: {len(df_test)} gün")

# ─── Ölçekleme — SADECE train fit! ───────────────────────────
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(df_train.values)   # fit+transform
val_scaled   = scaler.transform(df_val.values)         # sadece transform
test_scaled  = scaler.transform(df_test.values)        # sadece transform

# ─── Sadece hedef (price) için scaler — ters dönüşüm için ────
price_scaler = MinMaxScaler()
price_scaler.fit(df_train[["price"]].values)

# ═════════════════════════════════════════════════════════════
# 2. PENCERELİ DİZİ HAZIRLAMA
# ═════════════════════════════════════════════════════════════
print("\n[2] Pencereli diziler oluşturuluyor...")

N_IN    = 60    # 60 günlük geçmiş
N_OUT   = 1     # 1 günlük tahmin (tek adım)
N_FEAT  = 4     # özellik sayısı

def create_sequences(data, n_in, n_out, target_col=0):
    """
    Pencereli girdi-hedef çiftleri oluştur.
    data   : (N, n_features)
    Çıkış  : X (N_seq, n_in, n_features), y (N_seq, n_out)
    """
    X, y = [], []
    for i in range(len(data) - n_in - n_out + 1):
        X.append(data[i : i + n_in])                         # tüm özellikler
        y.append(data[i + n_in : i + n_in + n_out, target_col])  # sadece fiyat
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = create_sequences(train_scaled, N_IN, N_OUT)
X_val,   y_val   = create_sequences(val_scaled,   N_IN, N_OUT)
X_test,  y_test  = create_sequences(test_scaled,  N_IN, N_OUT)

print(f"    X_train şekli: {X_train.shape}  y_train: {y_train.shape}")
print(f"    X_val şekli  : {X_val.shape}")
print(f"    X_test şekli : {X_test.shape}")

# Çok adımlı tahmin için (30 gün ilerisi)
N_OUT_MULTI = 30
X_tr_m, y_tr_m = create_sequences(train_scaled, N_IN, N_OUT_MULTI)
X_va_m, y_va_m = create_sequences(val_scaled,   N_IN, N_OUT_MULTI)
X_te_m, y_te_m = create_sequences(test_scaled,  N_IN, N_OUT_MULTI)
print(f"    Çok adımlı — y_train şekli: {y_tr_m.shape}")

# ═════════════════════════════════════════════════════════════
# 3. METRIK FONKSİYONLARI
# ═════════════════════════════════════════════════════════════
def inverse_price(y_scaled):
    """Ölçeklenmiş fiyatı gerçek ölçeğe çevir."""
    return price_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

def calc_metrics(y_true_s, y_pred_s, label=""):
    """Tahmin kalitesini hesapla (gerçek ölçekte)."""
    y_true = inverse_price(y_true_s)
    y_pred = inverse_price(y_pred_s)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    if label:
        print(f"    {label:<22}: MAE={mae:6.3f}  RMSE={rmse:6.3f}  MAPE={mape:5.2f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}

# ═════════════════════════════════════════════════════════════
# 4. NAIVE BASELINE
# ═════════════════════════════════════════════════════════════
print("\n[3] Naive baseline hesaplanıyor...")

# Son gözlemi kopyala
y_naive = X_test[:, -1, 0]  # son gün fiyatı (ölçeklenmiş)
naive_metrics = calc_metrics(y_test.flatten(), y_naive, "Naive (son değer)")

# ═════════════════════════════════════════════════════════════
# 5. MODEL TANIMLAMALARI
# ═════════════════════════════════════════════════════════════
print("\n[4] Modeller tanımlanıyor...")

BATCH_SIZE = 64
EPOCHS     = 40

def compile_and_train(model, X_tr, y_tr, X_v, y_v, epochs=EPOCHS, verbose=0):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="huber",             # outlier'lara daha dayanıklı
        metrics=["mae"],
    )
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                       restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                                           patience=5, min_lr=1e-7),
    ]
    t0  = time.time()
    h   = model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
                    epochs=epochs, batch_size=BATCH_SIZE,
                    callbacks=cbs, verbose=verbose)
    elapsed = time.time() - t0
    return h.history, elapsed

# ─── Model A: Basit LSTM ─────────────────────────────────────
def build_lstm_simple(n_in, n_feat, n_out=1, name="lstm_simple"):
    inp = keras.Input(shape=(n_in, n_feat))
    x   = layers.LSTM(128, return_sequences=True, dropout=0.2)(inp)
    x   = layers.LSTM(64,  return_sequences=False, dropout=0.2)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_out)(x)
    return keras.Model(inp, out, name=name)

# ─── Model B: Conv1D (Causal) ─────────────────────────────────
def build_conv1d(n_in, n_feat, n_out=1, name="conv1d"):
    inp = keras.Input(shape=(n_in, n_feat))
    x   = layers.Conv1D(64, kernel_size=5, padding="causal",
                        activation="relu", dilation_rate=1)(inp)
    x   = layers.Conv1D(32, kernel_size=3, padding="causal",
                        activation="relu", dilation_rate=2)(x)
    x   = layers.Conv1D(16, kernel_size=3, padding="causal",
                        activation="relu", dilation_rate=4)(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_out)(x)
    return keras.Model(inp, out, name=name)

# ─── Model C: Hibrit Conv1D + LSTM ────────────────────────────
def build_conv_lstm(n_in, n_feat, n_out=1, name="conv_lstm"):
    inp = keras.Input(shape=(n_in, n_feat))
    # Conv1D: yerel zaman örüntüleri (causal → gelecek sızmaz)
    x   = layers.Conv1D(64, kernel_size=5, padding="causal",
                        activation="relu")(inp)
    x   = layers.Conv1D(32, kernel_size=3, padding="causal",
                        activation="relu", dilation_rate=2)(x)
    x   = layers.Dropout(0.1)(x)
    # LSTM: uzun süreli bağımlılık
    x   = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
    x   = layers.LSTM(64,  return_sequences=False, dropout=0.2)(x)
    x   = layers.Dense(32, activation="relu")(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(n_out)(x)
    return keras.Model(inp, out, name=name)

# ─── Model D: BiLSTM ─────────────────────────────────────────
def build_bilstm(n_in, n_feat, n_out=1, name="bilstm"):
    inp = keras.Input(shape=(n_in, n_feat))
    x   = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2))(inp)
    x   = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False, dropout=0.2))(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_out)(x)
    return keras.Model(inp, out, name=name)

# ═════════════════════════════════════════════════════════════
# 6. TEK ADIM EĞİTİM
# ═════════════════════════════════════════════════════════════
print("\n[5] Tek adımlı modeller eğitiliyor (1 gün ilerisi)...")

single_results = {}

for mname, mfunc in [
    ("Naive",       None),
    ("LSTM Basit",  build_lstm_simple),
    ("Conv1D",      build_conv1d),
    ("Conv+LSTM",   build_conv_lstm),
    ("BiLSTM",      build_bilstm),
]:
    if mfunc is None:
        # Naive zaten hesaplandı
        single_results["Naive"] = {
            "metrics": naive_metrics, "history": None,
            "y_pred": y_naive, "elapsed": 0,
        }
        continue

    print(f"    [{mname}] eğitiliyor...")
    tf.random.set_seed(42)
    m = mfunc(N_IN, N_FEAT, n_out=1, name=mname.replace(" ", "_"))
    hist, elapsed = compile_and_train(m, X_train, y_train, X_val, y_val, verbose=0)

    y_pred_s = m.predict(X_test, verbose=0).flatten()
    metrics  = calc_metrics(y_test.flatten(), y_pred_s, mname)
    single_results[mname] = {
        "metrics": metrics, "history": hist,
        "y_pred": y_pred_s, "elapsed": elapsed,
        "params": m.count_params(),
    }

# ═════════════════════════════════════════════════════════════
# 7. ÇOK ADIMLI EĞİTİM (30 gün)
# ═════════════════════════════════════════════════════════════
print("\n[6] Çok adımlı model eğitiliyor (30 gün ilerisi)...")

tf.random.set_seed(42)
m_multi = build_conv_lstm(N_IN, N_FEAT, n_out=N_OUT_MULTI, name="conv_lstm_multi")
hist_m, elapsed_m = compile_and_train(
    m_multi, X_tr_m, y_tr_m, X_va_m, y_va_m, epochs=EPOCHS, verbose=1
)
y_pred_multi = m_multi.predict(X_te_m, verbose=0)
print(f"\n    Çok adımlı tahmin şekli: {y_pred_multi.shape}")

# Her adım için RMSE
step_rmse = []
for step in range(N_OUT_MULTI):
    y_t = inverse_price(y_te_m[:, step])
    y_p = inverse_price(y_pred_multi[:, step])
    step_rmse.append(np.sqrt(mean_squared_error(y_t, y_p)))
print(f"    1. gün RMSE: {step_rmse[0]:.3f}  |  30. gün RMSE: {step_rmse[-1]:.3f}")

# ═════════════════════════════════════════════════════════════
# 8. 30 GÜNLÜK GELECEK TAHMİNİ
# ═════════════════════════════════════════════════════════════
print("\n[7] 30 günlük gelecek tahmini yapılıyor...")

# Son 60 günü al → 30 gün ileriye tahmin
last_window = test_scaled[-N_IN:].reshape(1, N_IN, N_FEAT)
future_pred_s = m_multi.predict(last_window, verbose=0)[0]   # (30,)
future_pred   = inverse_price(future_pred_s)
future_dates  = pd.date_range(df_test.index[-1] + pd.Timedelta(days=1),
                               periods=N_OUT_MULTI, freq="D")

print(f"    Son gerçek fiyat    : {df_test['price'].iloc[-1]:.2f}")
print(f"    30 gün sonra tahmin : {future_pred[-1]:.2f}")
print(f"    Tahmin aralığı      : {future_pred.min():.2f} — {future_pred.max():.2f}")

# ═════════════════════════════════════════════════════════════
# 9. GÖRSELLEŞTİRME
# ═════════════════════════════════════════════════════════════
print("\n[8] Görselleştirmeler hazırlanıyor...")

PALETTE = {
    "Naive":     "#6B7280",
    "LSTM Basit":"#7C3AED",
    "Conv1D":    "#0891B2",
    "Conv+LSTM": "#2563EB",
    "BiLSTM":    "#059669",
}

# Gerçek fiyatlar (test seti)
y_test_real = inverse_price(y_test.flatten())
test_dates_plot = df_test.index[N_IN - 1 : N_IN - 1 + len(y_test_real)]

fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)
fig.suptitle("Zaman Serisi Tahmini — LSTM + Conv1D Hibrit", fontsize=15, fontweight="bold")

# ── 9a. Ham veri + train/val/test ayrımı ─────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_train.index, df_train["price"], color="#6B7280", lw=1, label="Eğitim")
ax1.plot(df_val.index,   df_val["price"],   color="#D97706", lw=1.5, label="Val")
ax1.plot(df_test.index,  df_test["price"],  color="#2563EB", lw=1.5, label="Test")
ax1.axvline(df_train.index[-1], color="#D97706", ls="--", lw=1.5, alpha=0.7)
ax1.axvline(df_val.index[-1],   color="#2563EB", ls="--", lw=1.5, alpha=0.7)
ax1.set_title("Ham Veri — Temporal Split (Eğitim / Val / Test)", fontweight="bold")
ax1.set_ylabel("Fiyat"); ax1.legend(); ax1.grid(alpha=0.3)

# ── 9b. Tek adımlı tahmin karşılaştırma ──────────────────────
ax2 = fig.add_subplot(gs[1, :2])
ax2.plot(test_dates_plot, y_test_real,
         color="black", lw=2, label="Gerçek", zorder=5)
for mname, res in single_results.items():
    if mname == "Naive": continue
    y_pred_real = inverse_price(res["y_pred"])
    n_show = min(len(test_dates_plot), len(y_pred_real))
    ax2.plot(test_dates_plot[:n_show], y_pred_real[:n_show],
             color=PALETTE.get(mname, "gray"), lw=1.5,
             alpha=0.8, label=f"{mname}")
ax2.set_title("Tek Adımlı Tahmin — Test Seti", fontweight="bold")
ax2.set_ylabel("Fiyat"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

# ── 9c. Metrik karşılaştırma ─────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
model_names  = list(single_results.keys())
rmse_vals    = [r["metrics"]["rmse"] for r in single_results.values()]
mae_vals     = [r["metrics"]["mae"]  for r in single_results.values()]
x_m = np.arange(len(model_names))
w_m = 0.38
bars3a = ax3.bar(x_m - w_m/2, rmse_vals, w_m,
                  color=[PALETTE.get(n, "#6B7280") for n in model_names],
                  alpha=0.85, label="RMSE")
bars3b = ax3.bar(x_m + w_m/2, mae_vals,  w_m,
                  color=[PALETTE.get(n, "#6B7280") for n in model_names],
                  alpha=0.5, label="MAE")
for bar, v in zip(bars3a, rmse_vals):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
             f"{v:.2f}", ha="center", fontsize=8)
ax3.set_xticks(x_m)
ax3.set_xticklabels([n[:9] for n in model_names], fontsize=8, rotation=15)
ax3.set_title("RMSE & MAE Karşılaştırması", fontweight="bold")
ax3.legend(); ax3.grid(axis="y", alpha=0.3)

# ── 9d. Eğitim loss eğrisi ────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
for mname, res in single_results.items():
    if res["history"] is None: continue
    ax4.plot(res["history"]["val_loss"], lw=2,
             color=PALETTE.get(mname, "gray"), label=mname)
ax4.set_title("Val Loss Karşılaştırması", fontweight="bold")
ax4.set_xlabel("Epoch"); ax4.set_ylabel("Huber Loss")
ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

# ── 9e. Adım bazında RMSE (30 gün) ────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(range(1, N_OUT_MULTI + 1), step_rmse,
         "o-", color="#2563EB", lw=2.5, markersize=5)
ax5.fill_between(range(1, N_OUT_MULTI + 1), step_rmse, alpha=0.15, color="#2563EB")
ax5.set_title("Çok Adımlı RMSE — Adım Başına Hata", fontweight="bold")
ax5.set_xlabel("Tahmin Adımı (gün)"); ax5.set_ylabel("RMSE")
ax5.grid(alpha=0.3)

# ── 9f. 30 Günlük Gelecek Tahmini ────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
# Son 90 gün gerçek + 30 gün tahmin
last_90 = df_test.iloc[-90:]
ax6.plot(last_90.index, last_90["price"], color="#2563EB", lw=2, label="Gerçek (son 90 gün)")
ax6.plot(future_dates, future_pred, color="#DC2626", lw=2.5, ls="--",
         marker="o", markersize=4, label="30 Günlük Tahmin")
# Belirsizlik aralığı (basit: ±1 RMSE)
best_rmse = min(r["metrics"]["rmse"] for r in single_results.values() if r["history"])
ax6.fill_between(future_dates,
                  future_pred - best_rmse,
                  future_pred + best_rmse,
                  alpha=0.2, color="#DC2626", label=f"±RMSE={best_rmse:.2f}")
ax6.axvline(df_test.index[-1], color="#6B7280", ls=":", lw=1.5, label="Bugün")
ax6.set_title("30 Günlük Gelecek Tahmini (Conv+LSTM)", fontweight="bold")
ax6.legend(fontsize=8); ax6.grid(alpha=0.3)

plt.savefig("04_zaman_serisi_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ Kaydedildi: 04_zaman_serisi_analiz.png")
plt.close()

# ─── Kapsamlı özet tablo ─────────────────────────────────────
print("\n" + "=" * 65)
print("  SONUÇLAR — Tek Adımlı Tahmin (1 Gün İlerisi)")
print("─" * 65)
print(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Parametre':>12}")
print("  " + "─" * 60)
for mname, res in sorted(single_results.items(), key=lambda x: x[1]["metrics"]["rmse"]):
    params_str = f"{res.get('params', 0):,}" if res.get("params") else "N/A"
    print(f"  {mname:<22} {res['metrics']['mae']:>8.3f} "
          f"{res['metrics']['rmse']:>8.3f} {res['metrics']['mape']:>7.2f}% "
          f"{params_str:>12}")

print(f"\n  Çok Adımlı (30 gün) — Conv+LSTM:")
print(f"    1. gün RMSE : {step_rmse[0]:.3f}")
print(f"   15. gün RMSE : {step_rmse[14]:.3f}")
print(f"   30. gün RMSE : {step_rmse[-1]:.3f}")

print("\n  ✅ UYGULAMA 04 TAMAMLANDI")
print("  Çıktı: 04_zaman_serisi_analiz.png")
print("=" * 65)
