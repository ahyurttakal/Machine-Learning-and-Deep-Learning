"""
=============================================================================
HAFTA 5 CUMARTESİ — UYGULAMA 03
Flask & FastAPI ile ML Model Deployment — REST API
=============================================================================
Kapsam:
  - sklearn modeli eğit → joblib ile kaydet (Iris + RandomForest)
  - Flask: /health, /predict, /predict/batch endpoint'leri
  - Flask: CORS, hata yönetimi, istek doğrulama, loglama
  - FastAPI: Pydantic şema, async endpoint, Swagger otomatik
  - Flask vs FastAPI yan yana performans karşılaştırması
  - Gunicorn / Uvicorn multi-worker production konfigürasyonu
  - API test senaryoları: başarılı, hatalı girdi, edge case
  - Yanıt süresi & throughput analizi
  - Kapsamlı görselleştirme (8 panel)

Kurulum:
  pip install flask flask-cors fastapi uvicorn pydantic
  pip install scikit-learn joblib numpy requests httpx
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import json
import threading
import warnings
warnings.filterwarnings("ignore")

# ── Bağımlılık kontrolü ───────────────────────────────────────────────────
try:
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from flask import Flask, request, jsonify, abort
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

print("=" * 65)
print("  HAFTA 5 CUMARTESİ — UYGULAMA 03")
print("  Flask & FastAPI ile ML Model Deployment")
print("=" * 65)
print(f"  scikit-learn : {'✅' if SKLEARN_AVAILABLE else '❌  pip install scikit-learn'}")
print(f"  Flask        : {'✅' if FLASK_AVAILABLE else '❌  pip install flask flask-cors'}")
print(f"  FastAPI      : {'✅' if FASTAPI_AVAILABLE else '❌  pip install fastapi uvicorn pydantic'}")
print(f"  requests     : {'✅' if REQUESTS_AVAILABLE else '❌  pip install requests'}")
print()

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 1: MODEL EĞİTİMİ & KAYIT
# ─────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("  BÖLÜM 1: Iris Modeli Eğitimi & Kayıt")
print("─" * 65)

SINIF_ADLARI = ["setosa", "versicolor", "virginica"]
MODEL_DOSYA  = "/tmp/iris_model.pkl"
SCALER_DOSYA = "/tmp/iris_scaler.pkl"

if SKLEARN_AVAILABLE:
    iris      = load_iris()
    X, y      = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    dogruluk = accuracy_score(y_test, model.predict(X_test_s))

    joblib.dump(model,  MODEL_DOSYA)
    joblib.dump(scaler, SCALER_DOSYA)

    print(f"  Veri seti   : Iris (150 örnek, 4 özellik, 3 sınıf)")
    print(f"  Model       : RandomForest (n_estimators=100, max_depth=5)")
    print(f"  Doğruluk    : {dogruluk:.4f}  ({dogruluk*100:.1f}%)")
    print(f"  Kayıt       : {MODEL_DOSYA}")
    print(f"  Scaler      : {SCALER_DOSYA}")
else:
    print("  [SIM] Iris modeli — RandomForest, doğruluk: 0.9667")
    print(f"  [SIM] Modeli simüle ediyoruz, gerçek dosya oluşturulmadı.")

    class SimModel:
        def predict(self, X):
            return np.array([int(np.sum(row) % 3) for row in X])
        def predict_proba(self, X):
            n = len(X)
            proba = np.random.dirichlet([5, 2, 1], n)
            return proba

    class SimScaler:
        def transform(self, X):
            return (np.array(X) - 5.8) / 0.83

    model  = SimModel()
    scaler = SimScaler()
    dogruluk = 0.9667

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 2: FLASK UYGULAMASI (TAM KOD)
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 2: Flask REST API Tanımı")
print("─" * 65)

FLASK_KODU = '''
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import joblib, numpy as np, logging, time
from functools import wraps

# ── Uygulama & CORS ──────────────────────────────────────────────────────
app    = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "https://myapp.com"])

# ── Loglama ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Model (başlangıçta bir kez yükle) ────────────────────────────────────
model       = joblib.load("iris_model.pkl")
scaler      = joblib.load("iris_scaler.pkl")
SINIF_ADLARI = ["setosa", "versicolor", "virginica"]
MODEL_META  = {"versiyon": "1.0", "dogruluk": 0.9667, "algoritma": "RandomForest"}

# ── İstek süresi dekoratörü ───────────────────────────────────────────────
def sure_olc(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        t0     = time.perf_counter()
        yanit  = f(*args, **kwargs)
        sure   = (time.perf_counter() - t0) * 1000
        logger.info(f"{request.method} {request.path}  {sure:.1f}ms")
        return yanit
    return wrapper

# ── /health ───────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
@sure_olc
def health():
    return jsonify({
        "durum":    "ok",
        "model":    MODEL_META["versiyon"],
        "dogruluk": MODEL_META["dogruluk"],
    })

# ── /model/info ───────────────────────────────────────────────────────────
@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "model":      MODEL_META,
        "ozellikler": ["sepal_length","sepal_width","petal_length","petal_width"],
        "siniflar":   SINIF_ADLARI,
    })

# ── /predict (tekli) ──────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
@sure_olc
def predict():
    veri = request.get_json(force=True, silent=True)
    if veri is None:
        abort(400, description="JSON gövdesi gerekli")
    if "features" not in veri:
        abort(400, description="'features' alanı zorunlu")

    features = veri["features"]
    if not isinstance(features, list) or len(features) != 4:
        abort(422, description="'features' 4 float içermeli")
    if any(not isinstance(v, (int, float)) for v in features):
        abort(422, description="Tüm değerler sayısal olmalı")

    try:
        X       = np.array(features, dtype=float).reshape(1, -1)
        X_s     = scaler.transform(X)
        tahmin  = int(model.predict(X_s)[0])
        proba   = model.predict_proba(X_s)[0].tolist()
        return jsonify({
            "tahmin":      tahmin,
            "sinif_adi":   SINIF_ADLARI[tahmin],
            "olasiliklar": {k: round(v, 4)
                            for k, v in zip(SINIF_ADLARI, proba)},
            "versiyon":    MODEL_META["versiyon"],
        })
    except Exception as e:
        logger.error(f"Tahmin hatası: {e}")
        abort(500, description=f"Model hatası: {str(e)}")

# ── /predict/batch (toplu) ────────────────────────────────────────────────
@app.route("/predict/batch", methods=["POST"])
@sure_olc
def predict_batch():
    veri = request.get_json(force=True, silent=True)
    if veri is None or "samples" not in veri:
        abort(400, description="'samples' alanı zorunlu")

    samples = veri["samples"]
    if not isinstance(samples, list) or len(samples) > 100:
        abort(422, description="samples: liste, maks 100 örnek")

    try:
        X   = np.array(samples, dtype=float)
        X_s = scaler.transform(X)
        tahminler = model.predict(X_s).tolist()
        probalar  = model.predict_proba(X_s).tolist()
        sonuclar  = [
            {"tahmin": t, "sinif_adi": SINIF_ADLARI[t],
             "olasilik_max": round(max(p), 4)}
            for t, p in zip(tahminler, probalar)
        ]
        return jsonify({"sonuclar": sonuclar, "sayi": len(sonuclar)})
    except Exception as e:
        abort(500, description=str(e))

# ── Hata yöneticileri ─────────────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"hata": "Geçersiz istek", "detay": str(e)}), 400

@app.errorhandler(422)
def unprocessable(e):
    return jsonify({"hata": "İşlenemeyen varlık", "detay": str(e)}), 422

@app.errorhandler(500)
def server_error(e):
    return jsonify({"hata": "Sunucu hatası", "detay": str(e)}), 500

if __name__ == "__main__":
    # Geliştirme: debug=True
    # Production: gunicorn -w 4 -b 0.0.0.0:5000 app_flask:app
    app.run(host="0.0.0.0", port=5000, debug=False)
'''
print("  Flask kodu tanımlandı (gerçek çalıştırma için app_flask.py olarak kaydet)")
print(f"  Endpoint'ler: /health  /model/info  /predict  /predict/batch")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 3: FASTAPI UYGULAMASI (TAM KOD)
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 3: FastAPI REST API Tanımı")
print("─" * 65)

FASTAPI_KODU = '''
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import numpy as np, joblib, time, logging
from typing import Optional

logger = logging.getLogger("uvicorn")

# ── FastAPI Uygulaması ────────────────────────────────────────────────────
app = FastAPI(
    title     = "Iris ML API",
    version   = "1.0",
    description = "RandomForest ile Iris sınıflandırma",
    docs_url  = "/docs",      # Swagger UI
    redoc_url = "/redoc",     # ReDoc
)

# ── CORS ─────────────────────────────────────────────────────────────────
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Middleware: istek loglama ─────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0     = time.perf_counter()
    yanit  = await call_next(request)
    sure   = (time.perf_counter() - t0) * 1000
    logger.info(f"{request.method} {request.url.path}  {sure:.1f}ms")
    return yanit

# ── Model Yükleme ──────────────────────────────────────────────────────
model        = joblib.load("iris_model.pkl")
scaler       = joblib.load("iris_scaler.pkl")
SINIF_ADLARI = ["setosa", "versicolor", "virginica"]

# ── Pydantic Şemalar ──────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="[sepal_length, sepal_width, petal_length, petal_width]"
    )
    model_version: Optional[str] = Field(default="v1")

    @validator("features")
    def no_nan_inf(cls, v):
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError("NaN veya Inf değer kabul edilmez")
        if any(x < 0 or x > 30 for x in v):
            raise ValueError("Değerler 0-30 aralığında olmalı")
        return v

class PredictResponse(BaseModel):
    tahmin:      int
    sinif_adi:   str
    olasiliklar: dict[str, float]
    versiyon:    str

class BatchRequest(BaseModel):
    samples: list[list[float]] = Field(..., max_length=100)

class BatchResponse(BaseModel):
    sonuclar: list[dict]
    sayi:     int

# ── Endpoint'ler ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"durum": "ok", "model": "v1.0", "dogruluk": 0.9667}

@app.get("/model/info")
async def model_info():
    return {
        "algoritma":  "RandomForestClassifier",
        "ozellikler": ["sepal_length","sepal_width","petal_length","petal_width"],
        "siniflar":   SINIF_ADLARI,
        "dogruluk":   0.9667,
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        X      = np.array(req.features).reshape(1, -1)
        X_s    = scaler.transform(X)
        t      = int(model.predict(X_s)[0])
        proba  = model.predict_proba(X_s)[0]
        return PredictResponse(
            tahmin      = t,
            sinif_adi   = SINIF_ADLARI[t],
            olasiliklar = {k: round(float(v), 4)
                           for k, v in zip(SINIF_ADLARI, proba)},
            versiyon    = "1.0",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest):
    try:
        X   = np.array(req.samples)
        X_s = scaler.transform(X)
        tahminler = model.predict(X_s).tolist()
        probalar  = model.predict_proba(X_s).tolist()
        return BatchResponse(
            sonuclar=[
                {"tahmin": t, "sinif_adi": SINIF_ADLARI[t],
                 "olasilik_max": round(max(p), 4)}
                for t, p in zip(tahminler, probalar)
            ],
            sayi=len(tahminler),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Başlatma ──────────────────────────────────────────────────────────────
# Geliştirme: uvicorn app_fastapi:app --reload --port 8000
# Production:  uvicorn app_fastapi:app --host 0.0.0.0 --workers 4
'''
print("  FastAPI kodu tanımlandı (gerçek çalıştırma için app_fastapi.py olarak kaydet)")
print(f"  Swagger UI: http://localhost:8000/docs")
print(f"  ReDoc     : http://localhost:8000/redoc")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 4: API TEST SENARYOLARİ (IN-PROCESS)
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 4: API Test Senaryoları (In-Process Simülasyon)")
print("─" * 65)

class APISunucu:
    """Flask/FastAPI uç noktalarını doğrudan test etmek için in-process simülatör."""

    def __init__(self, model, scaler, sinif_adlari):
        self.model       = model
        self.scaler      = scaler
        self.sinif_adlari = sinif_adlari
        self.istek_sayaci = 0
        self.hata_sayaci  = 0

    def health(self):
        return {"durum": "ok", "model": "v1.0", "dogruluk": dogruluk}, 200

    def predict(self, veri):
        self.istek_sayaci += 1
        t0 = time.perf_counter()

        # Doğrulama
        if not isinstance(veri, dict):
            self.hata_sayaci += 1
            return {"hata": "JSON gövdesi gerekli"}, 400
        if "features" not in veri:
            self.hata_sayaci += 1
            return {"hata": "'features' alanı zorunlu"}, 400
        features = veri["features"]
        if not isinstance(features, list) or len(features) != 4:
            self.hata_sayaci += 1
            return {"hata": "'features' 4 float içermeli"}, 422
        if any(not isinstance(v, (int, float)) for v in features):
            self.hata_sayaci += 1
            return {"hata": "Tüm değerler sayısal olmalı"}, 422

        # Tahmin
        X      = np.array(features, dtype=float).reshape(1, -1)
        X_s    = self.scaler.transform(X)
        tahmin = int(self.model.predict(X_s)[0])
        proba  = self.model.predict_proba(X_s)[0]
        sure   = (time.perf_counter() - t0) * 1000

        return {
            "tahmin":      tahmin,
            "sinif_adi":   self.sinif_adlari[tahmin],
            "olasiliklar": {k: round(float(v), 4)
                            for k, v in zip(self.sinif_adlari, proba)},
            "versiyon":    "1.0",
            "_sure_ms":    round(sure, 3),
        }, 200

    def predict_batch(self, veri):
        self.istek_sayaci += 1
        t0 = time.perf_counter()
        if "samples" not in veri:
            return {"hata": "'samples' zorunlu"}, 400
        samples = veri["samples"]
        if len(samples) > 100:
            return {"hata": "Maks 100 örnek"}, 422
        X      = np.array(samples, dtype=float)
        X_s    = self.scaler.transform(X)
        t_arr  = self.model.predict(X_s).tolist()
        p_arr  = self.model.predict_proba(X_s).tolist()
        sure   = (time.perf_counter() - t0) * 1000
        return {
            "sonuclar": [
                {"tahmin": t, "sinif_adi": self.sinif_adlari[t],
                 "olasilik_max": round(max(p), 4)}
                for t, p in zip(t_arr, p_arr)
            ],
            "sayi":     len(t_arr),
            "_sure_ms": round(sure, 3),
        }, 200


api = APISunucu(model, scaler, SINIF_ADLARI)

# Test senaryoları
SENARYOLAR = [
    # (açıklama, istek, beklenen_kod)
    ("Sağlık kontrolü",         None,                                   200),
    ("Geçerli setosa",          {"features": [5.1, 3.5, 1.4, 0.2]},   200),
    ("Geçerli versicolor",      {"features": [6.3, 2.3, 4.4, 1.3]},   200),
    ("Geçerli virginica",       {"features": [6.5, 3.0, 5.2, 2.0]},   200),
    ("Eksik alan",              {"degerler": [5.1, 3.5, 1.4, 0.2]},   400),
    ("Yanlış boyut (3 eleman)", {"features": [5.1, 3.5, 1.4]},        422),
    ("Metin içeriyor",          {"features": ["a", 3.5, 1.4, 0.2]},   422),
    ("Toplu tahmin (5 örnek)",  {
        "samples": [
            [5.1, 3.5, 1.4, 0.2], [6.3, 2.3, 4.4, 1.3],
            [6.5, 3.0, 5.2, 2.0], [4.9, 3.0, 1.4, 0.2],
            [5.8, 2.7, 5.1, 1.9],
        ]
    }, 200),
]

print(f"  {'#':>3}  {'Test Adı':<35} {'HTTP':>5}  {'Beklenen':>8}  {'Sonuç'}")
print("  " + "-" * 72)

test_sonuclari = []
for i, (aciklama, istek, beklenen) in enumerate(SENARYOLAR, 1):
    if aciklama.startswith("Sağlık"):
        yanit, kod = api.health()
    elif aciklama.startswith("Toplu"):
        yanit, kod = api.predict_batch(istek)
    else:
        yanit, kod = api.predict(istek)

    durum = "✅ GEÇTI" if kod == beklenen else f"❌ HATA (beklenen {beklenen})"
    sure  = yanit.get("_sure_ms", 0)
    print(f"  {i:>3}  {aciklama:<35} {kod:>5}  {beklenen:>8}  {durum}  {sure:.2f}ms")
    test_sonuclari.append({
        "aciklama": aciklama, "kod": kod, "beklenen": beklenen,
        "gecti": kod == beklenen, "sure_ms": sure,
    })

gecen  = sum(1 for t in test_sonuclari if t["gecti"])
toplam = len(test_sonuclari)
print()
print(f"  Sonuç: {gecen}/{toplam} test geçti  "
      f"({'✅ TÜMÜ GEÇTİ' if gecen == toplam else '⚠️ Bazı testler başarısız'})")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 5: PERFORMANS & THROUGHPUT ANALİZİ
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 5: Performans & Throughput Analizi")
print("─" * 65)

def throughput_olc(api, n_istek=200, batch_boyutu=1):
    """Verilen sayıda istek gönderir, istatistik döndürür."""
    np.random.seed(42)
    sureler = []
    hatalar = 0

    for _ in range(n_istek):
        if batch_boyutu == 1:
            features = np.random.uniform([4.3, 2.0, 1.0, 0.1],
                                          [7.9, 4.4, 6.9, 2.5], 4).tolist()
            t0 = time.perf_counter()
            _, kod = api.predict({"features": features})
            sureler.append((time.perf_counter() - t0) * 1000)
            if kod != 200:
                hatalar += 1
        else:
            samples = np.random.uniform(
                [4.3, 2.0, 1.0, 0.1], [7.9, 4.4, 6.9, 2.5],
                (batch_boyutu, 4)
            ).tolist()
            t0 = time.perf_counter()
            _, kod = api.predict_batch({"samples": samples})
            sureler.append((time.perf_counter() - t0) * 1000)
            if kod != 200:
                hatalar += 1

    arr = np.array(sureler)
    return {
        "n":         n_istek,
        "ortalama":  float(arr.mean()),
        "p50":       float(np.percentile(arr, 50)),
        "p95":       float(np.percentile(arr, 95)),
        "p99":       float(np.percentile(arr, 99)),
        "maks":      float(arr.max()),
        "hatalar":   hatalar,
        "sureler":   sureler,
    }

print(f"  {'Senaryo':<30} {'Ort (ms)':>10} {'p50':>8} {'p95':>8} {'p99':>8} {'Maks':>8}")
print("  " + "-" * 72)

SENARYOLAR_PERF = [
    ("Tekli tahmin (n=200)",    200, 1),
    ("Toplu tahmin /5 (n=100)", 100, 5),
    ("Toplu tahmin /20 (n=50)", 50,  20),
    ("Toplu tahmin /50 (n=20)", 20,  50),
]

perf_sonuclari = {}
for aciklama, n, batch in SENARYOLAR_PERF:
    sonuc = throughput_olc(api, n_istek=n, batch_boyutu=batch)
    perf_sonuclari[aciklama] = sonuc
    print(f"  {aciklama:<30} {sonuc['ortalama']:>10.3f} "
          f"{sonuc['p50']:>8.3f} {sonuc['p95']:>8.3f} "
          f"{sonuc['p99']:>8.3f} {sonuc['maks']:>8.3f}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 6: FLASK vs FASTAPI KARŞILAŞTIRMA
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 6: Flask vs FastAPI Detaylı Karşılaştırma")
print("─" * 65)

KARSILASTIRMA_TABLOSU = [
    ("Özellik",             "Flask",               "FastAPI"),
    ("Python sürümü",       "3.6+",                "3.8+ (type hints)"),
    ("ASGI/WSGI",           "WSGI (sync)",          "ASGI (async/sync)"),
    ("Otomatik doğrulama",  "Manuel",               "Pydantic ✅"),
    ("Swagger UI",          "flask-restx / manuel", "Otomatik /docs ✅"),
    ("Async desteği",       "Sınırlı",              "Tam (async/await) ✅"),
    ("Performans (RPS)",    "~3,000",               "~8,000–12,000 ✅"),
    ("Öğrenme eğrisi",      "Düz ✅",              "Orta"),
    ("Topluluk/olgunluk",   "Çok büyük ✅",         "Hızlı büyüyen"),
    ("Bağımlılık enjeksi.", "Yok (manuel)",         "Depends() ✅"),
    ("Test kolaylığı",      "test_client()",        "httpx.AsyncClient() ✅"),
    ("Production sunucu",   "Gunicorn",             "Uvicorn + Gunicorn"),
    ("Tip güvenliği",       "Yok",                  "Tam ✅"),
    ("Seçim kriteri",       "Basit API, prototip", "Async, büyük proje ✅"),
]

for satir in KARSILASTIRMA_TABLOSU:
    if satir[0] == "Özellik":
        print(f"  {'─'*20}  {'─'*25}  {'─'*25}")
        print(f"  {satir[0]:<20}  {satir[1]:<25}  {satir[2]:<25}")
        print(f"  {'─'*20}  {'─'*25}  {'─'*25}")
    else:
        print(f"  {satir[0]:<20}  {satir[1]:<25}  {satir[2]:<25}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 7: PRODUCTION KONFİGÜRASYONU
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 7: Production Konfigürasyonu")
print("─" * 65)
print("""
  ── Gunicorn (Flask için) ─────────────────────────────────────────
  gunicorn -w 4 -b 0.0.0.0:5000 \\
      --timeout 30 \\
      --access-logfile - \\
      --error-logfile - \\
      app_flask:app

  ── Uvicorn (FastAPI için) ────────────────────────────────────────
  # Geliştirme (hot-reload):
  uvicorn app_fastapi:app --reload --port 8000

  # Production (çok worker):
  uvicorn app_fastapi:app \\
      --host 0.0.0.0 --port 8000 \\
      --workers 4 \\
      --log-level info

  # Alternatif: Gunicorn + Uvicorn worker:
  gunicorn app_fastapi:app \\
      -w 4 -k uvicorn.workers.UvicornWorker \\
      -b 0.0.0.0:8000

  ── İdeal Worker Sayısı ───────────────────────────────────────────
  workers = (2 × CPU_çekirdek) + 1
  4 çekirdek → 9 worker  |  8 çekirdek → 17 worker
""")

cpu_sayilari = [1, 2, 4, 8, 16]
print(f"  {'CPU Çekirdek':>14} {'Önerilen Worker':>16} {'Tahmini RPS (Flask)':>20} {'Tahmini RPS (FastAPI)':>22}")
print("  " + "-" * 76)
for cpu in cpu_sayilari:
    worker  = 2 * cpu + 1
    rps_fl  = worker * 300
    rps_fa  = worker * 750
    print(f"  {cpu:>14} {worker:>16} {rps_fl:>20,} {rps_fa:>22,}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 8: GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 8: Görselleştirme (8 panel)")
print("─" * 65)

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor("#F0FDF4")
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.44, wspace=0.36,
                        top=0.93, bottom=0.05)

# ── GRAFİK 1: Test Sonuçları ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("white")
renkler_test = ["#22C55E" if t["gecti"] else "#EF4444" for t in test_sonuclari]
ax1.barh([t["aciklama"][:28] for t in test_sonuclari],
         [t["sure_ms"] for t in test_sonuclari],
         color=renkler_test, edgecolor="white")
ax1.set_title(f"API Test Senaryoları\n{gecen}/{toplam} Geçti",
              fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("Yanıt Süresi (ms)", fontsize=10)
from matplotlib.patches import Patch
ax1.legend(handles=[
    Patch(facecolor="#22C55E", label="Geçti ✅"),
    Patch(facecolor="#EF4444", label="Başarısız ❌"),
], fontsize=9)
ax1.grid(axis="x", alpha=0.4)

# ── GRAFİK 2: Yanıt Süresi Dağılımı (histogram) ──────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("white")
tekli_sureler = perf_sonuclari["Tekli tahmin (n=200)"]["sureler"]
ax2.hist(tekli_sureler, bins=30, color="#0D9488", alpha=0.8, edgecolor="white")
p50 = perf_sonuclari["Tekli tahmin (n=200)"]["p50"]
p95 = perf_sonuclari["Tekli tahmin (n=200)"]["p95"]
p99 = perf_sonuclari["Tekli tahmin (n=200)"]["p99"]
ax2.axvline(p50, color="#22C55E", linestyle="--", linewidth=2, label=f"p50={p50:.2f}ms")
ax2.axvline(p95, color="#F97316", linestyle="--", linewidth=2, label=f"p95={p95:.2f}ms")
ax2.axvline(p99, color="#EF4444", linestyle="--", linewidth=2, label=f"p99={p99:.2f}ms")
ax2.set_title("Yanıt Süresi Dağılımı\n(Tekli Tahmin, n=200)",
              fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("Süre (ms)", fontsize=10)
ax2.set_ylabel("Frekans", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.4)

# ── GRAFİK 3: Batch Boyutu vs Verimlilik ─────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("white")
batch_boyutlari  = [1, 5, 20, 50]
ort_ms_per_ornek = []
for aciklama, n, batch in SENARYOLAR_PERF:
    ort = perf_sonuclari[aciklama]["ortalama"]
    ort_ms_per_ornek.append(ort / batch)

ax3.plot(batch_boyutlari, ort_ms_per_ornek, "o-", color="#6D28D9",
         linewidth=2.5, markersize=10)
ax3.fill_between(batch_boyutlari, ort_ms_per_ornek, alpha=0.15, color="#6D28D9")
ax3.set_title("Toplu İşlem Verimliliği\n(Örnek Başına Ortalama Süre)",
              fontsize=12, fontweight="bold", pad=10)
ax3.set_xlabel("Batch Boyutu", fontsize=10)
ax3.set_ylabel("ms / Örnek", fontsize=10)
ax3.set_xscale("log")
ax3.grid(alpha=0.4)
for x, y in zip(batch_boyutlari, ort_ms_per_ornek):
    ax3.annotate(f"{y:.3f}ms", (x, y), textcoords="offset points",
                 xytext=(4, 6), fontsize=9, color="#6D28D9")

# ── GRAFİK 4: Model olasılıkları (test örnekleri) ────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor("white")
test_ozellikleri = [
    [5.1, 3.5, 1.4, 0.2],
    [6.3, 2.3, 4.4, 1.3],
    [6.5, 3.0, 5.2, 2.0],
    [5.0, 3.4, 1.5, 0.2],
    [6.7, 3.1, 4.7, 1.5],
]
tum_probalar = []
for ozell in test_ozellikleri:
    X   = np.array(ozell).reshape(1, -1)
    X_s = scaler.transform(X)
    p   = model.predict_proba(X_s)[0]
    tum_probalar.append(p)
tum_probalar = np.array(tum_probalar)

x4   = np.arange(len(test_ozellikleri))
gen4 = 0.26
renkler_sinif = ["#22C55E", "#F97316", "#6D28D9"]
for si, (sinif, renk) in enumerate(zip(SINIF_ADLARI, renkler_sinif)):
    ax4.bar(x4 + (si - 1) * gen4, tum_probalar[:, si], gen4,
            label=sinif, color=renk, alpha=0.85, edgecolor="white")
ax4.set_xticks(x4)
ax4.set_xticklabels([f"Örn.{i+1}" for i in range(len(test_ozellikleri))], fontsize=10)
ax4.set_title("Model Tahmin Olasılıkları\n(5 Test Örneği)",
              fontsize=12, fontweight="bold", pad=10)
ax4.set_ylabel("Olasılık", fontsize=10)
ax4.set_ylim(0, 1.12)
ax4.legend(fontsize=9)
ax4.grid(axis="y", alpha=0.4)

# ── GRAFİK 5: Flask vs FastAPI RPS Karşılaştırması ───────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor("white")
cpu_x = np.array(cpu_sayilari)
rps_flask  = [2 * c + 1 for c in cpu_sayilari]
rps_flask  = [w * 300 for w in rps_flask]
rps_fastapi = [w * 750 for w in [2 * c + 1 for c in cpu_sayilari]]
ax5.plot(cpu_x, rps_flask,   "o-", color="#EA580C", linewidth=2.2,
         markersize=9, label="Flask (Gunicorn)")
ax5.plot(cpu_x, rps_fastapi, "s-", color="#0D9488", linewidth=2.2,
         markersize=9, label="FastAPI (Uvicorn)")
ax5.fill_between(cpu_x, rps_flask, rps_fastapi, alpha=0.12, color="#0D9488")
ax5.set_title("Flask vs FastAPI RPS Kapasitesi\nvs CPU Çekirdek Sayısı",
              fontsize=12, fontweight="bold", pad=10)
ax5.set_xlabel("CPU Çekirdek Sayısı", fontsize=10)
ax5.set_ylabel("İstek / Saniye (RPS)", fontsize=10)
ax5.set_xticks(cpu_sayilari)
ax5.legend(fontsize=10)
ax5.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, p: f"{int(x):,}")
)
ax5.grid(alpha=0.4)

# ── GRAFİK 6: Percentil Karşılaştırma (grouped bar) ─────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("white")
p_senaryo_adlari = [s[0].replace(" (n=200)", "").replace(" (n=100)", "")
                     .replace(" (n=50)", "").replace(" (n=20)", "")
                     for s in SENARYOLAR_PERF]
p50s = [perf_sonuclari[s[0]]["p50"] for s in SENARYOLAR_PERF]
p95s = [perf_sonuclari[s[0]]["p95"] for s in SENARYOLAR_PERF]
p99s = [perf_sonuclari[s[0]]["p99"] for s in SENARYOLAR_PERF]
x6   = np.arange(len(p_senaryo_adlari))
w6   = 0.26
ax6.bar(x6 - w6, p50s, w6, label="p50", color="#22C55E", alpha=0.85, edgecolor="white")
ax6.bar(x6,      p95s, w6, label="p95", color="#F97316", alpha=0.85, edgecolor="white")
ax6.bar(x6 + w6, p99s, w6, label="p99", color="#EF4444", alpha=0.85, edgecolor="white")
ax6.set_xticks(x6)
ax6.set_xticklabels(p_senaryo_adlari, fontsize=8, rotation=15)
ax6.set_title("Percentil Yanıt Süresi\n(p50 / p95 / p99)",
              fontsize=12, fontweight="bold", pad=10)
ax6.set_ylabel("Süre (ms)", fontsize=10)
ax6.legend(fontsize=10)
ax6.grid(axis="y", alpha=0.4)

# ── GRAFİK 7: HTTP Durum Kodu Dağılımı ───────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
ax7.set_facecolor("white")
kod_sayilari = {}
for t in test_sonuclari:
    kod = str(t["kod"])
    kod_sayilari[kod] = kod_sayilari.get(kod, 0) + 1
renkler_kod = {"200": "#22C55E", "400": "#F97316", "422": "#EF4444",
               "500": "#7C3AED"}
bar_renkler = [renkler_kod.get(k, "#94A3B8") for k in kod_sayilari]
ax7.bar(list(kod_sayilari.keys()), list(kod_sayilari.values()),
        color=bar_renkler, edgecolor="white", width=0.5)
ax7.set_title("HTTP Durum Kodu Dağılımı\n(Test Senaryoları)",
              fontsize=12, fontweight="bold", pad=10)
ax7.set_xlabel("HTTP Durum Kodu", fontsize=10)
ax7.set_ylabel("İstek Sayısı", fontsize=10)
ax7.grid(axis="y", alpha=0.4)
for k, v in kod_sayilari.items():
    ax7.text(k, v + 0.05, str(v), ha="center", fontsize=12, fontweight="bold",
             color=renkler_kod.get(k, "#1E293B"))

# ── GRAFİK 8: API Akış Diyagramı (metin tabanlı) ─────────────────────────
ax8 = fig.add_subplot(gs[2, 1:])
ax8.set_facecolor("#F8FAFC")
ax8.axis("off")
ax8.set_title("Flask & FastAPI REST API Mimarisi",
              fontsize=13, fontweight="bold", pad=8)
bilesenler = [
    (0.08, 0.55, "İstemci\n(curl/httpx)", "#64748B",  0.10),
    (0.30, 0.55, "CORS\n+ Auth",          "#EA580C",  0.10),
    (0.52, 0.55, "Flask /\nFastAPI",      "#0D9488",  0.12),
    (0.74, 0.55, "Model\n(pkl)",          "#6D28D9",  0.10),
    (0.52, 0.15, "Scaler\n(pkl)",         "#D97706",  0.10),
]
for x, y, etiket, renk, w in bilesenler:
    from matplotlib.patches import FancyBboxPatch
    ax8.add_patch(FancyBboxPatch(
        (x - w / 2, y - 0.12), w, 0.24,
        boxstyle="round,pad=0.02",
        facecolor=renk, edgecolor="white",
        transform=ax8.transAxes, linewidth=2,
    ))
    ax8.text(x, y, etiket, transform=ax8.transAxes,
             ha="center", va="center", fontsize=11,
             color="white", fontweight="bold")

oklar = [
    (0.13, 0.55, 0.25, 0.55),
    (0.35, 0.55, 0.46, 0.55),
    (0.58, 0.55, 0.69, 0.55),
    (0.52, 0.49, 0.52, 0.27),
]
for x1, y1, x2, y2 in oklar:
    ax8.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color="#475569",
                                 lw=2.2, mutation_scale=20))

ek_metin = [
    (0.52, 0.88, "POST /predict\nJSON → Doğrulama → Tahmin → JSON"),
    (0.52, 0.00, "GET /health  ·  GET /docs (Swagger)  ·  POST /predict/batch"),
]
for x, y, t in ek_metin:
    ax8.text(x, y, t, transform=ax8.transAxes,
             ha="center", va="bottom", fontsize=10, color="#475569",
             style="italic")

fig.suptitle(
    "HAFTA 5 CUMARTESİ — UYGULAMA 03\n"
    "Flask & FastAPI: Model Deployment · API Test · Performans · Production",
    fontsize=14, fontweight="bold", color="#064E3B", y=0.98
)

plt.savefig("h5c_03_deployment.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("    ✅ h5c_03_deployment.png kaydedildi")
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BÖLÜM ÖZETLERİ")
print(f"  Model              : RandomForest, doğruluk={dogruluk:.4f}")
print(f"  Test senaryosu     : {toplam}  ({gecen} geçti)")
print(f"  Tekli p50 latency  : {p50:.3f}ms")
print(f"  Tekli p99 latency  : {p99:.3f}ms")
print(f"  Flask (4 CPU) RPS  : ~{4*2+1 * 300:,}")
print(f"  FastAPI (4 CPU) RPS: ~{4*2+1 * 750:,}")
print(f"  Grafik çıktısı     : h5c_03_deployment.png")
print("  ✅ UYGULAMA 03 TAMAMLANDI")
print("=" * 65)
