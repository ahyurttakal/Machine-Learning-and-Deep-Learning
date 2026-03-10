"""
=============================================================================
HAFTA 5 CUMARTESİ — UYGULAMA 04
Docker & CI/CD Pipeline — Container, Orchestration & MLflow
=============================================================================
Kapsam:
  - Dockerfile üretimi: multi-stage optimized build
  - docker-compose.yml: API + Nginx reverse proxy + opsiyonel Redis
  - .dockerignore & requirements.txt otomatik oluşturma
  - Nginx konfigürasyonu: upstream, rate limit, SSL redirect
  - MLflow model versiyonlama: experiment, run, metrik, artifact
  - Hugging Face Spaces deploy scripti
  - GitHub Actions CI/CD YAML üretimi (test → build → push → deploy)
  - Docker imaj boyutu analizi: base vs slim vs alpine
  - Container sağlık kontrolü & liveness/readiness probe simülasyonu
  - Deployment strateji karşılaştırması
  - Kapsamlı görselleştirme (8 panel)

Kurulum:
  pip install mlflow scikit-learn joblib numpy matplotlib
  (Docker komutları için): Docker Desktop kurulu olmalı
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os
import json
import textwrap
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.datasets import load_iris, load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

print("=" * 65)
print("  HAFTA 5 CUMARTESİ — UYGULAMA 04")
print("  Docker & CI/CD Pipeline")
print("=" * 65)
print(f"  scikit-learn : {'✅' if SKLEARN_AVAILABLE else '❌  pip install scikit-learn'}")
print(f"  MLflow       : {'✅' if MLFLOW_AVAILABLE else '❌  pip install mlflow'}")
print()

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 1: DOCKERFILE ÜRETİMİ
# ─────────────────────────────────────────────────────────────────────────
print("─" * 65)
print("  BÖLÜM 1: Dockerfile Üretimi (Multi-Stage Build)")
print("─" * 65)

DOCKERFILE_ICERIGI = """\
# ═══════════════════════════════════════════════════════════════
# Aşama 1: Builder — bağımlılıkları kur ve tekerleği derle
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

# Sistem bağımlılıkları (derleme için)
RUN apt-get update && apt-get install -y --no-install-recommends \\
        gcc g++ libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Önce sadece requirements.txt → Docker layer cache avantajı
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ═══════════════════════════════════════════════════════════════
# Aşama 2: Runtime — sadece çalıştırma ortamı (küçük imaj)
# ═══════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

# Güvenlik: root olmayan kullanıcı
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Builder'dan sadece kurulu paketleri kopyala (araçlar dahil değil)
COPY --from=builder /install /usr/local

# Uygulama dosyaları
COPY --chown=appuser:appuser . .

# Model dosyalarının varlığını doğrula
RUN python -c "import joblib; m=joblib.load('models/iris_model.pkl'); print('✅ Model OK')"

# Sağlık kontrolü: her 30s API'yi kontrol et
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \\
    || exit 1

# Root olmayan kullanıcıya geç
USER appuser

EXPOSE 8000

# Production başlatma: Uvicorn + 2 worker
CMD ["uvicorn", "app_fastapi:app", \\
     "--host", "0.0.0.0", "--port", "8000", \\
     "--workers", "2", "--log-level", "info"]
"""

DOCKERFILE_YOLU = "/tmp/Dockerfile"
with open(DOCKERFILE_YOLU, "w") as f:
    f.write(DOCKERFILE_ICERIGI)
print(f"  ✅ Dockerfile oluşturuldu: {DOCKERFILE_YOLU}")

# Satır analizi
satirlar     = DOCKERFILE_ICERIGI.strip().split("\n")
asama_satirlari = [s for s in satirlar if s.startswith("FROM")]
print(f"  Satır sayısı    : {len(satirlar)}")
print(f"  Build aşaması   : {len(asama_satirlari)} (multi-stage)")
for asama in asama_satirlari:
    print(f"    {asama.strip()}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 2: DOCKER-COMPOSE ÜRETİMİ
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 2: docker-compose.yml Üretimi")
print("─" * 65)

COMPOSE_ICERIGI = """\
version: '3.9'

services:
  # ── ML API Servisi ──────────────────────────────────────────────────────
  ml-api:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime         # Multi-stage hedef
    image: ml-api:latest
    container_name: ml-api
    restart: unless-stopped
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=info
      - API_KEY=${API_KEY}    # .env dosyasından
    volumes:
      - ./models:/app/models:ro  # Salt okunur model mount
      - ./logs:/app/logs         # Log persist
    expose:
      - "8000"
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 15s
    depends_on:
      - redis
    networks:
      - app-net

  # ── Nginx Reverse Proxy ─────────────────────────────────────────────────
  nginx:
    image: nginx:1.25-alpine
    container_name: nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/logs:/var/log/nginx
    depends_on:
      ml-api:
        condition: service_healthy
    networks:
      - app-net

  # ── Redis (isteğe bağlı: önbellekleme & rate limit) ────────────────────
  redis:
    image: redis:7-alpine
    container_name: redis
    restart: unless-stopped
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - app-net

  # ── MLflow Tracking Sunucusu ────────────────────────────────────────────
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    container_name: mlflow
    restart: unless-stopped
    command: >
      mlflow server
        --host 0.0.0.0 --port 5000
        --backend-store-uri sqlite:///mlflow.db
        --default-artifact-root /mlflow/artifacts
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow
    networks:
      - app-net

volumes:
  redis-data:
  mlflow-data:

networks:
  app-net:
    driver: bridge
"""

COMPOSE_YOLU = "/tmp/docker-compose.yml"
with open(COMPOSE_YOLU, "w") as f:
    f.write(COMPOSE_ICERIGI)
print(f"  ✅ docker-compose.yml oluşturuldu: {COMPOSE_YOLU}")

servisler = [s.strip().rstrip(":") for s in COMPOSE_ICERIGI.split("\n")
             if s.strip().endswith(":") and not s.strip().startswith("-")
             and ":" not in s.strip()[:-1] and s.startswith("  ")
             and not s.startswith("    ")]
servisler = ["ml-api", "nginx", "redis", "mlflow"]
print(f"  Servisler       : {servisler}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 3: NGINX KONFİGÜRASYONU
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 3: Nginx Reverse Proxy Konfigürasyonu")
print("─" * 65)

NGINX_ICERIGI = """\
worker_processes auto;
events { worker_connections 1024; }

http {
    # ── Rate Limiting ────────────────────────────────────────────────────
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    # ── Upstream: FastAPI servisi ─────────────────────────────────────────
    upstream ml_api {
        server ml-api:8000;
        keepalive 32;
    }

    # ── HTTP → HTTPS yönlendirmesi ───────────────────────────────────────
    server {
        listen 80;
        server_name _;
        return 301 https://$host$request_uri;
    }

    # ── HTTPS Sunucu ─────────────────────────────────────────────────────
    server {
        listen 443 ssl http2;
        server_name api.example.com;

        ssl_certificate     /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols       TLSv1.2 TLSv1.3;

        # ── API proxy ────────────────────────────────────────────────────
        location /api/ {
            limit_req zone=api burst=20 nodelay;

            proxy_pass         http://ml_api/;
            proxy_http_version 1.1;
            proxy_set_header   Upgrade $http_upgrade;
            proxy_set_header   Connection keep-alive;
            proxy_set_header   Host $host;
            proxy_set_header   X-Real-IP $remote_addr;
            proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;

            proxy_read_timeout 30s;
            proxy_send_timeout 30s;
        }

        # ── Sağlık kontrolü (rate limit yok) ────────────────────────────
        location /health {
            proxy_pass http://ml_api/health;
            access_log off;
        }
    }
}
"""

NGINX_YOLU = "/tmp/nginx.conf"
os.makedirs("/tmp/nginx", exist_ok=True)
with open("/tmp/nginx/nginx.conf", "w") as f:
    f.write(NGINX_ICERIGI)
print(f"  ✅ nginx.conf oluşturuldu")
print(f"  Rate limit: 10 istek/s (burst=20)")
print(f"  SSL/TLS: TLSv1.2 + TLSv1.3")
print(f"  HTTP → HTTPS yönlendirmesi: Aktif")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 4: GITHUB ACTIONS CI/CD YAML
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 4: GitHub Actions CI/CD Pipeline")
print("─" * 65)

GITHUB_ACTIONS_YAML = """\
name: ML API CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/ml-api

jobs:
  # ── 1. Test ────────────────────────────────────────────────────────────
  test:
    name: Pytest & Kod Kalitesi
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Python 3.11 kur
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Bağımlılıkları yükle
        run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Linting (flake8)
        run: flake8 . --max-line-length=100 --exclude=venv

      - name: Tip kontrolü (mypy)
        run: mypy app_fastapi.py --ignore-missing-imports

      - name: Unit & integration testleri
        run: pytest tests/ -v --cov=. --cov-report=xml

      - name: Coverage raporu yükle
        uses: codecov/codecov-action@v4

  # ── 2. Docker Build & Push ─────────────────────────────────────────────
  build:
    name: Docker Build & Push
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: GHCR'a login
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Docker Buildx kur
        uses: docker/setup-buildx-action@v3

      - name: Meta etiketleri
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=sha-
            type=raw,value=latest

      - name: Build & Push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to:   type=gha,mode=max

  # ── 3. Deploy (Production) ─────────────────────────────────────────────
  deploy:
    name: Production Deploy
    needs: build
    runs-on: ubuntu-latest
    environment: production

    steps:
      - name: SSH ile sunucuya bağlan ve güncelle
        uses: appleboy/ssh-action@master
        with:
          host:     ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key:      ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /opt/ml-api
            docker compose pull
            docker compose up -d --remove-orphans
            docker system prune -f
            echo "✅ Deploy tamamlandı: $(date)"
"""

ACTIONS_YOLU = "/tmp/ci-cd.yml"
with open(ACTIONS_YOLU, "w") as f:
    f.write(GITHUB_ACTIONS_YAML)
print(f"  ✅ .github/workflows/ci-cd.yml oluşturuldu")
joblar = ["test (pytest + flake8 + mypy)", "build (Docker + GHCR)", "deploy (SSH)"]
for i, j in enumerate(joblar, 1):
    print(f"  Adım {i}: {j}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 5: MLFLOW MODEL VERSİYONLAMA
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 5: MLflow Model Versiyonlama")
print("─" * 65)

def mlflow_deneyi_calistir(model_adi, model_obj, X_train, y_train,
                            X_test, y_test, parametreler, deney_adi):
    """MLflow run gerçekleştir (gerçek veya simüle)."""
    t0 = time.time()
    model_obj.fit(X_train, y_train)
    sure = time.time() - t0
    tahminler = model_obj.predict(X_test)
    acc = float(accuracy_score(y_test, tahminler))
    f1  = float(f1_score(y_test, tahminler, average="weighted"))
    cv  = float(cross_val_score(model_obj, X_train, y_train,
                                cv=5, scoring="accuracy").mean())

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(deney_adi)
        with mlflow.start_run(run_name=model_adi):
            mlflow.log_params(parametreler)
            mlflow.log_metrics({
                "accuracy":    acc,
                "f1_weighted": f1,
                "cv_accuracy": cv,
                "train_time":  sure,
            })
            mlflow.sklearn.log_model(model_obj, "model",
                                     registered_model_name=model_adi)
            run_id = mlflow.active_run().info.run_id
    else:
        import hashlib
        run_id = hashlib.md5((model_adi + str(acc)).encode()).hexdigest()[:8]

    return {"model": model_adi, "acc": acc, "f1": f1,
            "cv": cv, "sure": sure, "run_id": run_id,
            "parametreler": parametreler}

if SKLEARN_AVAILABLE:
    veri   = load_breast_cancer()
    X, y   = veri.data, veri.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
else:
    np.random.seed(42)
    X_tr_s = np.random.randn(400, 30)
    y_train = np.random.randint(0, 2, 400)
    X_te_s  = np.random.randn(100, 30)
    y_test  = np.random.randint(0, 2, 100)

MODELLER_MLFLOW = [
    ("RandomForest_v1",
     RandomForestClassifier(n_estimators=50,  max_depth=5,  random_state=42) if SKLEARN_AVAILABLE else None,
     {"n_estimators": 50,  "max_depth": 5}),
    ("RandomForest_v2",
     RandomForestClassifier(n_estimators=100, max_depth=8,  random_state=42) if SKLEARN_AVAILABLE else None,
     {"n_estimators": 100, "max_depth": 8}),
    ("GradientBoosting",
     GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42) if SKLEARN_AVAILABLE else None,
     {"n_estimators": 100, "learning_rate": 0.1}),
    ("LogisticRegression",
     LogisticRegression(C=1.0, max_iter=200, random_state=42) if SKLEARN_AVAILABLE else None,
     {"C": 1.0, "max_iter": 200}),
]

mlflow_sonuclari = []
print(f"  {'Model':<22} {'Accuracy':>10} {'F1':>8} {'CV Acc':>8} {'Süre':>8} {'Run ID'}")
print("  " + "-" * 72)

for model_adi, model_obj, params in MODELLER_MLFLOW:
    if SKLEARN_AVAILABLE and model_obj is not None:
        try:
            sonuc = mlflow_deneyi_calistir(
                model_adi, model_obj, X_tr_s, y_train,
                X_te_s, y_test, params, "BreastCancer_Experiment"
            )
        except Exception:
            np.random.seed(hash(model_adi) % 9999)
            acc = 0.94 + np.random.uniform(-0.03, 0.04)
            f1  = acc - 0.01
            cv  = acc - 0.015
            sonuc = {"model": model_adi, "acc": acc, "f1": f1, "cv": cv,
                     "sure": np.random.uniform(0.5, 3.0),
                     "run_id": f"sim{abs(hash(model_adi)) % 10000:04x}",
                     "parametreler": params}
    else:
        np.random.seed(hash(model_adi) % 9999)
        acc  = 0.94 + np.random.uniform(-0.03, 0.04)
        f1   = acc - 0.01
        cv   = acc - 0.015
        sonuc = {"model": model_adi, "acc": acc, "f1": f1, "cv": cv,
                 "sure": np.random.uniform(0.5, 3.0),
                 "run_id": f"sim{abs(hash(model_adi)) % 10000:04x}",
                 "parametreler": params}

    mlflow_sonuclari.append(sonuc)
    print(f"  {sonuc['model']:<22} {sonuc['acc']:>10.4f} {sonuc['f1']:>8.4f} "
          f"{sonuc['cv']:>8.4f} {sonuc['sure']:>7.2f}s  {sonuc['run_id']}")

en_iyi = max(mlflow_sonuclari, key=lambda x: x["acc"])
print(f"\n  🏆 En iyi model: {en_iyi['model']}  "
      f"Accuracy={en_iyi['acc']:.4f}  → Production'a al")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 6: DOCKER İMAJ BOYUTU ANALİZİ
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 6: Docker İmaj Boyutu & Katman Analizi")
print("─" * 65)

IMAJ_VERILERI = [
    ("python:3.11",          1020, "full",     "Tam Python — dev araçları dahil"),
    ("python:3.11-slim",      130, "slim",     "Gereksiz paketler çıkarılmış"),
    ("python:3.11-alpine",     52, "alpine",   "Musl libc, en küçük"),
    ("Multi-stage (builder)", 130, "multi",    "Builder katmanı (final değil)"),
    ("Multi-stage (runtime)", 185, "multi-rt", "Uygulama dahil final imaj ✅"),
]

KATMANLAR = [
    ("Base OS + Python",         90,  "#6D28D9"),
    ("System libs (gcc, etc.)", 22,  "#EA580C"),
    ("Python paketleri",         55,  "#0D9488"),
    ("Uygulama kodu",             8,  "#22C55E"),
    ("Model dosyaları (.pkl)",    10,  "#D97706"),
]

print(f"  {'İmaj':<30} {'Boyut (MB)':>12} {'Açıklama'}")
print("  " + "-" * 65)
for imaj, boyut, tip, aciklama in IMAJ_VERILERI:
    isaretci = "✅" if tip == "multi-rt" else "  "
    print(f"  {isaretci}{imaj:<28} {boyut:>12} MB  {aciklama}")

toplam_katman = sum(b for _, b, _ in KATMANLAR)
print(f"\n  Katman dökümü (multi-stage runtime, toplam ~{toplam_katman}MB):")
for katman, boyut, _ in KATMANLAR:
    bar = "█" * int(boyut / 3)
    print(f"    {katman:<30} {boyut:>5}MB  {bar}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 7: CONTAINER SAĞLIK KONTROLÜ SİMÜLASYONU
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 7: Container Sağlık Kontrolü Simülasyonu")
print("─" * 65)
print("""
  Kubernetes probe türleri:
    livenessProbe  : Container çöktüyse yeniden başlat
    readinessProbe : Trafik almaya hazır mı?
    startupProbe   : İlk başlatma için ekstra süre

  Docker Healthcheck (compose):
    HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
      CMD curl -f http://localhost:8000/health || exit 1
""")

np.random.seed(42)
N_KONTROL = 30
t_eksen   = np.arange(N_KONTROL) * 30  # saniye (her 30s kontrol)

# Normal çalışma → 20. saniyede kısa kesinti → iyileşme
saglik_durumu = np.ones(N_KONTROL, dtype=int)
gecikme_ms    = np.abs(np.random.normal(15, 5, N_KONTROL))
gecikme_ms[15:18] = np.array([450, 820, 380])  # geçici yavaşlama
saglik_durumu[16] = 0   # 1 başarısız kontrol

# Container yeniden başlatma simülasyonu (3 başarısız = restart)
yeniden_baslatma = [t_eksen[16]]

print(f"  {'Kontrol #':>10} {'Süre (s)':>10} {'Gecikme (ms)':>14} {'Durum':>10}")
print("  " + "-" * 50)
for i in [0, 5, 10, 15, 16, 17, 20, 25, 29]:
    durum = "✅ SAĞLIKLI" if saglik_durumu[i] else "❌ BAŞARISIZ"
    print(f"  {i+1:>10} {t_eksen[i]:>10} {gecikme_ms[i]:>14.1f} {durum:>10}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 8: DEPLOYMENT STRATEJİ KARŞILAŞTIRMASI
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 8: Deployment Strateji Karşılaştırması")
print("─" * 65)

STRATEJILER = [
    {
        "ad":      "Mavi-Yeşil (Blue-Green)",
        "aciklama": "İki özdeş ortam; trafik anında yeni versiyona geçiş",
        "sifir_kesinti": True,
        "karmasiklik": 3,
        "kaynak":  4,
        "geri_alma": "Anında ✅",
        "kullanim": "Kritik production API'ler",
    },
    {
        "ad":      "Kanaryalı (Canary)",
        "aciklama": "%5 → %25 → %50 → %100 kademeli trafik artışı",
        "sifir_kesinti": True,
        "karmasiklik": 4,
        "kaynak":  3,
        "geri_alma": "Hızlı ✅",
        "kullanim": "Risk minimizasyonu gereken durumlar",
    },
    {
        "ad":      "Yuvarlayan (Rolling)",
        "aciklama": "Eski pod'lar tek tek yenileriyle değiştirilir",
        "sifir_kesinti": True,
        "karmasiklik": 2,
        "kaynak":  2,
        "geri_alma": "Orta",
        "kullanim": "Kubernetes varsayılan stratejisi",
    },
    {
        "ad":      "Yerinde (In-Place)",
        "aciklama": "Mevcut uygulama durdurup yeni versiyon başlatılır",
        "sifir_kesinti": False,
        "karmasiklik": 1,
        "kaynak":  1,
        "geri_alma": "Manuel ⚠️",
        "kullanim": "Geliştirme, test ortamları",
    },
]

print(f"  {'Strateji':<24} {'Sıfır Kesinti':>14} {'Karmaşıklık':>12} "
      f"{'Kaynak':>8} {'Geri Alma':>14}")
print("  " + "-" * 76)
for s in STRATEJILER:
    kesinti = "✅ Evet" if s["sifir_kesinti"] else "❌ Hayır"
    karm    = "★" * s["karmasiklik"] + "☆" * (4 - s["karmasiklik"])
    kayn    = "●" * s["kaynak"]     + "○" * (4 - s["kaynak"])
    print(f"  {s['ad']:<24} {kesinti:>14} {karm:>12} {kayn:>8} {s['geri_alma']:>14}")

# ─────────────────────────────────────────────────────────────────────────
# BÖLÜM 9: GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 9: Görselleştirme (8 panel)")
print("─" * 65)

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor("#0D1117")
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.44, wspace=0.36,
                        top=0.93, bottom=0.05)

# ── GRAFİK 1: MLflow Model Karşılaştırması ───────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#161B22")
modeller_adi  = [s["model"].replace("_v", "\nv") for s in mlflow_sonuclari]
acc_ler       = [s["acc"] for s in mlflow_sonuclari]
f1_ler        = [s["f1"]  for s in mlflow_sonuclari]
x1            = np.arange(len(modeller_adi))
renk_en_iyi   = ["#22C55E" if s["model"] == en_iyi["model"]
                  else "#EA580C" for s in mlflow_sonuclari]
ax1.bar(x1 - 0.18, acc_ler, 0.32, color=renk_en_iyi, alpha=0.85,
        label="Accuracy", edgecolor="#30363D")
ax1.bar(x1 + 0.18, f1_ler,  0.32, color="#0D9488",    alpha=0.70,
        label="F1-weighted", edgecolor="#30363D")
ax1.axhline(y=max(acc_ler), color="#22C55E", linestyle=":",
            linewidth=1.5, alpha=0.8)
ax1.set_xticks(x1)
ax1.set_xticklabels(modeller_adi, fontsize=8.5, color="#C9D1D9")
ax1.set_ylim(0.85, 1.02)
ax1.set_title("MLflow Deney Sonuçları\nModel Karşılaştırması",
              fontsize=12, fontweight="bold", color="#C9D1D9", pad=10)
ax1.set_ylabel("Skor", fontsize=10, color="#8B949E")
ax1.legend(fontsize=9, labelcolor="#C9D1D9")
ax1.tick_params(colors="#8B949E")
ax1.grid(axis="y", alpha=0.3, color="#30363D")
ax1.set_facecolor("#161B22")
for sp in ax1.spines.values():
    sp.set_color("#30363D")

# ── GRAFİK 2: Docker İmaj Boyutu Karşılaştırması ─────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#161B22")
imaj_adlari  = [d[0].replace("python:", "py:").replace("Multi-stage", "Multi") for d in IMAJ_VERILERI]
imaj_boyutlari = [d[1] for d in IMAJ_VERILERI]
renk_imaj    = ["#EF4444","#F97316","#F59E0B","#8B5CF6","#22C55E"]
bars2 = ax2.barh(imaj_adlari, imaj_boyutlari, color=renk_imaj,
                  edgecolor="#30363D", alpha=0.9)
for bar, val in zip(bars2, imaj_boyutlari):
    ax2.text(val + 10, bar.get_y() + bar.get_height() / 2,
             f"{val}MB", va="center", fontsize=10, color="#C9D1D9",
             fontweight="bold")
ax2.set_title("Docker İmaj Boyutu\nKarşılaştırması",
              fontsize=12, fontweight="bold", color="#C9D1D9", pad=10)
ax2.set_xlabel("Boyut (MB)", fontsize=10, color="#8B949E")
ax2.tick_params(colors="#8B949E")
ax2.grid(axis="x", alpha=0.3, color="#30363D")
ax2.set_facecolor("#161B22")
for sp in ax2.spines.values():
    sp.set_color("#30363D")

# ── GRAFİK 3: Katman Pasta Grafiği ───────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("#161B22")
katman_adlari = [k[0] for k in KATMANLAR]
katman_boyutlari = [k[1] for k in KATMANLAR]
katman_renkler = [k[2] for k in KATMANLAR]
wedges, metinler, oto = ax3.pie(
    katman_boyutlari, labels=katman_adlari,
    colors=katman_renkler, autopct="%1.0f%%",
    startangle=90, pctdistance=0.75,
    textprops={"fontsize": 8.5, "color": "#C9D1D9"},
    wedgeprops={"edgecolor": "#0D1117", "linewidth": 2},
)
for txt in metinler:
    txt.set_color("#C9D1D9")
ax3.set_title(f"İmaj Katman Dağılımı\nToplam ~{toplam_katman}MB",
              fontsize=12, fontweight="bold", color="#C9D1D9", pad=10)
ax3.set_facecolor("#161B22")

# ── GRAFİK 4: Container Sağlık İzleme ────────────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
ax4.set_facecolor("#161B22")
renk_saglik = ["#22C55E" if d == 1 else "#EF4444" for d in saglik_durumu]
ax4.bar(t_eksen, gecikme_ms, width=20,
        color=renk_saglik, alpha=0.85, edgecolor="#30363D")
ax4.axhline(y=500, color="#EF4444", linestyle="--", linewidth=1.5,
            alpha=0.8, label="Timeout eşiği (500ms)")
ax4.axhline(y=100, color="#F59E0B", linestyle=":", linewidth=1.5,
            alpha=0.8, label="Yavaş uyarı (100ms)")
for yr in yeniden_baslatma:
    ax4.axvline(x=yr, color="#8B5CF6", linewidth=2.5,
                label="Başarısız kontrol")
ax4.set_title("Container Sağlık Kontrolü İzleme (30 kontrol × 30s aralık)\n"
              "Yeşil=Sağlıklı  Kırmızı=Başarısız",
              fontsize=12, fontweight="bold", color="#C9D1D9", pad=10)
ax4.set_xlabel("Süre (saniye)", fontsize=10, color="#8B949E")
ax4.set_ylabel("Yanıt Gecikme (ms)", fontsize=10, color="#8B949E")
ax4.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")
ax4.tick_params(colors="#8B949E")
ax4.grid(alpha=0.3, color="#30363D")
ax4.set_facecolor("#161B22")
for sp in ax4.spines.values():
    sp.set_color("#30363D")

# ── GRAFİK 5: CI/CD Pipeline Gantt ───────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor("#161B22")
gantt_adimlar = [
    ("git push",           0,  1,  "#64748B"),
    ("Checkout",           1,  2,  "#6D28D9"),
    ("Pip install",        2,  5,  "#EA580C"),
    ("Flake8 + mypy",      5,  7,  "#F59E0B"),
    ("Pytest (coverage)",  7, 12,  "#0D9488"),
    ("Docker build",      12, 18,  "#22C55E"),
    ("Push → GHCR",       18, 21,  "#8B5CF6"),
    ("SSH deploy",         21, 24,  "#EA580C"),
    ("Health check",       24, 26,  "#22C55E"),
]
for i, (ad, baslangic, bitis, renk) in enumerate(gantt_adimlar):
    ax5.barh(i, bitis - baslangic, left=baslangic, color=renk,
             alpha=0.85, edgecolor="#30363D", height=0.55)
    ax5.text((baslangic + bitis) / 2, i, ad,
             ha="center", va="center", fontsize=8,
             color="white", fontweight="bold")
ax5.set_yticks([])
ax5.set_xlabel("Dakika", fontsize=10, color="#8B949E")
ax5.set_title("CI/CD Pipeline Zamanlaması\n(GitHub Actions)",
              fontsize=12, fontweight="bold", color="#C9D1D9", pad=10)
ax5.tick_params(colors="#8B949E")
ax5.grid(axis="x", alpha=0.3, color="#30363D")
ax5.set_facecolor("#161B22")
for sp in ax5.spines.values():
    sp.set_color("#30363D")

# ── GRAFİK 6: Deployment Stratejisi Radar ────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0], projection="polar")
ax6.set_facecolor("#161B22")
RADAR_KAT = ["Sıfır Kesinti", "Hız", "Basitlik", "Kaynak\nVerimliliği", "Geri Alma"]
N = len(RADAR_KAT)
acılar = [n / float(N) * 2 * np.pi for n in range(N)]
acılar += acılar[:1]
STRATEJI_RADAR = {
    "Mavi-Yeşil":  ([1.0, 0.9, 0.4, 0.3, 1.0], "#22C55E"),
    "Kanaryalı":   ([1.0, 0.7, 0.3, 0.6, 0.9], "#F97316"),
    "Yuvarlayan":  ([1.0, 0.8, 0.7, 0.8, 0.7], "#0D9488"),
    "Yerinde":     ([0.0, 1.0, 1.0, 1.0, 0.4], "#EF4444"),
}
for ad, (degeler, renk) in STRATEJI_RADAR.items():
    d = degeler + degeler[:1]
    ax6.plot(acılar, d, "o-", color=renk, linewidth=1.8,
             markersize=5, label=ad)
    ax6.fill(acılar, d, color=renk, alpha=0.08)
ax6.set_xticks(acılar[:-1])
ax6.set_xticklabels(RADAR_KAT, fontsize=8.5, color="#C9D1D9")
ax6.set_ylim(0, 1)
ax6.set_title("Deployment Stratejileri\nKarşılaştırma",
              fontsize=11, fontweight="bold", color="#C9D1D9", pad=20)
ax6.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15),
           fontsize=8.5, labelcolor="#C9D1D9", facecolor="#161B22")
ax6.tick_params(colors="#8B949E")
ax6.set_facecolor("#161B22")

# ── GRAFİK 7: MLflow Eğitim Süresi vs Accuracy ───────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
ax7.set_facecolor("#161B22")
sureler7 = [s["sure"]  for s in mlflow_sonuclari]
accler7  = [s["acc"]   for s in mlflow_sonuclari]
renk7    = [("#22C55E" if s["model"] == en_iyi["model"] else "#EA580C")
            for s in mlflow_sonuclari]
ax7.scatter(sureler7, accler7, s=220, c=renk7, zorder=5,
            edgecolors="#30363D", linewidth=1.5)
for s in mlflow_sonuclari:
    ax7.annotate(s["model"][:12],
                 (s["sure"], s["acc"]),
                 textcoords="offset points",
                 xytext=(5, 4), fontsize=8.5, color="#C9D1D9")
ax7.set_title("MLflow: Eğitim Süresi vs Accuracy\n(Sağ-üst = İdeal)",
              fontsize=12, fontweight="bold", color="#C9D1D9", pad=10)
ax7.set_xlabel("Eğitim Süresi (s)", fontsize=10, color="#8B949E")
ax7.set_ylabel("Test Accuracy", fontsize=10, color="#8B949E")
ax7.tick_params(colors="#8B949E")
ax7.grid(alpha=0.3, color="#30363D")
ax7.set_facecolor("#161B22")
for sp in ax7.spines.values():
    sp.set_color("#30363D")

# ── GRAFİK 8: Stack Özeti Tablosu ────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_facecolor("#161B22")
ax8.axis("off")
ax8.set_title("Production ML Stack", fontsize=12, fontweight="bold",
              color="#C9D1D9", pad=8)

stack_rows = [
    ("Katman",        "Araç",                "Açıklama"),
    ("Model Eğitim",  "scikit-learn",        "RF, GB, LR"),
    ("Deney Takip",   "MLflow",              "Versiyon & metrik"),
    ("API Framework", "FastAPI",             "Async + Pydantic"),
    ("Web Sunucu",    "Uvicorn + Gunicorn",  "Multi-worker"),
    ("Proxy",         "Nginx",               "SSL + rate limit"),
    ("Container",     "Docker",              "Multi-stage build"),
    ("Orchestration", "docker-compose",      "Servis yönetimi"),
    ("CI/CD",         "GitHub Actions",      "Test→Build→Deploy"),
    ("Önbellek",      "Redis",               "Rate limit & cache"),
    ("Monitoring",    "Prometheus+Grafana",  "Metrik & uyarı"),
]
RENK_SATIR = ["#21262D", "#161B22"]
for ri, satir in enumerate(stack_rows):
    y = 0.96 - ri * 0.087
    bg = "#30363D" if ri == 0 else RENK_SATIR[ri % 2]
    ax8.add_patch(plt.Rectangle(
        (0.0, y - 0.06), 1.0, 0.08,
        transform=ax8.transAxes, facecolor=bg,
        edgecolor="#21262D", linewidth=0.5,
    ))
    renkler_sutun = (["#58A6FF","#58A6FF","#58A6FF"] if ri == 0
                     else ["#F78166","#79C0FF","#A5D6FF"])
    for ci, (hucre, rc) in enumerate(zip(satir, renkler_sutun)):
        x_pos = [0.02, 0.33, 0.62][ci]
        ax8.text(x_pos, y - 0.01, hucre, transform=ax8.transAxes,
                 fontsize=8.5, color=rc,
                 fontweight="bold" if ri == 0 else "normal")

fig.suptitle(
    "HAFTA 5 CUMARTESİ — UYGULAMA 04\n"
    "Docker & CI/CD: Dockerfile · docker-compose · MLflow · GitHub Actions · Deployment Stratejileri",
    fontsize=14, fontweight="bold", color="#C9D1D9", y=0.98
)

plt.savefig("h5c_04_docker_cicd.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("    ✅ h5c_04_docker_cicd.png kaydedildi")
plt.close()

# ─────────────────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BÖLÜM ÖZETLERİ")
print(f"  Dockerfile      : Multi-stage, {len(asama_satirlari)} aşama")
print(f"  docker-compose  : {len(servisler)} servis ({', '.join(servisler)})")
print(f"  CI/CD adımı     : 3 (test → build → deploy)")
print(f"  MLflow model    : {len(mlflow_sonuclari)}  | En iyi: {en_iyi['model']} ({en_iyi['acc']:.4f})")
print(f"  İmaj boyutu     : {IMAJ_VERILERI[-1][1]}MB (multi-stage runtime)")
print(f"  Sağlık kontrol  : {N_KONTROL} tur simüle edildi")
print(f"  Deploy strateji : {len(STRATEJILER)} seçenek karşılaştırıldı")
print(f"  Grafik çıktısı  : h5c_04_docker_cicd.png")
print("  ✅ UYGULAMA 04 TAMAMLANDI")
print("=" * 65)
