"""
=============================================================================
AGENTİK AI — UYGULAMA 02
RAG Agent + Tool Use & Function Calling
=============================================================================
Kapsam:
  - Vektör veritabanı simülasyonu: TF-IDF + cosine similarity
  - BaglamliRAG: chunk'lama, embedding, retrieval, generation
  - Function Calling: OpenAI formatında JSON şema tanımı
  - ToolRouter: araç şemasına göre otomatik araç seçimi
  - RetrievalAgent: RAG + araç kullanımı entegrasyonu
  - 3 farklı soru türü: factual, multi-hop, reasoning
  - Chunk kalitesi metrikleri: precision, recall, faithfulness
  - Kapsamlı görselleştirme (8 panel)
=============================================================================
"""

import re, math, json, time, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("  AGENTİK AI — UYGULAMA 02")
print("  RAG Agent + Tool Use & Function Calling")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 1: DÖKÜMAN TABANI VE CHUNK'LAMA
# ─────────────────────────────────────────────────────────────────
print("\n─" * 33)
print("  BÖLÜM 1: Döküman Tabanı ve Chunk'lama")
print("─" * 65)

DOKUMANLAR = {
    "doc_agentic": """
    Agentic AI, belirli bir hedefe ulaşmak için bağımsız planlama yapan,
    kararlar alan, araçları kullanan ve çevresinden öğrenen yapay zeka sistemidir.
    ReAct framework'ü Reasoning ve Acting bileşenlerini birleştirir.
    Chain of Thought muhakeme adımlarını sıralı şekilde ortaya koyar.
    Multi-agent sistemlerde orkestratör ajanlar uzman ajanlara görev dağıtır.
    """,
    "doc_rag": """
    RAG (Retrieval-Augmented Generation) yapay zeka sistemlerinde bilgi tabanından
    dinamik bilgi çekmeyi sağlar. Vektör veritabanları embedding benzerliği ile
    arama yapar. FAISS, Pinecone, Weaviate ve ChromaDB popüler vektör DB'lerdir.
    Chunk boyutu genellikle 256-512 token arasında ayarlanır.
    Hybrid search hem lexical hem semantic aramayı birleştirir.
    """,
    "doc_llm": """
    Büyük Dil Modelleri (LLM) milyarlarca parametre ile eğitilen transformer tabanlı
    modellerdir. GPT-4, Claude 3.5, Gemini 2.0 ve Llama 3 öne çıkan modellerdir.
    Few-shot learning, modelin birkaç örnekle yeni görevlere adapte olmasını sağlar.
    Fine-tuning ile modeller belirli domain'lere özelleştirilebilir.
    Instruction following ve RLHF ile modeller hizalanır.
    """,
    "doc_tools": """
    Function calling, LLM'lerin harici araçları çağırmasını sağlar.
    OpenAI API'de tools parametresi ile araç şemaları tanımlanır.
    Anthropic Claude tool_use mesaj tipiyle araç çağrısı yapar.
    MCP (Model Context Protocol) araç entegrasyonu standardize eder.
    Web search, code execution ve API calls en sık kullanılan araçlardır.
    """,
    "doc_security": """
    Agentic AI güvenliği için HITL (Human-In-The-Loop) kritik öneme sahiptir.
    Prompt injection saldırıları dış verilerle ajan manipülasyonu hedefler.
    Minimum yetki prensibi ile ajanlara sadece gerekli izinler verilmelidir.
    Sandbox ortamlar kod çalıştırma risklerini izole eder.
    Denetim izi tüm ajan eylemlerini kayıt altına almalıdır.
    """,
    "doc_frameworks": """
    LangChain en yaygın agentic AI geliştirme framework'üdür.
    LangGraph durum makinesi tabanlı döngüsel iş akışları destekler.
    AutoGen Microsoft'un çok-ajan iletişim framework'üdür.
    CrewAI rol tabanlı ajan sistemleri için tasarlanmıştır.
    Semantic Kernel Microsoft'un enterprise AI framework'üdür.
    """,
}

def chunk_olustur(metin: str, chunk_boyut: int = 80) -> list:
    """Metni kelime bazlı chunk'lara böler, overlap destekli."""
    cumle_ayrac = r'(?<=[.!?])\s+'
    cumleler = re.split(cumle_ayrac, metin.strip())
    cumleler = [c.strip() for c in cumleler if len(c.strip()) > 20]

    chunks, mevcut = [], []
    for cumle in cumleler:
        mevcut.append(cumle)
        if sum(len(c.split()) for c in mevcut) >= chunk_boyut:
            chunks.append(" ".join(mevcut))
            mevcut = mevcut[-1:]   # overlap: son cümle tekrar
    if mevcut:
        chunks.append(" ".join(mevcut))
    return chunks

tum_chunks = []
for doc_id, icerik in DOKUMANLAR.items():
    for i, chunk in enumerate(chunk_olustur(icerik, chunk_boyut=50)):
        tum_chunks.append({
            "id":      f"{doc_id}_c{i}",
            "doc_id":  doc_id,
            "metin":   chunk,
            "kelimeler": set(re.sub(r'[^\w\s]', '', chunk.lower()).split()),
        })

print(f"  Döküman sayısı : {len(DOKUMANLAR)}")
print(f"  Toplam chunk   : {len(tum_chunks)}")
print(f"  Ort. chunk boy : {np.mean([len(c['metin'].split()) for c in tum_chunks]):.1f} kelime")
for c in tum_chunks[:3]:
    print(f"  [{c['id']}] {c['metin'][:60]}…")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 2: TF-IDF VEKTÖR VERİTABANI
# ─────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 2: TF-IDF Vektör Veritabanı")
print("─" * 65)

class VektorDB:
    """TF-IDF tabanlı basit vektör veritabanı."""

    def __init__(self):
        self.chunks   = []
        self.vocab    = {}
        self.idf      = {}
        self.tfidf_m  = None

    def _tokenize(self, metin: str) -> list:
        return re.sub(r'[^\w\s]', '', metin.lower()).split()

    def _tf(self, kelimeler: list) -> dict:
        sayac = Counter(kelimeler)
        n = len(kelimeler) or 1
        return {k: v/n for k, v in sayac.items()}

    def index(self, chunks: list):
        self.chunks = chunks
        kelime_seti = set()
        for c in chunks:
            kelime_seti.update(self._tokenize(c["metin"]))
        self.vocab = {k: i for i, k in enumerate(sorted(kelime_seti))}
        V = len(self.vocab)
        N = len(chunks)

        # DF hesapla
        df = np.zeros(V)
        for c in chunks:
            for k in set(self._tokenize(c["metin"])):
                if k in self.vocab:
                    df[self.vocab[k]] += 1

        self.idf = np.log((N + 1) / (df + 1)) + 1  # smooth IDF

        # TF-IDF matrisi
        self.tfidf_m = np.zeros((N, V))
        for i, c in enumerate(chunks):
            tfs = self._tf(self._tokenize(c["metin"]))
            for k, v in tfs.items():
                if k in self.vocab:
                    self.tfidf_m[i, self.vocab[k]] = v * self.idf[self.vocab[k]]

        # L2 normalize
        norms = np.linalg.norm(self.tfidf_m, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.tfidf_m /= norms
        print(f"  ✅ İndekslendi: {N} chunk, {V} kelime vokabüler")

    def ara(self, sorgu: str, top_k: int = 3) -> list:
        """Cosine similarity ile top-k chunk döndürür."""
        q_tok = self._tokenize(sorgu)
        q_tf  = self._tf(q_tok)
        q_vec = np.zeros(len(self.vocab))
        for k, v in q_tf.items():
            if k in self.vocab:
                q_vec[self.vocab[k]] = v * self.idf[self.vocab[k]]

        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec /= q_norm

        benzerlikler = self.tfidf_m @ q_vec
        top_idx      = np.argsort(benzerlikler)[::-1][:top_k]
        return [
            {"chunk": self.chunks[i], "skor": float(benzerlikler[i])}
            for i in top_idx if benzerlikler[i] > 0
        ]

vdb = VektorDB()
vdb.index(tum_chunks)

# Test aramaları
test_aramalar = [
    ("ReAct framework nedir?",        3),
    ("vektör veritabanları nelerdir", 3),
    ("LLM modelleri karşılaştırma",   3),
]
print(f"\n  {'Sorgu':<38} {'En Yüksek Skor':<15} {'Kaynak'}")
print("  " + "-" * 65)
for sorgu, k in test_aramalar:
    sonuclar = vdb.ara(sorgu, k)
    if sonuclar:
        en_iyi = sonuclar[0]
        print(f"  {sorgu:<38} {en_iyi['skor']:.4f}        {en_iyi['chunk']['doc_id']}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 3: FUNCTION CALLING — JSON ŞEMA TANIMI
# ─────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 3: Function Calling — JSON Şema Tanımı")
print("─" * 65)

FUNCTION_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": "Bilgi tabanından ilgili dökümanları çeker",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Arama sorgusu"},
                    "top_k": {"type": "integer", "description": "Döndürülecek chunk sayısı", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "İnternette güncel bilgi arar",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Arama terimi"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Matematiksel hesaplama yapar",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Matematiksel ifade"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_answer",
            "description": "Bağlam ve sorgudan yanıt üretir (RAG son adımı)",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {"type": "string"},
                    "question": {"type": "string"},
                    "style": {"type": "string", "enum": ["concise", "detailed", "bullet"]},
                },
                "required": ["context", "question"],
            },
        },
    },
]

print(f"  {'Fonksiyon Adı':<25} {'Zorunlu Parametre':<22} {'Açıklama'[:30]}")
print("  " + "-" * 75)
for schema in FUNCTION_SCHEMAS:
    fn  = schema["function"]
    req = ", ".join(fn["parameters"].get("required", []))
    print(f"  {fn['name']:<25} {req:<22} {fn['description'][:40]}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 4: TOOL ROUTER — OTOMATİK ARAÇ SEÇİMİ
# ─────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 4: ToolRouter — Otomatik Araç Seçimi")
print("─" * 65)

class ToolRouter:
    """Soru tipine göre araç zinciri belirler."""

    KURALLAR = [
        (r'\b(hesapla|topla|çarp|böl|\d+\s*[+\-*/]\s*\d+)\b', ["calculate"]),
        (r'\b(güncel|son dakika|bugün|haber|2024|2025)\b',      ["web_search", "generate_answer"]),
        (r'\b(nedir|ne|kim|nasıl|açıkla|tanımla)\b',            ["retrieve_documents", "generate_answer"]),
        (r'\b(karşılaştır|fark|hangisi|avantaj)\b',              ["retrieve_documents", "web_search", "generate_answer"]),
    ]

    def rota_belirle(self, soru: str) -> list:
        soru_l = soru.lower()
        for pattern, araclar in self.KURALLAR:
            if re.search(pattern, soru_l):
                return araclar
        return ["retrieve_documents", "generate_answer"]  # varsayılan

    def calistir(self, soru: str, vdb_ref) -> dict:
        araclar = self.rota_belirle(soru)
        sonuclar = {}

        for arac in araclar:
            if arac == "retrieve_documents":
                docs = vdb_ref.ara(soru, top_k=3)
                sonuclar["context"] = "\n".join(
                    d["chunk"]["metin"][:120] for d in docs
                )
                sonuclar["retrieved_docs"] = docs

            elif arac == "web_search":
                sonuclar["web"] = f"[Web] '{soru[:30]}...' araması tamamlandı"

            elif arac == "calculate":
                nums = re.findall(r'[\d.]+\s*[+\-*/]\s*[\d.]+', soru)
                if nums:
                    try: sonuclar["calc"] = f"{nums[0]} = {eval(nums[0]):.4f}"
                    except: sonuclar["calc"] = "Hesaplama hatası"

            elif arac == "generate_answer":
                ctx = sonuclar.get("context", "Bağlam yok")
                sonuclar["answer"] = f"[Yanıt] Bağlam analiz edildi. Soru: '{soru[:40]}...'"

        return {"soru": soru, "arac_zinciri": araclar, "sonuclar": sonuclar}

router = ToolRouter()
test_sorular = [
    "ReAct framework nedir ve nasıl çalışır?",
    "LLM ve RAG arasındaki fark nedir?",
    "125 * 8 / 4 + 50 hesapla",
    "Agentic AI güvenlik riskleri nelerdir?",
    "LangChain ile AutoGen karşılaştır",
]

print(f"  {'Soru':<45} {'Araç Zinciri'}")
print("  " + "-" * 75)
rota_sonuclari = []
for soru in test_sorular:
    sonuc = router.calistir(soru, vdb)
    rota_sonuclari.append(sonuc)
    zincir = " → ".join(sonuc["arac_zinciri"])
    print(f"  {soru[:43]:<45} {zincir}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 5: RAG AGENT
# ─────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 5: RAG Agent — Tam Pipeline")
print("─" * 65)

class RAGAgent:
    """RAG + Tool Use entegrasyonlu tam ajan."""

    def __init__(self, vdb: VektorDB, router: ToolRouter):
        self.vdb    = vdb
        self.router = router
        self.log    = []

    def _generate(self, sorgu: str, baglamlar: list) -> str:
        """Simüle yanıt üretimi."""
        kaynak_sayisi = len(baglamlar)
        dogruluk = min(0.60 + kaynak_sayisi * 0.12, 0.98)
        return (
            f"[RAG Yanıtı] '{sorgu[:40]}' için {kaynak_sayisi} kaynak "
            f"kullanıldı. Tahmini doğruluk: {dogruluk:.0%}"
        )

    def sor(self, sorgu: str, top_k: int = 3) -> dict:
        t0       = time.perf_counter()
        rota     = self.router.calistir(sorgu, self.vdb)
        belgeler = self.vdb.ara(sorgu, top_k=top_k)
        baglamlar= [b["chunk"]["metin"] for b in belgeler]
        yanit    = self._generate(sorgu, baglamlar)
        sure     = time.perf_counter() - t0

        kayit = {
            "sorgu":        sorgu,
            "arac_zinciri": rota["arac_zinciri"],
            "belgeler":     belgeler,
            "yanit":        yanit,
            "sure":         sure,
            "top_k":        top_k,
        }
        self.log.append(kayit)
        return kayit

rag_agent = RAGAgent(vdb, router)
print(f"  {'Sorgu':<45} {'Süre(ms)':<10} {'Kaynak'}")
print("  " + "-" * 65)
rag_log = []
for soru in test_sorular:
    k = rag_agent.sor(soru)
    rag_log.append(k)
    kaynaklar = ", ".join(set(b["chunk"]["doc_id"] for b in k["belgeler"]))
    print(f"  {soru[:43]:<45} {k['sure']*1000:.1f}ms    {kaynaklar[:25]}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 6: RAG KALİTE METRİKLERİ
# ─────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 6: RAG Kalite Metrikleri")
print("─" * 65)

def hesapla_precision(retrieved_docs: list, ilgili_doc: str) -> float:
    ilgili = sum(1 for d in retrieved_docs if ilgili_doc in d["chunk"]["doc_id"])
    return ilgili / len(retrieved_docs) if retrieved_docs else 0

def hesapla_recall(ilgili_toplam: int, bulunan: int) -> float:
    return bulunan / ilgili_toplam if ilgili_toplam else 0

def hesapla_mrr(retrieved_docs: list, ilgili_doc: str) -> float:
    for i, d in enumerate(retrieved_docs, 1):
        if ilgili_doc in d["chunk"]["doc_id"]:
            return 1.0 / i
    return 0.0

GT = {
    test_sorular[0]: "doc_agentic",
    test_sorular[1]: "doc_rag",
    test_sorular[2]: "doc_tools",
    test_sorular[3]: "doc_security",
    test_sorular[4]: "doc_frameworks",
}

print(f"  {'Soru (kısa)':<35} {'Precision':<12} {'Recall':<10} {'MRR':<10} {'NDCG'}")
print("  " + "-" * 75)
metrik_listesi = []
for kayit in rag_log:
    sorgu = kayit["sorgu"]
    gt    = GT.get(sorgu, "doc_agentic")
    prec  = hesapla_precision(kayit["belgeler"], gt)
    rec   = hesapla_recall(1, int(prec > 0))
    mrr   = hesapla_mrr(kayit["belgeler"], gt)
    ndcg  = round(random.uniform(0.55, 0.92), 3)  # simüle
    metrik_listesi.append({"precision": prec, "recall": rec, "mrr": mrr, "ndcg": ndcg})
    print(f"  {sorgu[:33]:<35} {prec:<12.3f} {rec:<10.3f} {mrr:<10.3f} {ndcg}")

ort_p = np.mean([m["precision"] for m in metrik_listesi])
ort_r = np.mean([m["recall"]    for m in metrik_listesi])
ort_m = np.mean([m["mrr"]       for m in metrik_listesi])
ort_n = np.mean([m["ndcg"]      for m in metrik_listesi])
print(f"  {'ORTALAMA':<35} {ort_p:<12.3f} {ort_r:<10.3f} {ort_m:<10.3f} {ort_n:.3f}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 7: GÖRSELLEŞTİRME (8 panel)
# ─────────────────────────────────────────────────────────────────
print()
print("─" * 65)
print("  BÖLÜM 7: Görselleştirme (8 panel)")
print("─" * 65)

plt.style.use("seaborn-v0_8-dark")
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#0D1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.36,
                        top=0.93, bottom=0.05)

PALETA = ["#1A6FD8","#0FBCCE","#7B52E8","#F5A623","#10C98F","#E879A0","#A78BFA","#FB923C"]

def ax_stili(ax, baslik):
    ax.set_facecolor("#161B22")
    ax.set_title(baslik, fontsize=11, fontweight="bold", color="#C9D1D9", pad=8)
    ax.tick_params(colors="#8B949E")
    ax.grid(alpha=0.25, color="#30363D")
    for sp in ax.spines.values(): sp.set_color("#30363D")

# ── G1: Chunk boyut dağılımı ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
boyutlar = [len(c["metin"].split()) for c in tum_chunks]
ax1.hist(boyutlar, bins=12, color=PALETA[0], edgecolor="#30363D", alpha=0.85)
ax_stili(ax1, "Chunk Boyut Dağılımı")
ax1.set_xlabel("Kelime Sayısı", fontsize=10, color="#8B949E")
ax1.set_ylabel("Frekans", fontsize=10, color="#8B949E")
ax1.axvline(np.mean(boyutlar), color="#F5A623", ls="--", lw=1.8, label=f"Ort={np.mean(boyutlar):.0f}")
ax1.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# ── G2: Döküman başına chunk sayısı ───────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
doc_chunk = Counter(c["doc_id"] for c in tum_chunks)
doc_isimler = [k.replace("doc_","") for k in doc_chunk.keys()]
ax2.bar(doc_isimler, list(doc_chunk.values()), color=PALETA[:len(doc_chunk)],
        edgecolor="#30363D", alpha=0.85)
ax_stili(ax2, "Döküman Başına Chunk Sayısı")
ax2.set_ylabel("Chunk", fontsize=10, color="#8B949E")
ax2.tick_params(axis="x", rotation=25)

# ── G3: Retrieval benzerlik skorları ─────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
tum_skorlar = []
for sorgu in test_sorular:
    docs = vdb.ara(sorgu, top_k=5)
    tum_skorlar.extend([d["skor"] for d in docs])
ax3.hist(tum_skorlar, bins=15, color=PALETA[1], edgecolor="#30363D", alpha=0.85)
ax_stili(ax3, "Retrieval Cosine Skor Dağılımı")
ax3.set_xlabel("Cosine Benzerlik", fontsize=10, color="#8B949E")
ax3.set_ylabel("Frekans", fontsize=10, color="#8B949E")
ax3.axvline(np.mean(tum_skorlar), color="#F5A623", ls="--", lw=1.8,
            label=f"Ort={np.mean(tum_skorlar):.3f}")
ax3.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# ── G4: RAG Pipeline akışı ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
ax4.set_facecolor("#161B22")
ax4.set_xlim(0, 10); ax4.set_ylim(0, 2.5); ax4.axis("off")
ax4.set_title("RAG Pipeline Akış Diyagramı", fontsize=11, fontweight="bold",
              color="#C9D1D9", pad=8)

adimlar_rag = [
    ("Sorgu\nGirişi",     "#1A6FD8"),
    ("Embedding\nÜretimi","#0FBCCE"),
    ("Vektör DB\nArama",  "#7B52E8"),
    ("Top-K\nChunk",      "#F5A623"),
    ("Bağlam\nBirleştir", "#10C98F"),
    ("LLM\nGenerasyon",   "#E879A0"),
    ("Yanıt\nÇıktısı",    "#A78BFA"),
]
xs = np.linspace(0.7, 9.3, len(adimlar_rag))
for j, (ad, renk) in enumerate(adimlar_rag):
    import matplotlib.patches as mpatches
    box = mpatches.FancyBboxPatch(
        (xs[j]-0.52, 0.9), 1.04, 0.9,
        boxstyle="round,pad=0.06", facecolor=renk,
        edgecolor="#0D1117", linewidth=2, alpha=0.88
    )
    ax4.add_patch(box)
    ax4.text(xs[j], 1.35, ad, ha="center", va="center",
             fontsize=9, color="white", fontweight="bold")
    if j < len(adimlar_rag) - 1:
        ax4.annotate("", xy=(xs[j+1]-0.55, 1.35), xytext=(xs[j]+0.55, 1.35),
                     arrowprops=dict(arrowstyle="->", color="#94A3B8",
                                     lw=1.8, mutation_scale=16))

# ── G5: Sorgu süreleri ────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
sorgu_sure = [k["sure"] * 1000 for k in rag_log]
sorgu_kisa = [f"S{i+1}" for i in range(len(rag_log))]
ax5.bar(sorgu_kisa, sorgu_sure, color=PALETA[2], edgecolor="#30363D", alpha=0.85)
ax_stili(ax5, "Sorgu Süreleri (ms)")
ax5.set_ylabel("Süre (ms)", fontsize=10, color="#8B949E")
ax5.axhline(np.mean(sorgu_sure), color="#F5A623", ls="--", lw=1.8,
            label=f"Ort={np.mean(sorgu_sure):.1f}ms")
ax5.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# ── G6: Retrieval kalite metrikleri ──────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
metrik_adlar = ["Precision", "Recall", "MRR", "NDCG"]
metrik_deger = [ort_p, ort_r, ort_m, ort_n]
renkler_m = [PALETA[0], PALETA[1], PALETA[2], PALETA[3]]
bars6 = ax6.bar(metrik_adlar, metrik_deger, color=renkler_m, edgecolor="#30363D", alpha=0.85)
ax_stili(ax6, "Retrieval Kalite Metrikleri\n(5 Sorgu Ortalaması)")
ax6.set_ylim(0, 1.1)
ax6.set_ylabel("Skor", fontsize=10, color="#8B949E")
for b, v in zip(bars6, metrik_deger):
    ax6.text(b.get_x()+b.get_width()/2, v+0.03, f"{v:.2f}",
             ha="center", va="bottom", fontsize=10, color="#C9D1D9")

# ── G7: Araç kullanım dağılımı ────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
arac_sayac = Counter()
for s in rota_sonuclari:
    for a in s["arac_zinciri"]:
        arac_sayac[a] += 1
wedge_colors = PALETA[:len(arac_sayac)]
wedges, texts, autos = ax7.pie(
    list(arac_sayac.values()),
    labels=list(arac_sayac.keys()),
    colors=wedge_colors,
    autopct="%1.0f%%",
    textprops={"fontsize":9, "color":"#C9D1D9"},
    wedgeprops={"edgecolor":"#0D1117","linewidth":2}
)
ax7.set_facecolor("#161B22")
ax7.set_title("Araç Kullanım Dağılımı\n(ToolRouter Çıktısı)", fontsize=11,
              fontweight="bold", color="#C9D1D9", pad=8)

# ── G8: top-k vs precision eğrisi ────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
topk_degerler = [1, 2, 3, 4, 5, 7, 10]
prec_egri = []
for k in topk_degerler:
    toplam_p = 0
    for sorgu in test_sorular:
        docs = vdb.ara(sorgu, top_k=k)
        gt   = GT.get(sorgu, "doc_agentic")
        toplam_p += hesapla_precision(docs, gt)
    prec_egri.append(toplam_p / len(test_sorular))

ax8.plot(topk_degerler, prec_egri, "o-", color="#0FBCCE", lw=2.5, markersize=8)
ax8.fill_between(topk_degerler, prec_egri, alpha=0.15, color="#0FBCCE")
ax_stili(ax8, "Top-K vs Precision@K Eğrisi")
ax8.set_xlabel("K (Döndürülen Chunk Sayısı)", fontsize=10, color="#8B949E")
ax8.set_ylabel("Precision@K", fontsize=10, color="#8B949E")
ax8.set_ylim(0, 1.05)

fig.suptitle(
    "AGENTİK AI — UYGULAMA 02  |  RAG Agent + Tool Use & Function Calling\n"
    "Vektör DB · TF-IDF Retrieval · JSON Schema · ToolRouter · Kalite Metrikleri",
    fontsize=13, fontweight="bold", color="#C9D1D9", y=0.98
)
plt.savefig("agentic_02_rag_agent.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("  ✅ agentic_02_rag_agent.png kaydedildi")
plt.close()

print()
print("=" * 65)
print("  ÖZET")
print(f"  Chunk sayısı    : {len(tum_chunks)}")
print(f"  Avg Precision   : {ort_p:.3f}")
print(f"  Avg Recall      : {ort_r:.3f}")
print(f"  Avg MRR         : {ort_m:.3f}")
print(f"  Function schemas: {len(FUNCTION_SCHEMAS)}")
print(f"  Grafik          : agentic_02_rag_agent.png")
print("  ✅ UYGULAMA 02 TAMAMLANDI")
print("=" * 65)
