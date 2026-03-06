"""
=============================================================================
UYGULAMA 02 — RAG Pipeline: Vektör Arama ve Soru-Cevap Sistemi
=============================================================================
Kapsam:
  - Belge parçalama: RecursiveCharacterTextSplitter (chunk + overlap ayarı)
  - Sentence-Transformers embedding modeli (BAAI/bge-small-en-v1.5)
  - FAISS ile vektör indeksleme ve k-NN araması
  - ChromaDB ile kalıcı vektör depolama
  - Cosine similarity görselleştirmesi (t-SNE embedding haritası)
  - Tam RAG pipeline: retrieve → prompt oluştur → LLM ile yanıt
  - Hallucination kontrolü: cevap-kaynak alıntı skoru
  - HuggingFace yoksa tam simülasyon modu

Kurulum: pip install sentence-transformers faiss-cpu chromadb numpy matplotlib scikit-learn
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import warnings
warnings.filterwarnings("ignore")

# ── Bağımlılık kontrolü ───────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

print("=" * 65)
print("  UYGULAMA 02 — RAG Pipeline")
print(f"  sentence-transformers : {'✅' if ST_AVAILABLE else '❌'}")
print(f"  FAISS                 : {'✅' if FAISS_AVAILABLE else '❌'}")
print(f"  ChromaDB              : {'✅' if CHROMA_AVAILABLE else '❌'}")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 1. BELGE KORPUSu
# ─────────────────────────────────────────────────────────────
print("\n[1] Belge korpusu hazırlanıyor...")

DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Transformer Mimarisi",
        "text": (
            "Transformer mimarisi, 2017 yılında Vaswani ve arkadaşları tarafından "
            "'Attention is All You Need' makalesiyle tanıtıldı. "
            "Mimari, self-attention mekanizmasını merkeze alarak RNN ve CNN'e olan "
            "bağımlılığı ortadan kaldırdı. Encoder-decoder yapısından oluşan orijinal "
            "Transformer, makine çevirisi görevinde devrim yarattı. "
            "Her encoder bloğu çok-başlı öz-dikkat (multi-head self-attention) ve "
            "ileri-beslemeli ağ (feed-forward network) katmanlarından oluşur. "
            "Artık kalıntı bağlantılar (residual connections) ve katman normalizasyonu "
            "eğitim stabilitesini artırmak için kullanılır."
        ),
        "source": "Vaswani et al., 2017"
    },
    {
        "id": "doc_002",
        "title": "BERT Modeli",
        "text": (
            "BERT (Bidirectional Encoder Representations from Transformers), Google tarafından "
            "2018 yılında yayınlandı. Yalnızca Transformer encoder katmanlarını kullanır ve "
            "çift yönlü bağlamı yakalar. İki ön-eğitim görevi kullanır: Maskeli Dil Modeli "
            "(MLM) ve Sonraki Cümle Tahmini (NSP). BERT-base modeli 12 encoder katmanı, "
            "12 dikkat kafası ve 768 boyutlu gizli durum içerir; toplam 110 milyon parametresi "
            "vardır. Fine-tuning sırasında CLS token'ının çıktısı sınıflandırma görevleri için "
            "kullanılır. RoBERTa, NSP'yi kaldırarak ve daha büyük veri ile BERT'i iyileştirdi."
        ),
        "source": "Devlin et al., 2018"
    },
    {
        "id": "doc_003",
        "title": "GPT ve Dil Modelleri",
        "text": (
            "GPT (Generative Pre-trained Transformer), OpenAI tarafından geliştirilen "
            "yalnızca-decoder mimarisine sahip otoregresif dil modelidir. Her token, yalnızca "
            "önceki tokenlara dikkat edebilir (causal masking). GPT-2 (2019), 1.5 milyar "
            "parametre ile sıfır-atış öğrenme kapasitesi sergiledi. GPT-3 (2020), 175 milyar "
            "parametre ile az-atış öğrenmede çığır açtı. InstructGPT, İnsan Geri Bildirimiyle "
            "Pekiştirmeli Öğrenme (RLHF) ile talimat takibini önemli ölçüde iyileştirdi. "
            "LLaMA ve Mistral, açık ağırlıklı alternatifler olarak ortaya çıktı."
        ),
        "source": "Brown et al., 2020"
    },
    {
        "id": "doc_004",
        "title": "LoRA Fine-Tuning",
        "text": (
            "LoRA (Low-Rank Adaptation), büyük dil modellerinin verimli ince ayarı için "
            "2021'de önerilen bir yöntemdir. Orijinal ağırlık matrisleri dondurulur ve "
            "her hedef matrise küçük rank-r matris çiftleri eklenir: ΔW = A·B. "
            "r tipik olarak 4 ile 64 arasındadır; eğitilebilir parametre sayısı modelin "
            "yalnızca yüzde 0.1-1'i kadardır. QLoRA, 4-bit kuantizasyon ile LoRA'yı "
            "birleştirerek büyük modelleri tek bir GPU'da ince ayar yapmayı mümkün kıldı. "
            "PEFT kütüphanesi, LoRA'yı Hugging Face modelleriyle sorunsuz entegre eder."
        ),
        "source": "Hu et al., 2021"
    },
    {
        "id": "doc_005",
        "title": "RAG: Bilgi Artırımlı Üretim",
        "text": (
            "RAG (Retrieval-Augmented Generation), dil modellerini dış bilgi tabanlarıyla "
            "güçlendiren bir mimaridir. İki aşamadan oluşur: indeksleme ve sorgulama. "
            "İndekleme aşamasında belgeler parçalanır, embedding'e dönüştürülür ve "
            "vektör veritabanına kaydedilir. Sorgulama aşamasında kullanıcı sorusu "
            "embedding'e çevrilir, en benzer belgeler bulunur ve LLM'e bağlam olarak verilir. "
            "RAG, parametrik bilgiye güvenmek yerine güncel bilgiye erişim sağlar ve "
            "hallüsinasyonları azaltır. Hibrit arama (vektör + BM25) performansı artırır."
        ),
        "source": "Lewis et al., 2020"
    },
    {
        "id": "doc_006",
        "title": "Vektör Veritabanları",
        "text": (
            "Vektör veritabanları, yüksek boyutlu vektörlerin depolanması ve hızlı "
            "benzerlik araması için tasarlanmıştır. FAISS (Facebook AI Similarity Search), "
            "milyarlarca vektör üzerinde milisaniyeler içinde arama yapabilir. "
            "ChromaDB, kalıcı disk depolama ve metadata filtreleme sunar. "
            "Pinecone, yönetilen bulut tabanlı bir çözümdür. Benzerlik metrikleri arasında "
            "kosinüs benzerliği, iç çarpım ve L2 mesafesi yer alır. "
            "Approximate Nearest Neighbor (ANN) algoritmaları, tam arama yerine hız-doğruluk "
            "dengesi sunar; HNSW ve IVF bu algoritmalara örnektir."
        ),
        "source": "Johnson et al., 2017"
    },
    {
        "id": "doc_007",
        "title": "LangChain Çerçevesi",
        "text": (
            "LangChain, LLM tabanlı uygulamalar geliştirmek için kapsamlı bir Python "
            "çerçevesidir. LCEL (LangChain Expression Language), prompt | llm | parser "
            "şeklinde zincirleme sözdizimi sunar. Agent'lar, LLM'in hangi aracı ne zaman "
            "kullanacağına otomatik olarak karar vermesini sağlar. Memory modülleri, "
            "konuşma geçmişini prompt'a enjekte eder. Tool'lar; web arama, Python REPL, "
            "SQL ve özel API çağrılarını kapsar. LangGraph, DAG tabanlı çok-adımlı "
            "iş akışları oluşturmayı mümkün kılar."
        ),
        "source": "LangChain Docs, 2023"
    },
    {
        "id": "doc_008",
        "title": "LLM Değerlendirme Metrikleri",
        "text": (
            "LLM'lerin değerlendirilmesi birden fazla metrik gerektirir. BLEU, n-gram "
            "örtüşmesini ölçerken özellikle makine çevirisi için kullanılır. ROUGE-L, "
            "en uzun ortak alt-diziyi ölçerek özetleme görevleri için uygundur. "
            "BERTScore, üretilen ve referans metinlerin gömme benzerliğini hesaplar. "
            "Perplexity, dil modelinin kalitesini ölçen temel bir metriktir. "
            "LLM-as-Judge yaklaşımında güçlü bir model (GPT-4 gibi) çıktıları değerlendirir. "
            "MT-Bench ve MMLU, standart kıyaslama paketleridir. Hallüsinasyon tespiti "
            "için self-consistency ve FActScore kullanılır."
        ),
        "source": "Chang et al., 2023"
    },
]

# Metin parçalama fonksiyonu
def recursive_text_splitter(text, chunk_size=200, chunk_overlap=40):
    """
    Özyinelemeli karakter tabanlı metin parçalayıcı.
    Önce paragraflara, sonra cümlelere, sonra kelimelere göre böler.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []

    def split_recursive(text, separators):
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []
        sep = separators[0] if separators else ""
        parts = text.split(sep) if sep else list(text)
        good_splits, current = [], ""
        for part in parts:
            if len(current) + len(part) + len(sep) <= chunk_size:
                current += (sep if current else "") + part
            else:
                if current:
                    good_splits.append(current)
                current = part
        if current:
            good_splits.append(current)
        # Overlap ekle
        result = []
        for i, chunk in enumerate(good_splits):
            if i > 0 and chunk_overlap > 0:
                prev_words = good_splits[i-1].split()[-chunk_overlap//5:]
                chunk = " ".join(prev_words) + " " + chunk
            result.append(chunk.strip())
        return result

    return split_recursive(text, separators)

# Belgeleri parçala
all_chunks = []
for doc in DOCUMENTS:
    chunks = recursive_text_splitter(doc["text"], chunk_size=250, chunk_overlap=50)
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "chunk_id": f"{doc['id']}_chunk_{i}",
            "doc_id":   doc["id"],
            "title":    doc["title"],
            "text":     chunk,
            "source":   doc["source"],
        })

print(f"    Belge sayısı : {len(DOCUMENTS)}")
print(f"    Toplam parça : {len(all_chunks)}")
print(f"    Ort. parça uzunluğu: {np.mean([len(c['text']) for c in all_chunks]):.0f} karakter")

# ─────────────────────────────────────────────────────────────
# 2. EMBEDDING ÜRETİMİ
# ─────────────────────────────────────────────────────────────
print("\n[2] Embedding'ler üretiliyor...")

EMBED_DIM = 384   # bge-small boyutu

if ST_AVAILABLE:
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    print(f"    Model: {MODEL_NAME}")
    embed_model = SentenceTransformer(MODEL_NAME)

    t0 = time.time()
    chunk_texts  = [c["text"] for c in all_chunks]
    embeddings   = embed_model.encode(
        chunk_texts,
        normalize_embeddings=True,   # Cosine similarity için normalize et
        show_progress_bar=True,
        batch_size=32,
    )
    t_embed = time.time() - t0
    EMBED_DIM = embeddings.shape[1]
    print(f"    Embedding boyutu : {EMBED_DIM}")
    print(f"    Embedding süresi : {t_embed:.2f}s")
else:
    print("    [SİMÜLASYON] sentence-transformers yok, rastgele embedding...")
    np.random.seed(42)
    embeddings = np.random.randn(len(all_chunks), EMBED_DIM).astype(np.float32)
    # Normalize et (cosine sim için)
    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

embeddings = np.array(embeddings, dtype=np.float32)
print(f"    Embedding matrisi : {embeddings.shape}")

# ─────────────────────────────────────────────────────────────
# 3. FAISS İNDEKSİ
# ─────────────────────────────────────────────────────────────
print("\n[3] FAISS indeksi oluşturuluyor...")

if FAISS_AVAILABLE:
    # IndexFlatIP: İç çarpım (normalize vektörlerde = cosine similarity)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    print(f"    FAISS IndexFlatIP: {index.ntotal} vektör eklendi")

    def faiss_search(query_emb, k=4):
        """FAISS ile k-NN araması."""
        q = np.array([query_emb], dtype=np.float32)
        scores, indices = index.search(q, k)
        return scores[0], indices[0]
else:
    print("    [SİMÜLASYON] FAISS yok — NumPy cosine search...")

    def faiss_search(query_emb, k=4):
        """Brute-force cosine similarity (normalize edilmiş)."""
        scores  = embeddings @ query_emb
        indices = np.argsort(scores)[::-1][:k]
        return scores[indices], indices

# ─────────────────────────────────────────────────────────────
# 4. CHROMA DB
# ─────────────────────────────────────────────────────────────
if CHROMA_AVAILABLE:
    print("\n[4] ChromaDB koleksiyonu oluşturuluyor...")
    chroma_client = chromadb.Client()   # Bellek içi (persist için: chromadb.PersistentClient)
    collection = chroma_client.create_collection(
        name="llm_bilgi_tabani",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids        = [c["chunk_id"] for c in all_chunks],
        embeddings = embeddings.tolist(),
        documents  = [c["text"] for c in all_chunks],
        metadatas  = [{"title": c["title"], "source": c["source"]} for c in all_chunks],
    )
    print(f"    ChromaDB: {collection.count()} döküman eklendi")

    def chroma_search(query_emb, k=4):
        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=k,
        )
        return results
else:
    print("\n[4] ChromaDB yok — FAISS ile devam ediliyor.")

# ─────────────────────────────────────────────────────────────
# 5. SORU SORGUSU & RETRIEVAL
# ─────────────────────────────────────────────────────────────
print("\n[5] Sorgular test ediliyor...")

TEST_QUERIES = [
    "Transformer mimarisinin temel bileşenleri nelerdir?",
    "LoRA fine-tuning nasıl çalışır ve ne kadar parametre kullanır?",
    "BERT ile GPT arasındaki temel farklar nelerdir?",
    "RAG sistemi hangi adımlardan oluşur?",
    "Vektör veritabanı için hangi benzerlik metrikleri kullanılabilir?",
]

retrieval_results = {}

for query in TEST_QUERIES:
    print(f"\n    ❓ Sorgu: '{query[:60]}...'")

    # Query embedding
    if ST_AVAILABLE:
        q_emb = embed_model.encode(
            [query], normalize_embeddings=True
        )[0].astype(np.float32)
    else:
        # Simülasyon: Belirli bir belgeyle yüksek benzerlik
        q_emb = embeddings[0].copy()
        q_emb += np.random.randn(EMBED_DIM).astype(np.float32) * 0.3
        q_emb /= np.linalg.norm(q_emb)

    # FAISS arama
    scores, idx_list = faiss_search(q_emb, k=3)

    retrieved = []
    for score, idx in zip(scores, idx_list):
        chunk = all_chunks[idx]
        retrieved.append({
            "chunk": chunk,
            "score": float(score),
        })
        print(f"      [{score:.3f}] {chunk['title']} — {chunk['text'][:60]}...")

    retrieval_results[query] = {
        "q_emb": q_emb,
        "retrieved": retrieved,
        "scores": scores.tolist(),
    }

# ─────────────────────────────────────────────────────────────
# 6. RAG PROMPT & YANITLAMA
# ─────────────────────────────────────────────────────────────
print("\n[6] RAG prompt oluşturma ve yanıt üretme...")

RAG_SYSTEM_PROMPT = """Sen yardımcı bir asistansın. Yalnızca aşağıdaki bağlamı kullanarak soruyu yanıtla.
Bağlamda olmayan bilgileri kesinlikle ekleme. Yanıtının sonunda hangi kaynakları kullandığını belirt.

BAĞLAM:
{context}

SORU: {question}

YANIT:"""

def build_rag_prompt(query, retrieved_chunks, max_context_chars=1200):
    """Alınan parçalardan RAG prompt'u oluştur."""
    context_parts = []
    total_chars   = 0
    sources_used  = []

    for item in retrieved_chunks:
        chunk  = item["chunk"]
        text   = chunk["text"]
        if total_chars + len(text) > max_context_chars:
            break
        context_parts.append(f"[{chunk['title']}]\n{text}")
        total_chars += len(text)
        if chunk["source"] not in sources_used:
            sources_used.append(chunk["source"])

    context = "\n\n".join(context_parts)
    prompt  = RAG_SYSTEM_PROMPT.format(context=context, question=query)
    return prompt, sources_used

def simulate_llm_response(query, retrieved):
    """
    LLM yanıtını simüle et — gerçek LLM entegrasyonu için
    OpenAI/HuggingFace API kullanılabilir.
    """
    top_chunk = retrieved[0]["chunk"] if retrieved else None
    if not top_chunk:
        return "Bilgi bulunamadı.", []

    # Sorguya göre basit yanıt simülasyonu
    sources = list({r["chunk"]["source"] for r in retrieved})
    title   = top_chunk["title"]

    responses = {
        "Transformer": f"{title} hakkında: Transformer mimarisi 2017'de tanıtılmış olup encoder-decoder yapısını kullanır. Multi-head self-attention temel bileşendir.",
        "LoRA":        f"{title} hakkında: LoRA, orijinal ağırlıkları dondurarak ΔW = A·B ile düşük ranklı güncellemeler ekler. Parametrelerin yalnızca %0.1-1'i eğitilir.",
        "BERT":        f"{title} hakkında: BERT yalnızca encoder kullanır ve çift yönlü bağlamı MLM ile öğrenir. GPT ise yalnızca decoder ile otoregresif üretim yapar.",
        "RAG":         f"{title} hakkında: RAG iki aşamadan oluşur: belge indeksleme (embedding + vektör DB) ve sorgulama (retrieve + generate).",
        "benzerlik":   f"{title} hakkında: Kosinüs benzerliği, iç çarpım ve L2 mesafesi yaygın metriklerdir. Normalize vektörlerde cosine = dot product.",
    }

    for keyword, resp in responses.items():
        if keyword.lower() in query.lower():
            return resp, sources

    return f"Bağlamdaki bilgiye göre: {top_chunk['text'][:150]}...", sources

# Yanıt simülasyonları
rag_outputs = {}
for query, res in retrieval_results.items():
    prompt, sources = build_rag_prompt(query, res["retrieved"])
    answer, used_sources = simulate_llm_response(query, res["retrieved"])

    # Hallucination skoru: cevabın bağlamda geçen kelimeleri
    context_words = set(" ".join([r["chunk"]["text"] for r in res["retrieved"]]).lower().split())
    answer_words  = set(answer.lower().split())
    grounding_score = len(answer_words & context_words) / max(len(answer_words), 1)

    rag_outputs[query] = {
        "answer":          answer,
        "sources":         used_sources,
        "grounding_score": grounding_score,
        "top_score":       res["scores"][0],
    }
    print(f"\n    Sorgu: {query[:50]}...")
    print(f"    Yanıt: {answer[:100]}...")
    print(f"    Grounding skoru: {grounding_score:.3f}  | Retrieval skoru: {res['scores'][0]:.3f}")

# ─────────────────────────────────────────────────────────────
# 7. t-SNE EMBEDDİNG HARİTASI
# ─────────────────────────────────────────────────────────────
print("\n[7] t-SNE görselleştirmesi hazırlanıyor...")

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# Belge başlıklarına göre renk
doc_labels = [c["title"] for c in all_chunks]
le         = LabelEncoder()
label_ids  = le.fit_transform(doc_labels)

# Query embedding'lerini de ekle
query_embs = np.array([retrieval_results[q]["q_emb"] for q in TEST_QUERIES], dtype=np.float32)
all_embs   = np.vstack([embeddings, query_embs])

# t-SNE (2D)
n_perplexity = min(15, len(all_embs) - 1)
tsne = TSNE(n_components=2, perplexity=n_perplexity, random_state=42, n_iter=1000)
coords = tsne.fit_transform(all_embs)

chunk_coords = coords[:len(all_chunks)]
query_coords = coords[len(all_chunks):]

# ─────────────────────────────────────────────────────────────
# 8. GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────
PALETTE = [
    "#6D28D9","#059669","#0284C7","#E11D48",
    "#D97706","#0F766E","#4338CA","#DC2626",
]
DOC_COLORS = {title: PALETTE[i % len(PALETTE)] for i, title in enumerate(le.classes_)}

fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)
fig.suptitle("RAG Pipeline — Vektör Arama & Soru-Cevap Analizi", fontsize=15, fontweight="bold")

# ── 8a. t-SNE Embedding Haritası ─────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for i, (title, color) in enumerate(DOC_COLORS.items()):
    mask = np.array(doc_labels) == title
    ax1.scatter(chunk_coords[mask, 0], chunk_coords[mask, 1],
                c=color, s=120, alpha=0.85, label=title[:25], zorder=4)
# Sorgu noktaları
for i, (qc, query) in enumerate(zip(query_coords, TEST_QUERIES)):
    ax1.scatter(qc[0], qc[1], c="black", marker="*", s=300, zorder=6)
    ax1.annotate(f"S{i+1}", (qc[0], qc[1]),
                 textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold")
ax1.set_title("t-SNE Embedding Haritası (★=Sorgular)", fontweight="bold")
ax1.legend(fontsize=8, loc="upper right", ncol=2)
ax1.grid(alpha=0.2); ax1.set_xlabel("t-SNE 1"); ax1.set_ylabel("t-SNE 2")

# ── 8b. Retrieval Skoru Bar ───────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
top_scores = [retrieval_results[q]["scores"][0] for q in TEST_QUERIES]
query_labels_short = [f"S{i+1}" for i in range(len(TEST_QUERIES))]
bars2 = ax2.bar(query_labels_short, top_scores, color=PALETTE[:len(TEST_QUERIES)], alpha=0.85)
for bar, v in zip(bars2, top_scores):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
ax2.set_title("En Yüksek Retrieval Skoru (Sorgu Başına)", fontweight="bold")
ax2.set_ylabel("Cosine Similarity"); ax2.grid(axis="y", alpha=0.3)

# ── 8c. Parça boyutu dağılımı ─────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
chunk_lens = [len(c["text"]) for c in all_chunks]
ax3.hist(chunk_lens, bins=15, color="#6D28D9", alpha=0.8, edgecolor="white")
ax3.axvline(np.mean(chunk_lens), color="#E11D48", ls="--", lw=2,
            label=f"Ort={np.mean(chunk_lens):.0f}")
ax3.set_title("Parça Uzunluğu Dağılımı", fontweight="bold")
ax3.set_xlabel("Karakter Sayısı"); ax3.set_ylabel("Frekans")
ax3.legend(); ax3.grid(axis="y", alpha=0.3)

# ── 8d. Grounding skoru ──────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
grounding = [rag_outputs[q]["grounding_score"] for q in TEST_QUERIES]
bars4 = ax4.barh(query_labels_short, grounding,
                  color=["#059669" if g > 0.3 else "#E11D48" for g in grounding], alpha=0.85)
for bar, v in zip(bars4, grounding):
    ax4.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
             f"{v:.3f}", va="center", fontsize=10, fontweight="bold")
ax4.axvline(0.3, color="gray", ls="--", lw=1.5, label="Eşik=0.30")
ax4.set_title("Grounding Skoru (Bağlam-Yanıt Örtüşmesi)", fontweight="bold")
ax4.set_xlabel("Örtüşme Oranı"); ax4.legend(); ax4.grid(axis="x", alpha=0.3)

# ── 8e. Cosine benzerlik matrisi (Belge başlıkları arası) ─────
ax5 = fig.add_subplot(gs[1, 2])
# Her belgenin ilk chunk embedding'ini al
doc_embs = {}
for chunk, emb in zip(all_chunks, embeddings):
    if chunk["doc_id"] not in doc_embs:
        doc_embs[chunk["doc_id"]] = emb
doc_ids     = [d["id"] for d in DOCUMENTS]
doc_titles_short = [d["title"][:12] for d in DOCUMENTS]
E  = np.array([doc_embs[d] for d in doc_ids])
CM = E @ E.T   # Cosine similarity (normalize edilmiş)
im5 = ax5.imshow(CM, cmap="Purples", vmin=0, vmax=1)
plt.colorbar(im5, ax=ax5)
ax5.set_xticks(range(len(doc_ids))); ax5.set_yticks(range(len(doc_ids)))
ax5.set_xticklabels(doc_titles_short, rotation=45, ha="right", fontsize=8)
ax5.set_yticklabels(doc_titles_short, fontsize=8)
ax5.set_title("Belge Cosine Benzerlik Matrisi", fontweight="bold")
for i in range(len(doc_ids)):
    for j in range(len(doc_ids)):
        ax5.text(j, i, f"{CM[i,j]:.2f}", ha="center", va="center",
                 fontsize=7, color="white" if CM[i,j] > 0.6 else "black")

# ── 8f. Retrieval skoru dağılımı (Tüm parçalar, ilk sorgu) ───
ax6 = fig.add_subplot(gs[2, 0])
q0      = TEST_QUERIES[0]
q0_emb  = retrieval_results[q0]["q_emb"]
all_sim = embeddings @ q0_emb
ax6.hist(all_sim, bins=20, color="#6D28D9", alpha=0.75, edgecolor="white")
top_s, top_i = faiss_search(q0_emb, k=3)
for s in top_s:
    ax6.axvline(s, color="#E11D48", ls="--", lw=2, alpha=0.8)
ax6.set_title(f"Benzerlik Dağılımı — Sorgu 1\n(kırmızı=top-3)", fontweight="bold")
ax6.set_xlabel("Cosine Similarity"); ax6.set_ylabel("Frekans")
ax6.grid(axis="y", alpha=0.3)

# ── 8g. RAG pipeline akış özeti ──────────────────────────────
ax7 = fig.add_subplot(gs[2, 1:])
ax7.axis("off")
ax7.set_title("RAG Pipeline Adım Özeti", fontweight="bold")

steps = [
    ("1. Belge Yükleme",     f"{len(DOCUMENTS)} belge, {len(all_chunks)} parça",  "#6D28D9"),
    ("2. Embedding",          f"{EMBED_DIM}D vektör, {'BGE' if ST_AVAILABLE else 'Simülasyon'}",       "#059669"),
    ("3. FAISS İndeksi",      f"{len(all_chunks)} vektör, IndexFlatIP (cosine)",  "#0284C7"),
    ("4. Sorgulama (top-3)",  f"Ort. top-1 skor: {np.mean(top_scores):.3f}",    "#D97706"),
    ("5. Prompt Oluşturma",   "Bağlam enjeksiyonu, kaynak atıfı",               "#E11D48"),
    ("6. Grounding Kontrolü", f"Ort. örtüşme: {np.mean(grounding):.3f}",        "#0F766E"),
]
for i, (step, detail, color) in enumerate(steps):
    y = 0.92 - i * 0.15
    ax7.add_patch(plt.Rectangle((0.01, y-0.1), 0.44, 0.12,
                                 transform=ax7.transAxes,
                                 facecolor=color, alpha=0.85, zorder=3))
    ax7.text(0.03, y-0.02, step, transform=ax7.transAxes,
             fontsize=11.5, color="white", fontweight="bold", va="center")
    ax7.text(0.49, y-0.02, detail, transform=ax7.transAxes,
             fontsize=11, color="#2E1065", va="center")

plt.savefig("02_rag_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ 02_rag_analiz.png kaydedildi")
plt.close()

print("\n" + "=" * 65)
print(f"  Belge sayısı   : {len(DOCUMENTS)}")
print(f"  Parça sayısı   : {len(all_chunks)}")
print(f"  Embedding dim  : {EMBED_DIM}")
print(f"  Ort. top-1 sim : {np.mean(top_scores):.4f}")
print(f"  Ort. grounding : {np.mean(grounding):.4f}")
print("  ✅ UYGULAMA 02 TAMAMLANDI — 02_rag_analiz.png")
print("=" * 65)
