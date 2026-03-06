"""
=============================================================================
UYGULAMA 03 — LangChain: Zincir, Agent, Bellek ve Araçlar
=============================================================================
Kapsam:
  - LCEL (LangChain Expression Language): prompt | llm | parser
  - Üç farklı görev zinciri (özetleme, çeviri, soru-cevap)
  - ConversationChain: çok dönüşlü diyalog belleği
  - ReAct Agent: düşün → araç seç → gözlemle → yanıtla döngüsü
  - Özel Tool dekoratörü ile alan-özgü araç yazma
  - Prompt template: FewShotPromptTemplate ile örnekli istem
  - LLM yoksa tam simülasyon modu (yerel model veya API anahtarı gerektirmez)

Kurulum: pip install langchain langchain-community langchain-openai numpy matplotlib
NOT: Gerçek LLM için OPENAI_API_KEY env değişkeni gerekir.
     Simülasyon modu API anahtarı olmadan çalışır.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import json
import re
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# ── Bağımlılık kontrolü ───────────────────────────────────────
try:
    from langchain_core.prompts import (
        ChatPromptTemplate, PromptTemplate,
        FewShotPromptTemplate, MessagesPlaceholder,
    )
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    LC_AVAILABLE = True
except ImportError:
    LC_AVAILABLE = False

try:
    import openai, os
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        OPENAI_AVAILABLE = True
    else:
        OPENAI_AVAILABLE = False
except ImportError:
    OPENAI_AVAILABLE = False

SIMULATION_MODE = not (LC_AVAILABLE and OPENAI_AVAILABLE)

print("=" * 65)
print("  UYGULAMA 03 — LangChain: Zincir, Agent, Bellek")
print(f"  LangChain Core : {'✅' if LC_AVAILABLE else '❌ pip install langchain'}")
print(f"  OpenAI API     : {'✅' if OPENAI_AVAILABLE else '❌ OPENAI_API_KEY yok'}")
print(f"  Çalışma Modu   : {'🤖 Gerçek LLM' if not SIMULATION_MODE else '🎭 Simülasyon'}")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# BÖLÜM 1: PROMPT TEMPLATE'LER
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 1: Prompt Template Türleri")
print("═" * 55)

# 1a. Basit PromptTemplate
SIMPLE_TEMPLATE = """Aşağıdaki metni {dil} diline çevir:

Metin: {metin}

Çeviri:"""

# 1b. ChatPromptTemplate (sistem + kullanıcı)
CHAT_TEMPLATE_STR = """Sen bir {uzmanlik} uzmanısın.
Teknik terimleri açık ve anlaşılır biçimde Türkçe açıkla."""

# 1c. FewShotPromptTemplate
FEW_SHOT_EXAMPLES = [
    {"sozu": "İki kafadar", "anlam": "Birbirine çok yakın, her zaman birlikte olan iki kişi."},
    {"sozu": "Taşı gediğine koymak", "anlam": "Doğru şeyi doğru yerde söylemek veya yapmak."},
    {"sozu": "El elden üstündür", "anlam": "Her güçlünün üzerinde daha güçlü biri vardır."},
]
EXAMPLE_TEMPLATE = "Söz: {sozu}\nAnlam: {anlam}"
EXAMPLE_PROMPT   = PromptTemplate(
    input_variables=["sozu", "anlam"],
    template=EXAMPLE_TEMPLATE,
) if LC_AVAILABLE else None

if LC_AVAILABLE:
    few_shot_prompt = FewShotPromptTemplate(
        examples=FEW_SHOT_EXAMPLES,
        example_prompt=EXAMPLE_PROMPT,
        prefix="Türkçe atasözü ve deyimlerin anlamlarını açıkla:\n",
        suffix="\nSöz: {sozu}\nAnlam:",
        input_variables=["sozu"],
    )
    test_fs = few_shot_prompt.format(sozu="Üzüm üzüme baka baka kararır")
    print(f"\n  FewShot Prompt örneği:\n{test_fs[:300]}")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 2: LCEL ZİNCİRLERİ
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 2: LCEL Zincirleri (prompt | llm | parser)")
print("═" * 55)

# Görevler
ZINCIR_GOREVLERI = [
    {
        "ad": "Özetleme",
        "sistem": "Metni 2-3 cümleyle özetle. Türkçe yanıt ver.",
        "girdi_key": "metin",
        "girdi": (
            "Transformer mimarisi, 2017'de Vaswani ve arkadaşları tarafından tanıtıldı. "
            "RNN ve CNN'e olan bağımlılığı kıran self-attention mekanizması kullanır. "
            "BERT, GPT ve T5 gibi büyük dil modelleri bu mimari üzerine inşa edilmiştir."
        ),
        "sim_yanit": "Transformer mimarisi, 2017'de tanıtılan ve self-attention mekanizmasına dayanan bir yapıdır. Bu mimari, RNN ve CNN'e olan bağımlılığı ortadan kaldırmıştır. BERT, GPT ve T5 gibi önemli modeller bu mimari üzerine geliştirilmiştir.",
    },
    {
        "ad": "Duygu Analizi (JSON)",
        "sistem": "Metni analiz et. JSON formatında yanıt ver: {\"duygu\": \"...\", \"skor\": 0.0-1.0, \"anahtar_kelimeler\": [...]}",
        "girdi_key": "metin",
        "girdi": "Bu kurs inanılmaz derecede faydalıydı! Transformer'ları ve LoRA'yı çok iyi öğrendim.",
        "sim_yanit": '{"duygu": "çok olumlu", "skor": 0.95, "anahtar_kelimeler": ["inanılmaz", "faydalıydı", "çok iyi"]}',
    },
    {
        "ad": "Soru-Cevap",
        "sistem": "Verilen bağlama dayanarak soruyu yanıtla. Bağlamda olmayan bilgi ekleme.",
        "girdi_key": "soru",
        "girdi": "BERT modelinin kaç parametresi vardır?",
        "baglam": "BERT-base: 110M parametre, 12 katman. BERT-large: 340M parametre, 24 katman.",
        "sim_yanit": "BERT-base modeli 110 milyon parametreye ve 12 katmana sahipken, BERT-large modeli 340 milyon parametre ve 24 katmana sahiptir.",
    },
]

zincir_sonuclari = []
latency_records  = []

for gorev in ZINCIR_GOREVLERI:
    print(f"\n  [{gorev['ad']}] Zinciri çalıştırılıyor...")

    if not SIMULATION_MODE:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=300)

        if "baglam" in gorev:
            prompt_tmpl = ChatPromptTemplate.from_messages([
                ("system", gorev["sistem"]),
                ("human", "Bağlam: {baglam}\n\nSoru: {soru}"),
            ])
            chain = prompt_tmpl | llm | StrOutputParser()
            t0     = time.time()
            yanit  = chain.invoke({"baglam": gorev["baglam"], "soru": gorev["girdi"]})
        else:
            prompt_tmpl = ChatPromptTemplate.from_messages([
                ("system", gorev["sistem"]),
                ("human",  "{" + gorev["girdi_key"] + "}"),
            ])
            chain = prompt_tmpl | llm | StrOutputParser()
            t0    = time.time()
            yanit = chain.invoke({gorev["girdi_key"]: gorev["girdi"]})
        latency = time.time() - t0
    else:
        # Simülasyon
        time.sleep(0.05)
        yanit   = gorev["sim_yanit"]
        latency = np.random.uniform(0.8, 2.5)

    zincir_sonuclari.append({
        "ad":      gorev["ad"],
        "girdi":   gorev["girdi"][:80],
        "yanit":   yanit,
        "latency": latency,
    })
    latency_records.append(latency)
    print(f"    Girdi : {gorev['girdi'][:70]}...")
    print(f"    Yanıt : {str(yanit)[:120]}...")
    print(f"    Gecikme: {latency:.2f}s")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 3: KONUŞMA BELLEĞİ
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 3: Konuşma Belleği (Conversation Memory)")
print("═" * 55)

KONUSMA = [
    {"insan": "Merhaba! Bana yapay zeka hakkında bilgi ver.",
     "ai":    "Merhaba! Yapay zeka, bilgisayarların insan benzeri zekâ sergileyebildiği sistemler bütünüdür. Derin öğrenme, doğal dil işleme ve bilgisayarla görme ana alt alanlarıdır."},
    {"insan": "Peki derin öğrenme nedir?",
     "ai":    "Derin öğrenme, çok katmanlı yapay sinir ağlarını kullanan makine öğrenmesi alt dalıdır. Az önce bahsettiğim yapay zekanın en güçlü dallarından biridir. CNN, RNN ve Transformer gibi mimariler bu alana girer."},
    {"insan": "Transformer hakkında daha fazla anlat.",
     "ai":    "Transformer, 2017'de tanıtılan ve 'Attention is All You Need' makalesiyle duyurulan bir mimaridir. Az önce bahsettiğim derin öğrenmede devrim yarattı. BERT ve GPT gibi büyük dil modelleri bu mimari üzerine kurulmuştur."},
    {"insan": "Şu ana kadar hangi konuları ele aldık?",
     "ai":    "Konuşmamızda sırasıyla şu konuları ele aldık: (1) Yapay zeka ve alt alanları, (2) Derin öğrenme ve ana mimarileri, (3) Transformer mimarisi ve önemi. Bellek sayesinde tüm bu bağlamı takip edebiliyorum."},
]

# Bellek simülasyonu: token sayacı
class ConversationBuffer:
    """Konuşma geçmişini tutan basit bellek."""
    def __init__(self, max_token_limit=500):
        self.history   = []
        self.max_limit = max_token_limit

    def add_turn(self, human_msg: str, ai_msg: str):
        self.history.append({"human": human_msg, "ai": ai_msg})
        # Token limit — eski dönüşleri at
        while self._token_count() > self.max_limit and len(self.history) > 1:
            self.history.pop(0)

    def _token_count(self):
        return sum(len(t["human"].split()) + len(t["ai"].split()) for t in self.history)

    def get_context(self):
        context = []
        for turn in self.history:
            context.append(f"İnsan: {turn['human']}")
            context.append(f"Asistan: {turn['ai']}")
        return "\n".join(context)

    def __len__(self):
        return len(self.history)

memory_log = []
buf = ConversationBuffer(max_token_limit=300)

for turn in KONUSMA:
    if not SIMULATION_MODE:
        if LC_AVAILABLE:
            msgs = [SystemMessage(content="Sen yardımcı bir asistansın. Türkçe yanıt ver.")]
            for h in buf.history:
                msgs.append(HumanMessage(content=h["human"]))
                msgs.append(AIMessage(content=h["ai"]))
            msgs.append(HumanMessage(content=turn["insan"]))
            llm_mem = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=200)
            yanit   = llm_mem.invoke(msgs).content
        else:
            yanit = turn["ai"]
    else:
        yanit = turn["ai"]

    buf.add_turn(turn["insan"], yanit)
    token_cnt = buf._token_count()
    memory_log.append({
        "tur":     len(memory_log) + 1,
        "insan":   turn["insan"],
        "yanit":   yanit,
        "tokens":  token_cnt,
        "tur_say": len(buf),
    })
    print(f"\n  Tur {len(memory_log)}: {turn['insan'][:60]}...")
    print(f"    Yanıt : {yanit[:100]}...")
    print(f"    Bellek: {token_cnt} token, {len(buf)} dönüş")

# ─────────────────────────────────────────────────────────────
# BÖLÜM 4: ARAÇLAR VE AGENT
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  BÖLÜM 4: Araçlar (Tools) ve ReAct Agent")
print("═" * 55)

# Özel araçlar
class KHesaplama:
    """Matematiksel hesaplama aracı."""
    name        = "hesaplama"
    description = "Matematiksel işlemler için. Örnek: '2 + 3 * 4'"

    def run(self, expression: str) -> str:
        try:
            # Güvenli eval (sadece sayı ve operatörler)
            safe_expr = re.sub(r"[^0-9+\-*/().\s]", "", expression)
            result    = eval(safe_expr)
            return f"Sonuç: {result}"
        except Exception as e:
            return f"Hata: {str(e)}"

class KSozluk:
    """Terim sözlüğü aracı."""
    name        = "sozluk"
    description = "Yapay zeka terimlerinin tanımları için."

    TERMS = {
        "transformer": "Self-attention tabanlı derin öğrenme mimarisi (Vaswani et al., 2017).",
        "lora":        "Low-Rank Adaptation; ön-eğitimli modellerin verimli ince ayarı yöntemi.",
        "rag":         "Retrieval-Augmented Generation; vektör arama ile desteklenmiş LLM üretimi.",
        "embedding":   "Metni sayısal vektörlere dönüştüren temsil öğrenme yöntemi.",
        "attention":   "Her token'ın diğer token'lara ne kadar odaklanacağını belirleyen mekanizma.",
        "bert":        "Google'ın çift yönlü encoder-only Transformer modeli (2018, 110M-340M param).",
        "gpt":         "OpenAI'ın tek yönlü decoder-only otoregresif dil modeli ailesi.",
        "langchain":   "LLM uygulamaları geliştirmek için Python çerçevesi.",
    }

    def run(self, term: str) -> str:
        key = term.lower().strip()
        if key in self.TERMS:
            return f"{term}: {self.TERMS[key]}"
        return f"'{term}' terimi sözlükte bulunamadı."

class KHesapMakinesi:
    """İstatistik hesaplama aracı."""
    name        = "istatistik"
    description = "Sayı listesinin istatistiklerini hesaplar. Örnek: '[1, 2, 3, 4, 5]'"

    def run(self, numbers_str: str) -> str:
        try:
            numbers = json.loads(numbers_str)
            arr     = np.array(numbers, dtype=float)
            return json.dumps({
                "ortalama": round(float(arr.mean()), 4),
                "std":      round(float(arr.std()), 4),
                "min":      round(float(arr.min()), 4),
                "max":      round(float(arr.max()), 4),
                "medyan":   round(float(np.median(arr)), 4),
            }, ensure_ascii=False)
        except Exception as e:
            return f"Hata: {str(e)}"

TOOLS = {
    "hesaplama": KHesaplama(),
    "sozluk":    KSozluk(),
    "istatistik": KHesapMakinesi(),
}

# ReAct Agent simülasyonu
class ReActAgent:
    """
    ReAct (Reasoning + Acting) döngüsü:
    Düşün → Araç Seç → Gözlemle → Yanıtla
    """
    def __init__(self, tools: dict, max_steps: int = 5):
        self.tools     = tools
        self.max_steps = max_steps

    def _select_tool(self, thought: str) -> Optional[str]:
        """Basit kural tabanlı araç seçimi."""
        thought_lower = thought.lower()
        if any(kw in thought_lower for kw in ["hesapla", "topla", "çarp", "böl", "+", "-", "*", "/"]):
            return "hesaplama"
        if any(kw in thought_lower for kw in ["nedir", "tanım", "terim", "ne demek"]):
            return "sozluk"
        if any(kw in thought_lower for kw in ["istatistik", "ortalama", "std", "sayı listesi"]):
            return "istatistik"
        return None

    def run(self, query: str, sim_steps: Optional[List[Dict]] = None) -> Dict:
        """Agent döngüsü."""
        if sim_steps:
            return {"steps": sim_steps, "final_answer": sim_steps[-1].get("answer", "?")}

        steps   = []
        thought = query

        for step_n in range(1, self.max_steps + 1):
            tool_name = self._select_tool(thought)
            if not tool_name:
                answer = f"Soruyu doğrudan yanıtlıyorum: {query}"
                steps.append({"step": step_n, "thought": thought, "action": "final_answer",
                               "observation": answer, "answer": answer})
                break

            tool        = self.tools[tool_name]
            observation = tool.run(query)
            steps.append({
                "step":        step_n,
                "thought":     f"Bu soruyu yanıtlamak için '{tool_name}' aracını kullanmalıyım.",
                "action":      tool_name,
                "action_input": query,
                "observation": observation,
            })

            if len(steps) >= 2 or tool_name in ["sozluk", "hesaplama", "istatistik"]:
                final = f"Araç sonucuna göre yanıt: {observation}"
                steps.append({"step": step_n + 1, "thought": "Gözlemden yeterli bilgiyi aldım.",
                               "action": "final_answer", "observation": final, "answer": final})
                break

        return {"steps": steps, "final_answer": steps[-1].get("answer", "Yanıt bulunamadı.")}

agent = ReActAgent(TOOLS, max_steps=5)

AGENT_SORULARI = [
    {
        "query": "transformer nedir?",
        "sim_steps": [
            {"step": 1, "thought": "Bu soru bir terim tanımı gerektiriyor, 'sozluk' aracını kullanmalıyım.",
             "action": "sozluk", "action_input": "transformer",
             "observation": KSozluk().run("transformer")},
            {"step": 2, "thought": "Gözlemden yeterli bilgiyi aldım.",
             "action": "final_answer",
             "observation": f"Araç sonucuna göre: {KSozluk().run('transformer')}",
             "answer": f"Araç sonucuna göre: {KSozluk().run('transformer')}"},
        ],
    },
    {
        "query": "15 * 8 + 32 / 4 hesapla",
        "sim_steps": [
            {"step": 1, "thought": "Matematiksel bir işlem var, 'hesaplama' aracını kullanmalıyım.",
             "action": "hesaplama", "action_input": "15 * 8 + 32 / 4",
             "observation": KHesaplama().run("15 * 8 + 32 / 4")},
            {"step": 2, "action": "final_answer",
             "observation": KHesaplama().run("15 * 8 + 32 / 4"),
             "answer": KHesaplama().run("15 * 8 + 32 / 4")},
        ],
    },
    {
        "query": "[0.92, 0.87, 0.95, 0.78, 0.91, 0.88] istatistik",
        "sim_steps": [
            {"step": 1, "thought": "Sayı listesi istatistiği gerekiyor.",
             "action": "istatistik", "action_input": "[0.92, 0.87, 0.95, 0.78, 0.91, 0.88]",
             "observation": KHesapMakinesi().run("[0.92, 0.87, 0.95, 0.78, 0.91, 0.88]")},
            {"step": 2, "action": "final_answer",
             "observation": KHesapMakinesi().run("[0.92, 0.87, 0.95, 0.78, 0.91, 0.88]"),
             "answer": KHesapMakinesi().run("[0.92, 0.87, 0.95, 0.78, 0.91, 0.88]")},
        ],
    },
]

agent_logs = []
for task in AGENT_SORULARI:
    print(f"\n  ❓ Sorgu: '{task['query']}'")
    result = agent.run(task["query"], sim_steps=task["sim_steps"])
    agent_logs.append({"query": task["query"], "result": result})
    for step in result["steps"]:
        emoji = {"sozluk":"📖","hesaplama":"🔢","istatistik":"📊","final_answer":"✅"}.get(step.get("action",""), "🔄")
        print(f"    {emoji} Adım {step['step']}: {step.get('action','?')} → {str(step.get('observation',''))[:80]}")

# ─────────────────────────────────────────────────────────────
# GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────
print("\n  Görselleştirmeler hazırlanıyor...")

fig = plt.figure(figsize=(22, 16))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)
fig.suptitle("LangChain — Zincir, Agent, Bellek ve Araçlar", fontsize=15, fontweight="bold")

PALETTE = ["#6D28D9", "#059669", "#0284C7", "#E11D48", "#D97706"]

# ── a. Zincir gecikmeleri ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
chain_names = [r["ad"] for r in zincir_sonuclari]
chain_lat   = [r["latency"] for r in zincir_sonuclari]
bars1       = ax1.bar(range(len(chain_names)), chain_lat,
                       color=PALETTE[:len(chain_names)], alpha=0.85)
for bar, v in zip(bars1, chain_lat):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f"{v:.2f}s", ha="center", fontsize=11, fontweight="bold")
ax1.set_xticks(range(len(chain_names)))
ax1.set_xticklabels([n[:12] for n in chain_names], rotation=15, fontsize=9)
ax1.set_title("LCEL Zinciri — Görev Gecikmeleri", fontweight="bold")
ax1.set_ylabel("Gecikme (saniye)"); ax1.grid(axis="y", alpha=0.3)

# ── b. Bellek token büyümesi ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
tur_nums = [m["tur"] for m in memory_log]
tokens   = [m["tokens"] for m in memory_log]
ax2.plot(tur_nums, tokens, "o-", lw=2.5, color="#6D28D9", markersize=10)
ax2.axhline(buf.max_limit, color="#E11D48", ls="--", lw=2, label=f"Limit={buf.max_limit}")
ax2.fill_between(tur_nums, tokens, alpha=0.15, color="#6D28D9")
for x, y in zip(tur_nums, tokens):
    ax2.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=10, fontweight="bold")
ax2.set_title("Konuşma Belleği — Token Büyümesi", fontweight="bold")
ax2.set_xlabel("Konuşma Turu"); ax2.set_ylabel("Token Sayısı")
ax2.legend(); ax2.grid(alpha=0.3)

# ── c. ReAct agent adım görselleştirme ───────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis("off")
ax3.set_title("ReAct Agent Döngüsü", fontweight="bold")

steps_schema = [
    ("🤔 Düşün",   "LLM sorguyu analiz eder,\nhangi araç gerekli?",   "#6D28D9"),
    ("🔧 Araç Seç","Uygun aracı ve girdiyi\nbelirle",                   "#059669"),
    ("⚡ Çalıştır","Araç çağrısı yap,\nsonucu bekle",                   "#0284C7"),
    ("👁️ Gözlemle", "Araç çıktısını incele,\nyeterli mi?",               "#D97706"),
    ("✅ Yanıtla", "Gözlemlerden\nnihai yanıtı üret",                   "#059669"),
]
for i, (step, desc, color) in enumerate(steps_schema):
    y = 0.92 - i * 0.18
    ax3.add_patch(plt.Rectangle((0.02, y-0.14), 0.96, 0.13,
                                 facecolor=color, alpha=0.85,
                                 transform=ax3.transAxes))
    ax3.text(0.06, y-0.055, step, transform=ax3.transAxes,
             fontsize=12, color="white", fontweight="bold", va="center")
    ax3.text(0.45, y-0.055, desc, transform=ax3.transAxes,
             fontsize=10.5, color="white", va="center")
    if i < len(steps_schema) - 1:
        ax3.annotate("", xy=(0.5, y-0.15), xytext=(0.5, y-0.145),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color="gray", lw=2))

# ── d. Araç kullanım dağılımı ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
tool_use = {}
for log in agent_logs:
    for step in log["result"]["steps"]:
        action = step.get("action", "")
        if action != "final_answer":
            tool_use[action] = tool_use.get(action, 0) + 1
if tool_use:
    ax4.pie(list(tool_use.values()), labels=list(tool_use.keys()),
            colors=PALETTE[:len(tool_use)], autopct="%1.0f%%",
            startangle=140, textprops={"fontsize": 11})
    ax4.set_title("Araç Kullanım Dağılımı", fontweight="bold")
else:
    ax4.text(0.5, 0.5, "Veri yok", ha="center", va="center", fontsize=14, color="gray")
    ax4.axis("off")

# ── e. LCEL zinciri akış diyagramı ───────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.axis("off")
ax5.set_title("LCEL Zincir Mimarisi", fontweight="bold")

lcel_nodes = [
    ("PromptTemplate",    "#6D28D9", "Girdi değişkenlerini\nbiçimlendirme"),
    ("ChatOpenAI / LLM",  "#059669", "Dil modeli\nçıkarımı"),
    ("OutputParser",      "#0284C7", "Ham çıktıyı\nstr/JSON/listeye dönüştür"),
    ("Uygulama / Kullanıcı", "#E11D48", "Son kullanıcıya\nteslim"),
]
for i, (node, color, desc) in enumerate(lcel_nodes):
    y = 0.88 - i * 0.22
    ax5.add_patch(plt.Rectangle((0.05, y-0.17), 0.9, 0.16,
                                 facecolor=color, alpha=0.85,
                                 transform=ax5.transAxes))
    ax5.text(0.5, y-0.085, node, transform=ax5.transAxes,
             fontsize=12, color="white", fontweight="bold",
             ha="center", va="center")
    ax5.text(0.5, y-0.145, desc, transform=ax5.transAxes,
             fontsize=9.5, color="#F5F3FF", ha="center", va="center")
    if i < len(lcel_nodes) - 1:
        pipe = "  |  (pipe operatörü)"
        ax5.text(0.5, y-0.19, pipe, transform=ax5.transAxes,
                 fontsize=9, color="#6D28D9", ha="center",
                 fontweight="bold", va="top")

# ── f. Özet kutu ─────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
ax6.set_title("LangChain Kavram Özeti", fontweight="bold")

summary = (
    "LCEL Sözdizimi:\n"
    "  chain = prompt | llm | parser\n"
    "  chain.invoke({'key': 'değer'})\n"
    "  chain.stream(...)  # Akış için\n\n"
    "Memory Türleri:\n"
    "  ConversationBufferMemory  → Tümünü sakla\n"
    "  ConversationSummaryMemory → Özetleyerek sakla\n"
    "  EntityMemory              → Varlık bazlı\n\n"
    "Agent Türleri:\n"
    "  ReAct                     → Düşün+Araç\n"
    "  OpenAI Functions          → Fonksiyon çağrısı\n"
    "  Structured Chat           → Çok araçlı\n\n"
    f"Bu oturumda:\n"
    f"  Zincir sayısı  : {len(zincir_sonuclari)}\n"
    f"  Konuşma turu   : {len(memory_log)}\n"
    f"  Agent araç say.: {len(TOOLS)}\n"
    f"  Agent görevi   : {len(agent_logs)}\n"
    f"  Mod            : {'Gerçek LLM' if not SIMULATION_MODE else 'Simülasyon'}"
)
ax6.text(0.03, 0.97, summary, transform=ax6.transAxes,
         fontsize=10.5, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F3FF",
                   edgecolor="#6D28D9", linewidth=1.5))

plt.savefig("03_langchain_analiz.png", dpi=150, bbox_inches="tight")
print("    ✅ 03_langchain_analiz.png kaydedildi")
plt.close()

print("\n" + "=" * 65)
print(f"  Zincir sayısı  : {len(zincir_sonuclari)}")
print(f"  Konuşma turları: {len(memory_log)}")
print(f"  Agent görevleri: {len(agent_logs)}")
print("  ✅ UYGULAMA 03 TAMAMLANDI — 03_langchain_analiz.png")
print("=" * 65)
