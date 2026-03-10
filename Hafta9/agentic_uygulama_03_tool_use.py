"""
=============================================================================
AGENTİK AI — Tool Use & Function Calling
OpenAI uyumlu JSON Schema araç tanımları · Araç çağrısı döngüsü
=============================================================================
Kapsam:
  - OpenAI tools formatında JSON Schema araç tanımları
  - 8 araç: web_search, calculate, get_weather, run_code,
            read_file, translate, classify_text, extract_entities
  - ToolRegistry: merkezi araç kaydı + çağrı dispatcher
  - Paralel araç çağrısı (tool_use → parallel execution)
  - Araç çağrısı ayrıştırma, doğrulama, hata yönetimi
  - Konuşma geçmişiyle çok turlu araç döngüsü
  - Kapsamlı metrikler ve 8-panel görselleştirme
=============================================================================
"""

import re, time, json, random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings; warnings.filterwarnings("ignore")

print("=" * 65)
print("  AGENTİK AI — Tool Use & Function Calling")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 1: JSON SCHEMA ARAÇ TANIMLARI
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 1: JSON Schema Araç Tanımları (OpenAI Formatı)")
print("─" * 65)

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Web'de güncel bilgi arar ve en alakalı sonuçları döndürür.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "Arama sorgusu"},
                    "num_results": {"type": "integer", "description": "Sonuç sayısı (1-10)", "default": 3}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Matematiksel ifadeleri hesaplar. Temel aritmetik ve üs destekler.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Hesaplanacak ifade, ör: '(15 * 4) / 2 + 8'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Belirtilen şehir için güncel hava durumu bilgisi alır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":  {"type": "string", "description": "Şehir adı"},
                    "units": {"type": "string", "enum": ["celsius","fahrenheit"], "default": "celsius"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Güvenli sandbox ortamında Python kodu çalıştırır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code":    {"type": "string",  "description": "Çalıştırılacak Python kodu"},
                    "timeout": {"type": "integer", "description": "Maks süre (saniye)", "default": 5}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Dosya içeriğini okur ve döndürür.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Okunacak dosya adı"},
                    "encoding": {"type": "string", "default": "utf-8"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "Metni kaynak dilden hedef dile çevirir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":        {"type": "string", "description": "Çevrilecek metin"},
                    "source_lang": {"type": "string", "description": "Kaynak dil kodu, ör: 'tr'"},
                    "target_lang": {"type": "string", "description": "Hedef dil kodu, ör: 'en'"}
                },
                "required": ["text", "target_lang"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "classify_text",
            "description": "Metni verilen kategorilerden birine sınıflandırır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":       {"type": "string", "description": "Sınıflandırılacak metin"},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Olası kategori listesi"
                    }
                },
                "required": ["text", "categories"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": "Metinden kişi, yer, tarih, teknoloji varlıklarını çıkarır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":        {"type": "string", "description": "Analiz edilecek metin"},
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Çıkarılacak varlık türleri: PERSON, LOC, DATE, TECH"
                    }
                },
                "required": ["text"]
            }
        }
    },
]

for t in TOOLS_SCHEMA:
    fn = t["function"]
    req = fn["parameters"].get("required", [])
    props = list(fn["parameters"]["properties"].keys())
    print(f"  ▸ {fn['name']:<20} params: {props}   required: {req}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 2: ARAÇ KAYIT SİSTEMİ (ToolRegistry)
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 2: ToolRegistry — Merkezi Araç Kaydı")
print("─" * 65)

@dataclass
class ToolCagri:
    """Tek bir araç çağrısının kaydı."""
    arac_adi   : str
    parametreler: dict
    sonuc       : Any    = None
    hata        : str    = ""
    sure_ms     : float  = 0.0
    basarili    : bool   = True
    cagri_id    : str    = field(default_factory=lambda: f"call_{random.randint(100,999)}")
    zaman       : float  = field(default_factory=time.time)

class ToolRegistry:
    """
    Araçları merkezi olarak kaydeder, doğrular ve çalıştırır.
    OpenAI tool_calls formatını destekler.
    """
    def __init__(self):
        self._araclar  : Dict[str, Callable] = {}
        self._sema_map : Dict[str, dict]     = {}
        self.cagri_log : List[ToolCagri]     = []
        self.sayac      = defaultdict(int)

    def kaydet(self, schema: dict, func: Callable):
        name = schema["function"]["name"]
        self._araclar[name]  = func
        self._sema_map[name] = schema
        return self

    def _dogrula(self, name: str, params: dict) -> Optional[str]:
        """Parametreleri JSON Schema'ya göre doğrular."""
        if name not in self._araclar:
            return f"Araç bulunamadı: '{name}'"
        schema = self._sema_map[name]["function"]["parameters"]
        for req in schema.get("required", []):
            if req not in params:
                return f"Eksik parametre: '{req}'"
        return None  # OK

    def cagir(self, name: str, params: dict) -> ToolCagri:
        """Tek araç çağrısı — doğrula → çalıştır → kaydet."""
        kayit = ToolCagri(arac_adi=name, parametreler=params)
        hata = self._dogrula(name, params)
        if hata:
            kayit.hata, kayit.basarili = hata, False
            self.cagri_log.append(kayit)
            return kayit
        t0 = time.perf_counter()
        try:
            kayit.sonuc  = self._araclar[name](**params)
            kayit.basarili = True
        except Exception as e:
            kayit.hata, kayit.basarili = str(e), False
        kayit.sure_ms = (time.perf_counter() - t0) * 1000
        self.cagri_log.append(kayit)
        self.sayac[name] += 1
        return kayit

    def paralel_cagir(self, cagrilar: List[dict]) -> List[ToolCagri]:
        """
        Birden fazla araç çağrısını sıralı çalıştırır
        (gerçek paralel için ThreadPoolExecutor kullanılabilir).
        """
        return [self.cagir(c["name"], c.get("arguments", {})) for c in cagrilar]

    def to_openai_result(self, kayit: ToolCagri) -> dict:
        """OpenAI tool_result mesaj formatına dönüştür."""
        return {
            "role":         "tool",
            "tool_call_id": kayit.cagri_id,
            "name":         kayit.arac_adi,
            "content":      json.dumps(kayit.sonuc, ensure_ascii=False)
                            if kayit.basarili else f"HATA: {kayit.hata}"
        }


# ─────────────────────────────────────────────────────────────────
# BÖLÜM 3: ARAÇ FONKSİYONLARI
# ─────────────────────────────────────────────────────────────────
print("─" * 65)
print("  BÖLÜM 3: Araç Fonksiyonları")
print("─" * 65)

def web_search(query: str, num_results: int = 3) -> dict:
    time.sleep(random.uniform(0.02, 0.08))
    db = {
        "python":      "Python 3.12 çıktı. PEP 695 tip takma adları eklendi.",
        "transformer": "Attention Is All You Need — Vaswani et al. 2017, 100k+ atıf.",
        "langchain":   "LangChain v0.3 çıktı. LCEL ile zincir oluşturma kolaylaştı.",
        "agentic":     "Agentic AI 2024 en büyük trendi. AutoGPT, CrewAI popüler.",
        "yapay zeka":  "Global AI pazarı 2024: $184B. GPT-4o, Claude 3.5, Gemini öne çıktı.",
    }
    sonuclar = []
    for k, v in db.items():
        if k.lower() in query.lower():
            sonuclar.append({"title": k.title(), "snippet": v, "url": f"https://example.com/{k}"})
    if not sonuclar:
        sonuclar = [{"title": query[:30], "snippet": f"{query} hakkında bilgi bulundu.", "url": "https://example.com"}]
    return {"query": query, "results": sonuclar[:num_results], "total": len(sonuclar)}

def calculate(expression: str) -> dict:
    try:
        temiz = re.sub(r'[^0-9+\-*/().\s%]', '', expression)
        sonuc = eval(temiz)
        return {"expression": temiz, "result": sonuc, "ok": True}
    except ZeroDivisionError:
        return {"expression": expression, "result": None, "ok": False, "error": "Sıfıra bölme"}
    except Exception as e:
        return {"expression": expression, "result": None, "ok": False, "error": str(e)}

def get_weather(city: str, units: str = "celsius") -> dict:
    time.sleep(random.uniform(0.01, 0.05))
    hava = {
        "istanbul": {"temp": 18, "desc": "Parçalı bulutlu", "humidity": 72, "wind": 15},
        "ankara":   {"temp": 12, "desc": "Güneşli",         "humidity": 45, "wind": 22},
        "izmir":    {"temp": 24, "desc": "Açık",            "humidity": 58, "wind": 10},
        "london":   {"temp":  9, "desc": "Yağmurlu",        "humidity": 85, "wind": 28},
    }
    veriler = hava.get(city.lower(), {"temp": 16, "desc": "Parçalı bulutlu", "humidity": 60, "wind": 12})
    if units == "fahrenheit":
        veriler["temp"] = veriler["temp"] * 9/5 + 32
    return {"city": city, "units": units, **veriler, "timestamp": time.strftime("%Y-%m-%d %H:%M")}

def run_code(code: str, timeout: int = 5) -> dict:
    yasak = ["import os", "import sys", "open(", "__import__", "exec(", "eval("]
    for k in yasak:
        if k in code:
            return {"ok": False, "error": f"Güvenlik ihlali: '{k}' yasak", "output": ""}
    cikti = []
    def safe_print(*args): cikti.append(" ".join(str(a) for a in args))
    t0 = time.perf_counter()
    try:
        exec(code, {"print": safe_print, "range": range, "len": len,
                    "sum": sum, "max": max, "min": min, "sorted": sorted,
                    "list": list, "dict": dict, "str": str, "int": int,
                    "float": float, "round": round, "abs": abs})
        return {"ok": True, "output": "\n".join(cikti) or "(çıktı yok)",
                "exec_time_ms": (time.perf_counter()-t0)*1000}
    except Exception as e:
        return {"ok": False, "error": str(e), "output": "\n".join(cikti)}

def read_file(filename: str, encoding: str = "utf-8") -> dict:
    sanal_fs = {
        "veri.csv":     "id,isim,puan,kategori\n1,Ali,92,A\n2,Ayşe,87,B\n3,Mehmet,95,A",
        "config.json":  '{"model":"gpt-4","temperature":0.7,"max_tokens":2000}',
        "notlar.txt":   "Proje toplantı notları:\n- Teslim tarihi: 20 Mart\n- Sorumlu: Ekip A\n- Bütçe onaylandı",
        "README.md":    "# Proje\nBu proje agentic AI pipeline örneğidir.\n## Kurulum\npip install -r requirements.txt",
    }
    if filename in sanal_fs:
        icerik = sanal_fs[filename]
        return {"filename": filename, "content": icerik,
                "size_bytes": len(icerik), "ok": True}
    return {"filename": filename, "ok": False, "error": "Dosya bulunamadı"}

def translate(text: str, target_lang: str, source_lang: str = "auto") -> dict:
    time.sleep(random.uniform(0.02, 0.06))
    sozluk = {
        "en": {"merhaba": "hello", "teşekkür": "thank you", "yapay zeka": "artificial intelligence",
               "ajan": "agent", "araç": "tool", "model": "model"},
        "de": {"merhaba": "hallo", "teşekkür": "danke", "yapay zeka": "künstliche intelligenz"},
        "fr": {"merhaba": "bonjour", "teşekkür": "merci", "yapay zeka": "intelligence artificielle"},
    }
    if target_lang in sozluk:
        ceviri = text
        for tr, hedef in sozluk[target_lang].items():
            ceviri = ceviri.lower().replace(tr, hedef)
    else:
        ceviri = f"[{target_lang.upper()}] {text}"
    return {"original": text, "translated": ceviri,
            "source": source_lang, "target": target_lang, "ok": True}

def classify_text(text: str, categories: list) -> dict:
    anahtar_kelimeler = {
        "teknik":   ["kod","python","algoritma","model","api","fonksiyon","class"],
        "haber":    ["bugün","dün","açıkladı","gelişme","olay","rapor"],
        "soru":     ["nedir","nasıl","neden","ne zaman","kim","hangi","?"],
        "olumlu":   ["güzel","harika","mükemmel","başarı","teşekkür","iyi"],
        "olumsuz":  ["hata","sorun","başarısız","kötü","yanlış","problem"],
    }
    text_lower = text.lower()
    skorlar    = {}
    for cat in categories:
        skor = 0
        anahtar = anahtar_kelimeler.get(cat.lower(), [cat.lower()])
        for k in anahtar:
            if k in text_lower:
                skor += 1
        skorlar[cat] = skor + random.uniform(0, 0.3)
    en_iyi = max(skorlar, key=skorlar.get)
    return {"text": text[:50], "label": en_iyi,
            "scores": {k: round(v, 3) for k, v in skorlar.items()},
            "confidence": round(skorlar[en_iyi] / max(sum(skorlar.values()), 1), 3)}

def extract_entities(text: str, entity_types: list = None) -> dict:
    entity_types = entity_types or ["PERSON", "LOC", "DATE", "TECH"]
    patterns = {
        "PERSON": r'\b[A-ZÜĞİŞÖÇ][a-züğışöç]+\s+[A-ZÜĞİŞÖÇ][a-züğışöç]+\b',
        "LOC":    r'\b(İstanbul|Ankara|İzmir|Türkiye|London|Paris|Berlin|New York)\b',
        "DATE":   r'\b(20\d\d|[0-9]{1,2}\s*(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık))\b',
        "TECH":   r'\b(Python|LangChain|AutoGen|GPT|Claude|BERT|Transformer|RAG|FAISS|API|JSON|REST)\b',
    }
    bulunanlar = {}
    for tip in entity_types:
        if tip in patterns:
            bulunanlar[tip] = list(set(re.findall(patterns[tip], text)))
    return {"entities": bulunanlar, "total": sum(len(v) for v in bulunanlar.values())}

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 4: REGISTRY'YE KAYIT
# ─────────────────────────────────────────────────────────────────
print("  Araçlar registry'ye kaydediliyor...")

registry = ToolRegistry()
fonksiyonlar = [web_search, calculate, get_weather, run_code,
                read_file, translate, classify_text, extract_entities]
for schema, func in zip(TOOLS_SCHEMA, fonksiyonlar):
    registry.kaydet(schema, func)

print(f"  ✅ {len(registry._araclar)} araç kayıtlı: {list(registry._araclar.keys())}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 5: ÇOK TURLU ARAÇ DÖNGÜSÜ
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 5: Çok Turlu Araç Döngüsü (Konuşma Geçmişiyle)")
print("─" * 65)

@dataclass
class Konusma:
    """Araç çağrılarını içeren çok turlu konuşma."""
    soru     : str
    turlar   : List[dict] = field(default_factory=list)
    mesajlar : List[dict] = field(default_factory=list)

def ajan_dongusu(soru: str, verbose: bool = True) -> Konusma:
    """
    Soru → Araç seç → Çağır → Sonucu konuşmaya ekle → Tekrar değerlendir
    OpenAI tool_calls akışını simüle eder.
    """
    k = Konusma(soru=soru)
    k.mesajlar.append({"role": "user", "content": soru})

    # Araç seçim tablosu (simülatör)
    planlama = {
        "hava": [{"name":"get_weather","arguments":{"city":"istanbul"}}],
        "hesapla": [{"name":"calculate","arguments":{"expression":"150 * 12 / 4"}}],
        "ara": [{"name":"web_search","arguments":{"query":soru[:40]}}],
        "kod": [{"name":"run_code","arguments":{"code":"print(sum(range(1,101)))"}}],
        "dosya": [{"name":"read_file","arguments":{"filename":"veri.csv"}}],
        "çevir": [{"name":"translate","arguments":{"text":soru[:30],"target_lang":"en"}}],
        "sınıf": [{"name":"classify_text","arguments":{"text":soru,"categories":["teknik","haber","soru","olumlu","olumsuz"]}}],
        "varlık": [{"name":"extract_entities","arguments":{"text":soru}}],
    }

    # Paralel çağrı seçimi
    secilen = []
    for anahtar, cagrilar in planlama.items():
        if anahtar.lower() in soru.lower():
            secilen.extend(cagrilar)
    if not secilen:
        secilen = [{"name":"web_search","arguments":{"query":soru[:40]}}]

    if verbose: print(f"\n  🎯 SORU: {soru}")
    if verbose: print(f"  ⚙️  Seçilen araçlar: {[c['name'] for c in secilen]}")

    sonuclar = registry.paralel_cagir(secilen)

    for cagri, kayit in zip(secilen, sonuclar):
        k.mesajlar.append({
            "role": "assistant",
            "tool_calls": [{"id": kayit.cagri_id, "type": "function",
                            "function": {"name": kayit.arac_adi,
                                         "arguments": json.dumps(cagri.get("arguments",{}))}}]
        })
        k.mesajlar.append(registry.to_openai_result(kayit))
        durum = "✅" if kayit.basarili else "❌"
        if verbose:
            print(f"  {durum} {kayit.arac_adi:<22} → {str(kayit.sonuc)[:55]}  ({kayit.sure_ms:.1f}ms)")
        k.turlar.append({"arac": kayit.arac_adi, "basarili": kayit.basarili,
                          "sure_ms": kayit.sure_ms})

    nihai = f"Tüm araç çağrıları tamamlandı ({len(sonuclar)} araç, {sum(t['sure_ms'] for t in k.turlar):.1f}ms)"
    k.mesajlar.append({"role": "assistant", "content": nihai})
    if verbose: print(f"  📋 {nihai}")
    return k


SORULAR = [
    "İstanbul'un bugünkü hava durumu nedir?",
    "150 sayısının 12 ile çarpımını hesapla, sonra 4'e böl",
    "Python ile kod çalıştır: 1'den 100'e sayıları topla",
    "veri.csv dosyasını oku ve içeriğini göster",
    "Yapay zeka hakkında en son gelişmeleri ara",
    "Bu metni çevir: 'Merhaba dünya, yapay zeka çağındayız'",
    "Bu soruyu sınıflandır: 'Transformer nedir ve nasıl çalışır?'",
    "Python, LangChain ve GPT varlıklarını metinden çıkar: AutoGen ve LangChain 2024'te en popüler araçlardı.",
]

konusmalar = []
for soru in SORULAR:
    k = ajan_dongusu(soru, verbose=True)
    konusmalar.append(k)

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 6: METRİKLER
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 6: Çağrı Metrikleri")
print("─" * 65)

toplam_cagri  = len(registry.cagri_log)
basarili      = sum(1 for c in registry.cagri_log if c.basarili)
ortalama_sure = np.mean([c.sure_ms for c in registry.cagri_log]) if registry.cagri_log else 0

print(f"  Toplam çağrı   : {toplam_cagri}")
print(f"  Başarılı        : {basarili}  ({basarili/max(toplam_cagri,1)*100:.0f}%)")
print(f"  Ort. süre       : {ortalama_sure:.2f} ms")
print(f"\n  {'Araç':<24} {'Çağrı':>6} {'Ort ms':>9}")
print("  " + "─" * 42)
arac_sure = defaultdict(list)
for c in registry.cagri_log:
    arac_sure[c.arac_adi].append(c.sure_ms)
for ad, sureler in sorted(arac_sure.items(), key=lambda x: -len(x[1])):
    print(f"  {ad:<24} {len(sureler):>6} {np.mean(sureler):>8.2f}")

# ─────────────────────────────────────────────────────────────────
# BÖLÜM 7: GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  BÖLÜM 7: Görselleştirme (8 panel)")
print("─" * 65)

plt.style.use("seaborn-v0_8-dark")
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#0D1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.44, wspace=0.38,
                        top=0.93, bottom=0.05)

TOOL_RENK = {
    "web_search":"#3B82F6","calculate":"#10B981","get_weather":"#F59E0B",
    "run_code":"#A78BFA","read_file":"#F472B6","translate":"#34D399",
    "classify_text":"#FB923C","extract_entities":"#38BDF8"
}

# G1 — Araç çağrı frekansı
ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#161B22")
arac_sayilari = dict(registry.sayac)
adlar = list(arac_sayilari.keys()); sayilar = list(arac_sayilari.values())
renkler1 = [TOOL_RENK.get(a,"#64748B") for a in adlar]
ax1.barh(adlar, sayilar, color=renkler1, edgecolor="#30363D", alpha=0.88)
ax1.set_title("Araç Çağrı Frekansı", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax1.set_xlabel("Çağrı Sayısı", fontsize=10, color="#8B949E")
ax1.tick_params(colors="#8B949E")
ax1.grid(axis="x", alpha=0.3, color="#30363D")
ax1.set_facecolor("#161B22")
for sp in ax1.spines.values(): sp.set_color("#30363D")

# G2 — Araç süre dağılımı (kutu grafiği)
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#161B22")
arac_sureler_listesi = [arac_sure[a] for a in adlar if arac_sure[a]]
aktif_adlar = [a for a in adlar if arac_sure[a]]
bp = ax2.boxplot(arac_sureler_listesi, patch_artist=True, vert=True,
                 widths=0.6, medianprops=dict(color="white", linewidth=2))
for patch, ad in zip(bp["boxes"], aktif_adlar):
    patch.set_facecolor(TOOL_RENK.get(ad,"#64748B")); patch.set_alpha(0.8)
for comp in ["whiskers","caps","fliers"]:
    for el in bp[comp]: el.set_color("#8B949E")
ax2.set_xticks(range(1, len(aktif_adlar)+1))
ax2.set_xticklabels([a.replace("_","\n") for a in aktif_adlar],
                     fontsize=7.5, color="#C9D1D9", rotation=30, ha="right")
ax2.set_title("Araç Süre Dağılımı (ms)", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax2.set_ylabel("ms", fontsize=10, color="#8B949E")
ax2.tick_params(colors="#8B949E")
ax2.grid(axis="y", alpha=0.3, color="#30363D")
ax2.set_facecolor("#161B22")
for sp in ax2.spines.values(): sp.set_color("#30363D")

# G3 — Başarı / başarısızlık
ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor("#161B22")
bas_sayilari = defaultdict(lambda: [0,0])
for c in registry.cagri_log:
    bas_sayilari[c.arac_adi][0 if c.basarili else 1] += 1
adlar3 = list(bas_sayilari.keys())
bas   = [bas_sayilari[a][0] for a in adlar3]
basiz = [bas_sayilari[a][1] for a in adlar3]
x3 = np.arange(len(adlar3)); w3 = 0.38
ax3.bar(x3-w3/2, bas,   w3, label="Başarılı",   color="#22C55E", edgecolor="#30363D", alpha=0.88)
ax3.bar(x3+w3/2, basiz, w3, label="Başarısız",  color="#EF4444", edgecolor="#30363D", alpha=0.80)
ax3.set_xticks(x3)
ax3.set_xticklabels([a.replace("_","\n") for a in adlar3], fontsize=7.5,
                     color="#C9D1D9", rotation=30, ha="right")
ax3.set_title("Başarı / Başarısızlık", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax3.legend(fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")
ax3.tick_params(colors="#8B949E")
ax3.grid(axis="y", alpha=0.3, color="#30363D")
ax3.set_facecolor("#161B22")
for sp in ax3.spines.values(): sp.set_color("#30363D")

# G4 — OpenAI mesaj akışı (konuşma 1)
ax4 = fig.add_subplot(gs[1, :2]); ax4.set_facecolor("#161B22")
ax4.set_xlim(0,10); ax4.set_ylim(-0.5, 3.5); ax4.axis("off")
ax4.set_title("OpenAI Mesaj Akışı — Konuşma 1", fontsize=12,
              fontweight="bold", color="#C9D1D9", pad=8)
ROL_RENK = {"user":"#3B82F6","assistant":"#10B981","tool":"#F59E0B"}
k1_mesajlar = konusmalar[0].mesajlar[:7]
xs = np.linspace(0.7, 9.3, len(k1_mesajlar))
for j, msg in enumerate(k1_mesajlar):
    rol   = msg["role"]
    renk  = ROL_RENK.get(rol, "#64748B")
    icerik = msg.get("content") or (msg.get("tool_calls") and
             msg["tool_calls"][0]["function"]["name"]) or msg.get("name","")
    y = 2.0
    ax4.add_patch(mpatches.FancyBboxPatch((xs[j]-.58, y-.40), 1.16, 0.80,
        boxstyle="round,pad=0.06", facecolor=renk, edgecolor="#0D1117",
        linewidth=2, alpha=0.82))
    ax4.text(xs[j], y+0.18, rol.upper()[:9], ha="center", va="center",
             fontsize=8.5, color="white", fontweight="bold")
    label = str(icerik)[:16]+("…" if len(str(icerik))>16 else "")
    ax4.text(xs[j], y-0.18, label, ha="center", va="center", fontsize=7.5, color="#D1D5DB")
    if j < len(k1_mesajlar)-1:
        ax4.annotate("", xy=(xs[j+1]-.60, y), xytext=(xs[j]+.60, y),
                     arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=2, mutation_scale=16))
ax4.legend(handles=[mpatches.Patch(facecolor=v, label=k) for k,v in ROL_RENK.items()],
           loc="lower right", fontsize=9, labelcolor="#C9D1D9", facecolor="#161B22")

# G5 — Araç süre karşılaştırma
ax5 = fig.add_subplot(gs[1, 2]); ax5.set_facecolor("#161B22")
ort_sureler5 = [np.mean(arac_sure[a]) for a in adlar if arac_sure[a]]
adlar5 = [a for a in adlar if arac_sure[a]]
ax5.barh(adlar5, ort_sureler5, color=[TOOL_RENK.get(a,"#64748B") for a in adlar5],
         edgecolor="#30363D", alpha=0.88)
ax5.set_title("Ortalama Araç Süresi", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax5.set_xlabel("Süre (ms)", fontsize=10, color="#8B949E")
ax5.tick_params(colors="#8B949E")
ax5.grid(axis="x", alpha=0.3, color="#30363D")
ax5.set_facecolor("#161B22")
for sp in ax5.spines.values(): sp.set_color("#30363D")

# G6 — Soru başına araç sayısı
ax6 = fig.add_subplot(gs[2, 0]); ax6.set_facecolor("#161B22")
arac_sayisi_per_soru = [len(k.turlar) for k in konusmalar]
ax6.bar([f"S{i+1}" for i in range(len(konusmalar))], arac_sayisi_per_soru,
        color="#0FBCCE", edgecolor="#30363D", alpha=0.88)
ax6.set_title("Soru Başına\nAraç Sayısı", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax6.set_ylabel("Araç Sayısı", fontsize=10, color="#8B949E")
ax6.tick_params(colors="#8B949E")
ax6.grid(axis="y", alpha=0.3, color="#30363D")
ax6.set_facecolor("#161B22")
for sp in ax6.spines.values(): sp.set_color("#30363D")

# G7 — Araç türü pasta
ax7 = fig.add_subplot(gs[2, 1]); ax7.set_facecolor("#161B22")
labels7 = list(arac_sayilari.keys()); sizes7 = list(arac_sayilari.values())
colors7 = [TOOL_RENK.get(a,"#64748B") for a in labels7]
ax7.pie(sizes7, labels=[l.replace("_","\n") for l in labels7],
        colors=colors7, autopct="%1.0f%%", startangle=90,
        textprops={"fontsize":8,"color":"#C9D1D9"},
        wedgeprops={"edgecolor":"#0D1117","linewidth":2})
ax7.set_title("Araç Kullanım Dağılımı", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax7.set_facecolor("#161B22")

# G8 — Zaman çizelgesi
ax8 = fig.add_subplot(gs[2, 2]); ax8.set_facecolor("#161B22")
cagri_log = registry.cagri_log
zaman_serisi = np.cumsum([c.sure_ms for c in cagri_log])
renkler8 = [TOOL_RENK.get(c.arac_adi,"#64748B") for c in cagri_log]
for i, (t, c) in enumerate(zip(zaman_serisi, cagri_log)):
    ax8.scatter(t, c.sure_ms, color=renkler8[i], s=90,
                marker="o" if c.basarili else "x",
                edgecolors="#0D1117", linewidth=1.2, zorder=4)
ax8.set_title("Araç Çağrı Zaman\nÇizelgesi", fontsize=12, fontweight="bold", color="#C9D1D9", pad=8)
ax8.set_xlabel("Kümülatif Süre (ms)", fontsize=10, color="#8B949E")
ax8.set_ylabel("Çağrı Süresi (ms)", fontsize=10, color="#8B949E")
ax8.tick_params(colors="#8B949E")
ax8.grid(alpha=0.3, color="#30363D")
ax8.set_facecolor("#161B22")
for sp in ax8.spines.values(): sp.set_color("#30363D")

fig.suptitle(
    "AGENTİK AI — Tool Use & Function Calling\n"
    "JSON Schema · ToolRegistry · Paralel Çağrı · OpenAI Protokolü",
    fontsize=14, fontweight="bold", color="#C9D1D9", y=0.98
)
plt.savefig("agentic_tool_use.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("  ✅ agentic_tool_use.png kaydedildi")
plt.close()

print("\n" + "=" * 65)
print(f"  Araç sayısı      : {len(registry._araclar)}")
print(f"  Toplam çağrı     : {toplam_cagri}")
print(f"  Başarı oranı     : {basarili/max(toplam_cagri,1)*100:.0f}%")
print(f"  Ort. süre        : {ortalama_sure:.2f} ms")
print(f"  Grafik           : agentic_tool_use.png")
print("  ✅ Tool Use & Function Calling TAMAMLANDI")
print("=" * 65)
