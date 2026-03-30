"""
HAGA Folder RAG — Server
=========================
FastAPI backend met RAG-pipeline: query → vector search → LLM generatie.
Serveert ook de chat-frontend.

Gebruik:
    python server.py
"""

import io
import os
import re
from pathlib import Path
from contextlib import asynccontextmanager

import chromadb
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

load_dotenv()

# ─────────────────────────────────────────────
# TTS CONFIG
# ─────────────────────────────────────────────

TTS_ENABLED = os.getenv("TTS_ENABLED", "true").lower() == "true"

# Pad naar ONNX model en voice bestanden (kokoro-onnx)
TTS_MODEL_PATH = os.getenv("TTS_MODEL_PATH", "./kokoro-v1.0.onnx")
TTS_VOICES_PATH = os.getenv("TTS_VOICES_PATH", "./voices-v1.0.bin")

# Taalconfiguratie voor Kokoro-ONNX
# voice: stem-ID, lang: taalcode voor kokoro-onnx create()
TTS_LANG_CONFIG = {
    "nl": {"voice": "af_heart", "lang": "nl",    "speed": 0.95, "label": "Nederlands"},
    "en": {"voice": "af_heart", "lang": "en-us",  "speed": 1.0,  "label": "English"},
    "tr": {"voice": "af_heart", "lang": "tr",     "speed": 0.95, "label": "Türkçe"},
    "ar": {"voice": "af_heart", "lang": "ar",     "speed": 0.90, "label": "العربية"},
    "fr": {"voice": "ff_siwis", "lang": "fr-fr",  "speed": 1.0,  "label": "Français"},
}

# Singleton Kokoro-ONNX instance (geladen bij eerste gebruik)
_kokoro_instance = None

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "haga_folders")
TOP_K = int(os.getenv("TOP_K", "15"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))  # min cosine similarity
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))  # aantal beurten (1 beurt = 1 user + 1 assistant)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# ── Confidence tiers (feature: gelaagd antwoordgedrag) ──
CONFIDENCE_HIGH = float(os.getenv("CONFIDENCE_HIGH", "0.65"))     # Direct antwoord
CONFIDENCE_MEDIUM = float(os.getenv("CONFIDENCE_MEDIUM", "0.45")) # Antwoord + caveat
# Onder CONFIDENCE_MEDIUM → disambiguation of zachte verduidelijking

# ── Disambiguation (feature: folder-keuze bij ambigue vragen) ──
DISAMBIGUATION_MAX_OPTIONS = int(os.getenv("DISAMBIGUATION_MAX_OPTIONS", "4"))
DISAMBIGUATION_MIN_FOLDERS = int(os.getenv("DISAMBIGUATION_MIN_FOLDERS", "2"))

# ─────────────────────────────────────────────
# VEILIGHEIDSCONSTANTEN
# ─────────────────────────────────────────────

# Crisis-signalen — altijd pre-processing, VOOR RAG
# Bij match: stuur direct crisisantwoord terug, nooit RAG aanroepen
CRISIS_SIGNALS = [
    "wil niet meer leven", "wil dood", "wil mezelf iets aandoen",
    "suicide", "suïcide", "zelfmoord", "mezelf van het leven beroven",
    "leven beëindigen", "niet meer verder willen", "nergens meer zin in",
    "wil er niet meer zijn", "suicidale", "suïcidale",
]

CRISIS_RESPONSE = """Het spijt me dat u zich zo voelt. U bent niet alleen.

Bel direct met een van deze hulplijnen:
• **113 Zelfmoordpreventie** — 113 (24/7 bereikbaar)
• **Huisarts of huisartsenpost** — uw eerste aanspreekpunt
• **112** — bij direct gevaar

Als u op dit moment in het ziekenhuis bent, spreek dan direct een medewerker aan.

Ik ben een informatiesysteem over patiëntenfolders en kan u hier niet de hulp bieden die u verdient. Neem alstublieft contact op met een van bovenstaande hulplijnen."""

# Dosering-signaalwoorden — bij aanwezigheid in antwoord: blokkeer en vervang
# Let op: alleen blokkeren als de *vraag* over dosering gaat (zie is_dosage_question())
# Anders worden bijwerkingenvragen onterecht geblokkeerd (bijv. "10% van patiënten")
DOSAGE_PATTERNS = [
    r"\d+\s*mg", r"\d+\s*ml", r"\d+\s*tablet", r"\d+\s*keer per dag",
    r"\d+\s*x per dag", r"\d+\s*maal", r"maximaal \d+", r"\d+ tot \d+ mg",
]

# Signaalwoorden die wijzen op een expliciete doseringsvraag
DOSAGE_QUESTION_KEYWORDS = [
    "hoeveel", "hoe veel", "dosis", "dosering", "hoeveelheid",
    "mg", "ml", "tablet", "pillen", "pil", "capsule",
    "keer per dag", "maal per dag", "per dag innemen",
]

DOSAGE_BLOCK_RESPONSE = (
    "De folder bevat informatie over dit medicijn, maar ik geef geen doseringsadvies. "
    "Dosering is persoonlijk en afhankelijk van uw situatie. "
    "Raadpleeg uw behandelend arts, apotheker of de bijsluiter voor de juiste dosering."
)


def is_dosage_question(message: str) -> bool:
    """
    Detecteer of de vraag expliciet over dosering gaat.

    Alleen dan blokkeren we een antwoord dat doseringscijfers bevat.
    Vragen over bijwerkingen, procedure of voorbereiding bevatten
    soms getallen (percentages, tijden) die we NIET willen blokkeren.
    """
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in DOSAGE_QUESTION_KEYWORDS)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")

# ─────────────────────────────────────────────
# GLOBALS (loaded at startup)
# ─────────────────────────────────────────────

collection = None
embedder = None
bm25_index = None       # BM25 index voor hybrid search
bm25_corpus_ids = None   # ChromaDB IDs gekoppeld aan BM25 posities
bm25_corpus_docs = None  # Documenten gekoppeld aan BM25 posities
bm25_corpus_meta = None  # Metadata gekoppeld aan BM25 posities

# Hybrid search configuratie
HYBRID_ENABLED = os.getenv("HYBRID_SEARCH", "true").lower() == "true"
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.35"))  # gewicht BM25 in RRF (0-1)
RRF_K = int(os.getenv("RRF_K", "60"))                   # RRF constante


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and DB at startup."""
    global collection, embedder, bm25_index, bm25_corpus_ids, bm25_corpus_docs, bm25_corpus_meta

    # Load ChromaDB
    print("📦 ChromaDB laden...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    doc_count = collection.count()
    print(f"   ✓ {doc_count} chunks in database")

    # Load embedder
    print("🧠 Embedding model laden...")
    if EMBEDDING_PROVIDER == "openai":
        from ingest import OpenAIEmbedder
        embedder = OpenAIEmbedder()
    else:
        from ingest import LocalEmbedder
        embedder = LocalEmbedder()
    print("   ✓ Embedder klaar")

    # ── BM25 index bouwen (voor hybrid search) ──
    if HYBRID_ENABLED:
        try:
            from rank_bm25 import BM25Okapi
            import re as _re

            print("🔍 BM25 index bouwen...")
            # Haal ALLE documenten op uit ChromaDB
            all_data = collection.get(include=["documents", "metadatas"])
            bm25_corpus_ids = all_data["ids"]
            bm25_corpus_docs = all_data["documents"]
            bm25_corpus_meta = all_data["metadatas"]

            # Tokenize: simpele whitespace + lowercase, strip punctuatie
            def tokenize(text: str) -> list[str]:
                return _re.findall(r'\b\w{2,}\b', text.lower())

            tokenized_corpus = [tokenize(doc) for doc in bm25_corpus_docs]
            bm25_index = BM25Okapi(tokenized_corpus)
            print(f"   ✓ BM25 index klaar ({len(tokenized_corpus)} documenten)")
        except ImportError:
            print("   ⚠️  rank-bm25 niet geïnstalleerd — hybrid search uit")
            print("      pip install rank-bm25")
        except Exception as e:
            print(f"   ⚠️  BM25 index fout: {e}")
    else:
        print("🔍 Hybrid search uitgeschakeld (HYBRID_SEARCH=false)")

    print(f"🤖 LLM provider: {LLM_PROVIDER}")
    print(f"🌐 Server draait op http://localhost:{os.getenv('PORT', '8000')}")

    # ── TTS pre-warm (optioneel) ──
    if TTS_ENABLED:
        try:
            print("🔊 TTS: Kokoro-ONNX model laden...")
            _get_kokoro()
            print("   ✓ TTS klaar (kokoro-onnx)")
        except Exception as e:
            print(f"   ⚠️  TTS niet beschikbaar: {e}")
            print("      Installeer: pip install kokoro-onnx soundfile")
            print("      Download:   kokoro-v1.0.onnx + voices-v1.0.bin")

    yield

    print("👋 Server gestopt")


app = FastAPI(title="HAGA Folder RAG", lifespan=lifespan)

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class PatientProfiel(BaseModel):
    """Profiel ingevuld via de onboarding flow in de frontend."""
    leeftijdsgroep: Optional[str] = None   # "kind", "volwassene", "oudere"


class ChatRequest(BaseModel):
    message: str
    top_k: int = TOP_K
    history: list[dict] = []
    language: str = "B1 (Eenvoudig en begrijpelijk Nederlands)"
    profiel: Optional[PatientProfiel] = None


class Source(BaseModel):
    filename: str
    title: str
    chunk_preview: str
    relevance: float


class DisambiguationOption(BaseModel):
    """Een folder-optie die de patiënt kan kiezen bij ambigue vragen."""
    title: str
    filename: str
    relevance: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    suggestions: list[str] = []                          # Proactieve vervolgvragen
    disambiguation: list[DisambiguationOption] = []      # Folder-keuze opties bij ambigue vragen


# ─────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """Je bent de FolderAssistent van het HagaZiekenhuis.
Je beantwoordt vragen van patiënten UITSLUITEND op basis van de meegeleverde context uit patiëntenfolders.

ABSOLUTE VEILIGHEIDSREGELS — NOOIT OVERTREDEN:

1. ALLEEN CONTEXT GEBRUIKEN: Elk feit in je antwoord MOET letterlijk aantoonbaar zijn in de meegeleverde context.
   Gebruik NOOIT kennis uit je training over medicijnen, dosering, bijwerkingen, procedures of diagnoses.
   Als je merkt dat je iets "weet" maar het staat NIET in de context: zwijg erover.

2. TWIJFEL = STOPPEN: Als je ook maar de minste twijfel hebt of iets klopt of volledig is,
   zeg dan LETTERLIJK: "Ik kan dit niet betrouwbaar beantwoorden op basis van de beschikbare folders.
   Neem contact op met uw behandelend arts of de afdeling van het HagaZiekenhuis."

   UITZONDERING voor praktische/logistieke vragen (openingstijden, locaties, telefoonnummers,
   routebeschrijving, parkeren): deze staan niet in de medische folders. Verwijs dan ALTIJD
   naar www.haga.nl of adviseer de patiënt het HagaZiekenhuis te bellen. Noem NIET de behandelend
   arts als doorverwijzing voor dit soort praktische informatie.

3. NOOIT AANVULLEN: Vul context NOOIT aan met algemene medische kennis. Geen "over het algemeen",
   geen "doorgaans", geen "in de meeste gevallen". Alleen wat in de folders staat.

4. TEGENSTRIJDIGE CONTEXT: Als verschillende bronnen elkaar tegenspreken, geef dan GEEN antwoord maar zeg:
   "De beschikbare informatie is niet eenduidig. Raadpleeg uw arts voor het juiste antwoord."

5. GEEN MEDISCH ADVIES: Geef nooit persoonlijk medisch advies over dosering, medicatie, symptomen of behandeling.
   Verwijs altijd naar de behandelend arts.

6. BRONNEN VERPLICHT: Vermeld altijd expliciet uit welke folder(s) je antwoord komt.
   Citeer bij voorkeur de exacte zin uit de bron.

{leeftijd_instructie}

AANPAK: Lees de context zorgvuldig. Citeer letterlijk waar mogelijk. Bij twijfel: verwijs door.

VERDUIDELIJKING:
Als de meegeleverde context niet duidelijk aansluit bij de vraag van de patiënt, stel dan EEN
verduidelijkende vraag. Geef daarbij aan wat je WEL hebt gevonden, zodat de patiënt weet dat
je meedenkt. Voorbeeld: "Ik heb informatie over het FibroScan-onderzoek van de lever. Bedoelt u
dit onderzoek, of zoekt u informatie over een ander leveronderzoek?"
Stel ALLEEN een verduidelijkende vraag als je echt niet kunt bepalen welk onderwerp bedoeld wordt.
Als de context wél bij de vraag past, geef dan gewoon antwoord.

VERVOLGVRAGEN:
Sluit je antwoord af met maximaal 2 korte vervolgvragen die de patiënt zou kunnen stellen,
gebaseerd op de folders in de context. Markeer deze met het label "SUGGESTIES:" op een nieuwe regel.
Geef GEEN suggesties als je het antwoord niet kon geven of als je doorverwijst."""

LEEFTIJD_INSTRUCTIES = {
    "kind":       "DOELGROEP: Dit is een KIND of een ouder/verzorger die namens een kind vraagt. Gebruik eenvoudige, vriendelijke taal. Vermijd angstaanjagende beschrijvingen. Richt je tot het kind én de ouder/verzorger.",
    "volwassene": "DOELGROEP: Dit is een volwassen patiënt. Spreek hem/haar direct aan.",
    "oudere":     "DOELGROEP: Dit is een oudere patiënt. Gebruik heldere taal, vermijd afkortingen en digitaal jargon. Wees geduldig en volledig in je uitleg.",
}


def build_system_prompt(profiel: Optional[PatientProfiel], language: str) -> str:
    """Bouw dynamische system prompt op basis van patiëntprofiel en taalvoorkeur."""
    leeftijd = profiel.leeftijdsgroep if profiel else None
    leeftijd_instructie = LEEFTIJD_INSTRUCTIES.get(leeftijd, "")

    base = BASE_SYSTEM_PROMPT.format(leeftijd_instructie=leeftijd_instructie)

    # Bepaal of de taal niet-Nederlands is — geef dan een extra vertaalinstructie
    is_non_dutch = not language.lower().startswith("b1") and not language.lower().startswith("b2") and not language.lower().startswith("c1")
    vertaal_instructie = (
        "De patiëntenfolders zijn in het Nederlands geschreven. "
        "Vertaal de informatie volledig naar de gevraagde taal. "
        "Gebruik GEEN Nederlandse woorden of zinnen in je antwoord."
    ) if is_non_dutch else (
        "De folders zijn in het Nederlands. Pas de stijl aan op het opgegeven niveau."
    )

    return (
        f"{base}\n\n"
        f"TAAL- EN STIJLINSTRUCTIE (VERPLICHT — NEGEER DIT NOOIT):\n"
        f"Taal/stijl: {language}\n"
        f"{vertaal_instructie}\n"
        f"Zorg dat alle medische informatie 100% accuraat blijft."
    )


def retrieve(query: str, top_k: int = 15) -> list[dict]:
    """Zoek relevante chunks via vector similarity."""
    is_e5 = "e5" in os.getenv("EMBEDDING_MODEL", "").lower()
    q_text = f"query: {query}" if is_e5 else query
    q_embedding = embedder.embed([q_text], batch_size=1)[0]

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        similarity = 1 - results["distances"][0][i]
        if similarity >= RELEVANCE_THRESHOLD:
            chunks.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity": similarity,
            })

    return chunks


def bm25_retrieve(query: str, top_k: int = 15) -> list[dict]:
    """Zoek relevante chunks via BM25 (keyword matching)."""
    import re as _re

    if bm25_index is None:
        return []

    tokens = _re.findall(r'\b\w{2,}\b', query.lower())
    if not tokens:
        return []

    scores = bm25_index.get_scores(tokens)

    # Sorteer op score, pak top_k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    chunks = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        chunks.append({
            "id": bm25_corpus_ids[idx],
            "text": bm25_corpus_docs[idx],
            "metadata": bm25_corpus_meta[idx],
            "bm25_score": float(scores[idx]),
        })

    return chunks


def hybrid_retrieve(query: str, top_k: int = 15) -> list[dict]:
    """
    Hybrid search: combineert dense (embedding) + sparse (BM25) via
    Reciprocal Rank Fusion (RRF).

    RRF score = Σ  1 / (k + rank_i)
    waarbij rank_i de positie in de i-de ranker is (1-indexed).

    Dense vangt semantische matches ("wat moet ik doen bij aambeien"
    → "hemorroiden behandeling"), BM25 vangt keyword matches
    ("telefoonnummer polikliniek" → chunk met exact dat telefoonnummer).
    """
    if not HYBRID_ENABLED or bm25_index is None:
        return retrieve(query, top_k)

    # 1. Beide retrievers draaien
    dense_results = retrieve(query, top_k=top_k * 2)  # overophalen voor betere fusion
    bm25_results = bm25_retrieve(query, top_k=top_k * 2)

    # 2. Bouw RRF scores per chunk-ID
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}
    dense_weight = 1.0 - BM25_WEIGHT

    # Dense resultaten
    for rank, chunk in enumerate(dense_results, 1):
        cid = chunk["id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + dense_weight / (RRF_K + rank)
        if cid not in chunk_map:
            chunk_map[cid] = chunk

    # BM25 resultaten
    for rank, chunk in enumerate(bm25_results, 1):
        cid = chunk["id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + BM25_WEIGHT / (RRF_K + rank)
        if cid not in chunk_map:
            # BM25 chunk heeft geen similarity — schat op basis van positie
            chunk_map[cid] = {
                "id": cid,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "distance": 0.5,  # placeholder
                "similarity": 0.5,
            }

    # 3. Sorteer op RRF-score, neem top_k
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    merged = []
    for cid, rrf_score in ranked:
        chunk = chunk_map[cid]
        # Gebruik de dense similarity als die er is, anders een schatting
        chunk["rrf_score"] = round(rrf_score, 5)
        merged.append(chunk)

    # Zet de similarity op basis van de dense score als die beschikbaar is
    # (nodig voor de rest van de pipeline die op similarity filtert)
    for chunk in merged:
        if "similarity" not in chunk or chunk.get("distance", 0) == 0.5:
            chunk["similarity"] = max(0.35, chunk["rrf_score"] * 50)
            chunk["distance"] = 1 - chunk["similarity"]

    bm25_only = sum(1 for cid, _ in ranked if cid not in {c["id"] for c in dense_results})
    if bm25_only > 0:
        print(f"   🔍 Hybrid: +{bm25_only} chunks via BM25 die dense miste")

    return merged


def classify_and_rewrite_query(message: str) -> str:
    """
    Fix 3: Herschrijf brede/vage patiëntvragen naar specifiekere zoektermen.

    "Ik heb last van aambeien, wat moet ik doen?" is te vaag voor de
    embedder — de similarity met een folder over "hemorroiden behandeling"
    is laag. We herschrijven dit naar keywords die wél matchen.

    Dit is een lightweight rule-based rewrite, geen LLM-call (te traag).
    """
    msg_lower = message.lower().strip()

    # Patroon: "Ik heb last van X" / "Mijn kind heeft X"
    # Extract klacht uit veelvoorkomende patronen
    patterns = [
        r"(?:ik heb|ik krijg|ik lijd aan|ik heb last van|last van)\s+(.+?)(?:\.|,|\?|$)",
        r"(?:mijn kind|mijn baby|mijn zoon|mijn dochter)\s+(?:heeft|krijgt)\s+(?:last van\s+)?(.+?)(?:\.|,|\?|$)",
        r"(?:wat (?:moet|kan) ik doen (?:bij|voor|tegen|als|wanneer))\s+(.+?)(?:\.|,|\?|$)",
        r"(?:wat te doen bij|hoe om te gaan met|tips voor)\s+(.+?)(?:\.|,|\?|$)",
    ]

    for pattern in patterns:
        m = re.search(pattern, msg_lower)
        if m:
            klacht = m.group(1).strip()
            # Verwijder stoplwoorden
            stopwoorden = {"een", "de", "het", "mijn", "erg", "veel", "heel", "best", "wel", "ook", "nog", "steeds"}
            klacht_woorden = [w for w in klacht.split() if w not in stopwoorden]
            if klacht_woorden:
                rewritten = f"{' '.join(klacht_woorden)} behandeling informatie patiëntenfolder"
                print(f"   ✏️  Query rewrite: '{message[:50]}' → '{rewritten[:60]}'")
                return rewritten

    # Patroon: vraag die begint met "wat is" → voeg "patiëntenfolder" toe
    if msg_lower.startswith("wat is een") or msg_lower.startswith("wat is de") or msg_lower.startswith("wat zijn"):
        return f"{message} patiëntenfolder HagaZiekenhuis"

    return message


def fetch_sibling_chunks(chunks: list[dict], window: int = 2) -> list[dict]:
    """
    Haal aangrenzende chunks op voor elk gevonden document.

    Het probleem: bij een folder over de daglounge scoort chunk 0
    (de intro) hoog, maar de openingstijden staan in chunk 1 of 2.
    Die lagere chunks vallen buiten TOP_K en worden nooit aan het
    LLM gegeven.

    Oplossing: voor elk uniek document in de resultaten, haal ook
    chunk_index ± window op uit ChromaDB. Voeg die toe aan de context
    met een lagere similarity-score zodat het LLM ze als aanvullend
    beschouwt.

    Parameters
    ----------
    chunks : list[dict]
        Resultaten van retrieve() — reeds gefilterd op threshold.
    window : int
        Aantal aangrenzende chunks aan elke kant (default 2).
        chunk_idx-2, chunk_idx-1, chunk_idx+1, chunk_idx+2
    """
    if not chunks:
        return chunks

    # Verzamel welke (filename, chunk_index) combinaties we al hebben
    seen_ids: set[str] = set()
    for c in chunks:
        meta = c["metadata"]
        fname = meta.get("filename", "")
        cidx = meta.get("chunk_index", -1)
        seen_ids.add(f"{fname}::{cidx}")

    extra_chunks = []

    # Per gevonden chunk: haal siblings op via metadata-filter
    for chunk in chunks:
        meta = chunk["metadata"]
        fname = meta.get("filename", "")
        cidx = int(meta.get("chunk_index", -1))
        total = int(meta.get("total_chunks", 0))

        if not fname or cidx < 0:
            continue

        for offset in range(-window, window + 1):
            if offset == 0:
                continue  # al aanwezig
            target_idx = cidx + offset
            if target_idx < 0 or (total > 0 and target_idx >= total):
                continue
            uid = f"{fname}::{target_idx}"
            if uid in seen_ids:
                continue

            # ChromaDB metadata-query op filename + chunk_index
            try:
                r = collection.get(
                    where={"$and": [
                        {"filename": {"$eq": fname}},
                        {"chunk_index": {"$eq": target_idx}},
                    ]},
                    include=["documents", "metadatas"],
                )
                if r["documents"]:
                    seen_ids.add(uid)
                    # Sibling krijgt iets lagere score dan de gevonden chunk
                    # zodat het LLM ze als context ziet maar primaire chunks
                    # voorrang geeft
                    sibling_sim = max(chunk["similarity"] - 0.05, 0.0)
                    extra_chunks.append({
                        "text": r["documents"][0],
                        "metadata": r["metadatas"][0],
                        "distance": round(1 - sibling_sim, 3),
                        "similarity": round(sibling_sim, 3),
                        "_is_sibling": True,
                    })
            except Exception:
                pass  # Stille fout — sibling ophalen is best-effort

    if extra_chunks:
        print(f"   +{len(extra_chunks)} sibling chunks toegevoegd")

    # Bewaar de volgorde: gevonden chunks eerst, siblings daarna gegroepeerd
    # per document zodat de context coherent leest
    all_chunks = chunks + extra_chunks

    # Deduplicate op (filename, chunk_index) — voor de zekerheid
    seen_final: set[str] = set()
    deduped = []
    for c in all_chunks:
        uid = f"{c['metadata'].get('filename','')}::{c['metadata'].get('chunk_index','')}"
        if uid not in seen_final:
            seen_final.add(uid)
            deduped.append(c)

    return deduped


def build_context(chunks: list[dict]) -> str:
    """Bouw context string voor de LLM prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        parts.append(
            f"--- Bron {i}: {meta.get('title', meta.get('filename', 'Onbekend'))} ---\n"
            f"{chunk['text']}\n"
        )
    return "\n".join(parts)


def detect_disambiguation(chunks: list[dict]) -> list[DisambiguationOption]:
    """
    Detecteer of de resultaten ambigue zijn: meerdere folders met vergelijkbare
    scores, geen duidelijke winnaar → bied de patiënt een keuze aan.

    Triggers:
    - Chunks komen uit ≥ DISAMBIGUATION_MIN_FOLDERS verschillende folders
    - Score-verschil tussen folder 1 en folder 2 is klein (< 0.08)
    - Folders hebben duidelijk verschillende titels (niet dezelfde folder)
    
    Werkt ook bij hogere scores: als 3 borst-folders allemaal ~0.70 scoren
    is er geen duidelijke winnaar, ook al is 0.70 normaal "hoge confidence".
    """
    if not chunks:
        return []

    # Groepeer chunks per folder, bewaar de beste score per folder
    folder_best: dict[str, dict] = {}
    for c in chunks:
        fname = c["metadata"].get("filename", "")
        title = c["metadata"].get("title", fname)
        sim = c["similarity"]
        if fname not in folder_best or sim > folder_best[fname]["sim"]:
            folder_best[fname] = {"title": title, "filename": fname, "sim": sim}

    unique_folders = list(folder_best.values())

    # Minder dan 2 folders → geen disambiguation
    if len(unique_folders) < DISAMBIGUATION_MIN_FOLDERS:
        return []

    # Sorteer op score (hoog → laag)
    unique_folders.sort(key=lambda x: x["sim"], reverse=True)

    # Kernlogica: is er een duidelijke winnaar?
    # Als folder #1 veel hoger scoort dan #2 (>0.08 verschil), is het duidelijk
    if len(unique_folders) >= 2:
        gap = unique_folders[0]["sim"] - unique_folders[1]["sim"]
        if gap > 0.08:
            return []  # Duidelijke winnaar

    # Tel hoeveel folders "dichtbij" de top scoren (binnen 0.08 van de beste)
    top_sim = unique_folders[0]["sim"]
    close_folders = [f for f in unique_folders if top_sim - f["sim"] <= 0.08]

    # Als maar 1 folder in de buurt zit, geen disambiguation nodig
    if len(close_folders) < 2:
        return []

    # Disambiguation nodig — geef de dichtbijzijnde folders terug
    options = []
    for f in close_folders[:DISAMBIGUATION_MAX_OPTIONS]:
        if f["sim"] >= RELEVANCE_THRESHOLD:
            options.append(DisambiguationOption(
                title=f["title"],
                filename=f["filename"],
                relevance=round(f["sim"], 3),
            ))

    return options if len(options) >= 2 else []


def get_confidence_tier(chunks: list[dict]) -> str:
    """
    Bepaal de confidence tier op basis van de beste chunk score.
    Retourneert 'high', 'medium', of 'low'.
    """
    if not chunks:
        return "low"

    best_sim = chunks[0]["similarity"]
    if best_sim >= CONFIDENCE_HIGH:
        return "high"
    elif best_sim >= CONFIDENCE_MEDIUM:
        return "medium"
    else:
        return "low"


def build_history_messages(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """
    Zet de gesprekshistorie om naar een echte OpenAI-compatible messages array.

    - Pakt de laatste `max_turns` beurten (1 beurt = user + assistant)
    - Strips de folder-context uit eerdere user-berichten — die stond alleen
      in de prompt als hulp voor het model, niet als echte inhoud.
    - Geeft een lijst van {role, content} dicts terug, klaar om in te voegen
      tussen system prompt en het nieuwe user-bericht.
    """
    if not history:
        return []

    # Pak maximaal max_turns * 2 berichten (user + assistant per beurt)
    recent = history[-(max_turns * 2):]

    messages = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Strip de folder-context injectie uit eerdere user-berichten
        # zodat we alleen de echte vraag van de patiënt bewaren
        if role == "user" and "Context uit patiëntenfolders:" in content:
            # Haal het stuk na "Vraag van de patiënt:" eruit
            if "Vraag van de patiënt:" in content:
                content = content.split("Vraag van de patiënt:")[-1]
                content = content.split("Beantwoord de vraag")[0].strip()
            else:
                # Fallback: alles na de context-blok
                content = content.split("\n\n")[-1].strip()

        if content:
            messages.append({"role": role, "content": content})

    return messages


def rewrite_query_with_context(message: str, history: list[dict]) -> str:
    """
    Verrijk een vage vervolgvraag met context uit de vorige assistent-beurt.

    'en hoe lang duren die?' + vorige beurt over BCG-bijwerkingen
    → 'hoe lang duren bijwerkingen na BCG behandeling?'

    Zonder dit scoort de embedding van een vervolgvraag te laag om
    relevante chunks te vinden.
    """
    if not history:
        return message

    # Haal het laatste assistantbericht op
    last_assistant = ""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "")[:300]  # eerste 300 chars is genoeg
            break

    if not last_assistant:
        return message

    # Simpele heuristiek: is dit een vage vervolgvraag?
    # Signaalwoorden die wijzen op een referentie naar eerder gesprek
    followup_signals = [
        "dat", "die", "dit", "die", "het", "ze", "hij", "haar",
        "en ", "maar ", "ook ", "verder", "meer", "nog", "daarna",
        "waarom", "hoe lang", "hoe vaak", "wanneer", "wat als",
        "en wat", "en hoe", "en waarom", "en wanneer",
    ]
    is_followup = (
        len(message.split()) <= 8  # korte vraag
        and any(message.lower().startswith(s) or f" {s}" in message.lower()
                for s in followup_signals)
    )

    if is_followup:
        # Verrijk de query: plak de context van de vorige beurt eraan
        return f"{last_assistant[:150]} — {message}"

    return message


async def generate_ollama(messages: list[dict]) -> str:
    """Genereer antwoord via Ollama (lokaal)."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


async def generate_openai(messages: list[dict]) -> str:
    """Genereer antwoord via OpenAI API."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.0,  # 0.0 = deterministisch, minimaliseert hallucinaties in medische context
        max_tokens=1000,
    )
    return response.choices[0].message.content


async def generate(messages: list[dict]) -> str:
    """Route naar de juiste LLM provider."""
    if LLM_PROVIDER == "openai":
        return await generate_openai(messages)
    return await generate_ollama(messages)


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Hoofdendpoint: ontvang vraag, geef RAG-antwoord."""

    # ── 0. Crisis-check — altijd als eerste, vóór RAG ──
    msg_lower = req.message.lower()
    if any(signal in msg_lower for signal in CRISIS_SIGNALS):
        print(f"🚨 CRISIS GEDETECTEERD: '{req.message[:60]}'")
        return ChatResponse(answer=CRISIS_RESPONSE, sources=[])

    # ── 1. Query expansion — gebruik ASSISTANT-context, niet user-context ──
    #
    # Fout patroon (oud): vorige USER-vraag plakken → koppelt altijd aan vorig onderwerp
    # Correct patroon:    ASSISTENT-antwoord gebruiken → bevat de medische termen
    #                     waar de verwijzing ("die", "dat") naar verwijst
    #
    # Detectie: anaphorische verwijswoorden (die/dat/het/ze) of korte vage zin
    ANAPHORA = {"die", "dat", "dit", "het", "ze", "hij", "haar", "hen", "daar"}
    # Strip leestekens zodat "die?" ook matcht
    woorden = {w.strip("?.!,") for w in req.message.lower().split()}
    is_anaphor = bool(woorden & ANAPHORA) and len(req.message.split()) <= 10

    search_query = req.message
    if is_anaphor and req.history:
        # Pak het LAATSTE ASSISTENT-antwoord (bevat de medische context)
        last_assistant = next(
            (m["content"] for m in reversed(req.history) if m["role"] == "assistant"), ""
        )
        if last_assistant:
            # Gebruik eerste 120 tekens: genoeg voor de embedder, niet te veel ruis
            search_query = f"{last_assistant[:120]} {req.message}"
            print(f"🔄 Anaphor query expansion: '{search_query[:100]}...'")
    elif req.history:
        # Geen anaphorische verwijswoorden, maar mogelijk toch een vervolgvraag
        rewritten = rewrite_query_with_context(req.message, req.history)
        if rewritten != req.message:
            search_query = rewritten
            print(f"🔄 Followup query rewrite: '{search_query[:100]}...'")
    else:
        # Geen history — probeer brede/vage vragen te herschrijven
        search_query = classify_and_rewrite_query(req.message)

    # ── 2. Hybrid retrieve + sibling chunks ──
    chunks = hybrid_retrieve(search_query, top_k=req.top_k)
    # Voeg aangrenzende chunks toe van gevonden folders.
    chunks = fetch_sibling_chunks(chunks, window=2)
    context = build_context(chunks)

    # ── 2b. Confidence tier bepalen ──
    confidence = get_confidence_tier(chunks)

    # ── 2c. Disambiguation check ──
    # Bij meerdere kandidaat-folders met vergelijkbare scores: bied keuze aan
    # Dit werkt op ELKE confidence tier — ook bij hoge scores kan het ambigue zijn
    # (bv. "borstoperatie" matcht 3 borst-folders die allemaal ~0.70 scoren)
    disambiguation_options = []
    if not req.history:
        # Alleen bij eerste vragen — bij follow-ups heeft disambiguation weinig zin
        # Sla disambiguation over als de query specifiek genoeg is:
        # - ≥ 5 woorden → patiënt heeft al voldoende context gegeven
        # - bevat een procedure/fase-term → aantoonbaar specifiek
        _SPECIFIC_TERMS = {
            "leefregels", "voorbereiding", "nazorg", "herstel", "operatie",
            "onderzoek", "behandeling", "bijwerkingen", "nuchter", "ingreep",
            "klachten", "symptomen", "thuiskomst", "ontslag", "controle",
            "medicatie", "medicijn", "revalidatie", "wond", "verbond",
        }
        _query_words = req.message.lower().split()
        _is_specific = (
            len(_query_words) >= 5
            or any(t in req.message.lower() for t in _SPECIFIC_TERMS)
        )
        if not _is_specific:
            disambiguation_options = detect_disambiguation(chunks)

    if disambiguation_options:
        # Meerdere folders scoren vergelijkbaar — bied keuze aan
        option_lines = "\n".join(
            f"• **{opt.title}**" for opt in disambiguation_options
        )
        disamb_answer = (
            f"Ik vind meerdere folders die bij uw vraag zouden kunnen passen. "
            f"Kunt u aangeven welk onderwerp u bedoelt?\n\n{option_lines}\n\n"
            f"U kunt ook uw vraag specifieker formuleren, dan kan ik gerichter zoeken."
        )
        print(f"🔀 DISAMBIGUATION: {len(disambiguation_options)} opties aangeboden (confidence={confidence})")
        return ChatResponse(
            answer=disamb_answer,
            sources=[],
            disambiguation=disambiguation_options,
        )

    # 3. Veiligheidscheck: geen relevante chunks gevonden
    if not chunks:
        if not req.history:
            # Check of de vraag vaag/kort is — dan verduidelijking i.p.v. harde doorverwijzing
            woord_count = len(req.message.split())
            if woord_count <= 5:
                no_context_answer = (
                    "Ik kan uw vraag nog niet goed genoeg plaatsen om de juiste folder te vinden. "
                    "Kunt u iets specifieker zijn? Bijvoorbeeld:\n\n"
                    "• Welk onderzoek of welke behandeling bedoelt u?\n"
                    "• Bij welke afdeling bent u onder behandeling?\n\n"
                    "Hoe specifieker uw vraag, hoe beter ik u kan helpen."
                )
            else:
                no_context_answer = (
                    "Op basis van de beschikbare patiëntenfolders kan ik uw vraag niet betrouwbaar beantwoorden. "
                    "De informatie die u zoekt staat mogelijk niet in onze folders, of uw vraag vereist persoonlijk medisch advies. "
                    "Neem contact op met uw behandelend arts of bel het HagaZiekenhuis via het algemene nummer."
                )
            return ChatResponse(answer=no_context_answer, sources=[])

    # 3. Bouw dynamische system prompt (leeftijdsgroep + taalvoorkeur)
    dynamic_system_prompt = build_system_prompt(req.profiel, req.language)
    messages = [{"role": "system", "content": dynamic_system_prompt}]

    # 4. Voeg eerdere beurten in als echte multi-turn messages
    history_messages = build_history_messages(req.history)
    messages.extend(history_messages)

    # ── 5. Confidence-tiered instructie ──
    if chunks:
        best_similarity = chunks[0]["similarity"]

        if confidence == "high":
            # Hoge confidence: direct antwoord, geen extra waarschuwing
            relevantie_waarschuwing = ""
        elif confidence == "medium":
            # Medium confidence: antwoord + caveat
            relevantie_waarschuwing = (
                f"\n⚠️ INSTRUCTIE VOOR DE ASSISTENT: De gevonden bronnen scoren matig "
                f"({best_similarity:.2f}). Geef je antwoord, maar sluit af met een korte opmerking: "
                f"'Dit antwoord is gebaseerd op de folder [foldernaam]. Als dit niet aansluit bij uw "
                f"situatie, neem dan contact op met uw behandelend arts.'\n"
            )
            # Bij medium + disambiguation: stuur de opties mee als hint
            if disambiguation_options:
                option_titles = ", ".join(f"'{o.title}'" for o in disambiguation_options)
                relevantie_waarschuwing += (
                    f"\nLET OP: Er zijn meerdere folders die passen: {option_titles}. "
                    f"Als je niet zeker weet welke folder bedoeld wordt, vraag dan verduidelijking "
                    f"en noem de mogelijke folders.\n"
                )
        else:
            # Lage confidence (maar niet gedisambigueerd, bv. bij follow-ups)
            relevantie_waarschuwing = (
                f"\n⚠️ WAARSCHUWING VOOR DE ASSISTENT: De gevonden bronnen hebben een lage relevantiescore "
                f"({best_similarity:.2f}). Wees extra voorzichtig. Als de context de vraag niet duidelijk "
                f"beantwoordt, stel dan een verduidelijkende vraag aan de patiënt of verwijs door.\n"
            )
    else:
        relevantie_waarschuwing = (
            "\n⚠️ WAARSCHUWING VOOR DE ASSISTENT: Er zijn geen folder-bronnen gevonden voor deze vraag. "
            "Beantwoord alleen op basis van wat eerder in dit gesprek is besproken. "
            "Voeg geen nieuwe medische feiten toe die niet in de folders of eerdere berichten stonden.\n"
        )

    # 6. Laatste user-bericht: folder-context + expliciete instructie voor specificiteit
    # De taal-reminder staat bewust ALS LAATSTE zodat het model niet teruggetrokken wordt
    # naar het Nederlands van de folders (die altijd NL zijn).
    user_prompt = f"""Context uit patiëntenfolders:
{context}
{relevantie_waarschuwing}
Vraag van de patiënt: {req.message}

Beantwoord de vraag op basis van bovenstaande context.
- Som concrete bijwerkingen, stappen of instructies ALTIJD op als een lijst.
- Noem GEEN concrete doseringen (mg, ml, aantal tabletten) — verwijs daarvoor naar de arts of apotheker.
- Als de taalinstelling een andere taal of stijl vereist, VERTAAL je de folder-inhoud; citeer NIET letterlijk in het Nederlands tenzij de gekozen taal/stijl Nederlands is.

⚠️ TAAL-INSTRUCTIE (heeft prioriteit boven alles): Geef je volledige antwoord UITSLUITEND in de volgende taal en stijl: {req.language}"""

    messages.append({"role": "user", "content": user_prompt})

    # 7. DEBUG log
    print("\n" + "=" * 80)
    print(f"🧪 DEBUG | msg='{req.message[:60]}' | history={len(req.history)} | chunks={len(chunks)} | lang='{req.language[:20]}'")
    print(f"   query_gebruikt='{search_query[:80]}'")
    print(f"   confidence_tier={confidence}")
    if disambiguation_options:
        print(f"   disambiguation={len(disambiguation_options)} opties (meegestuurd als hint)")
    if chunks:
        print(f"   beste_chunk_score={chunks[0]['similarity']:.2f} | bron={chunks[0]['metadata'].get('title','?')[:40]}")
    print("=" * 80 + "\n")

    answer = await generate(messages)

    # ── 7b. Suggesties extraheren uit het antwoord ──
    # Het LLM plaatst suggesties na "SUGGESTIES:" label — we parsen en verwijderen dit
    suggestions = []
    if "SUGGESTIES:" in answer:
        parts = answer.split("SUGGESTIES:", 1)
        answer = parts[0].rstrip()  # Hoofd-antwoord zonder suggesties
        raw_suggestions = parts[1].strip()
        for line in raw_suggestions.split("\n"):
            line = line.strip()
            line = re.sub(r"^[\-•\*\d\.]+\s*", "", line).strip()
            if line and len(line) > 10:  # filter ruis
                suggestions.append(line)
        suggestions = suggestions[:2]  # max 2

    # Server-side veiligheidscheck: GEEN suggesties bij doorverwijzingen
    # Het LLM luistert niet altijd naar "geen suggesties bij doorverwijzing",
    # dus we forceren dit hier
    doorverwijzing_signalen = [
        "kan ik uw vraag niet",
        "kan ik dit niet betrouwbaar",
        "niet in onze folders",
        "neem contact op met uw behandelend arts",
        "raadpleeg uw arts",
        "bel het hagaziekenhuis",
        "staat niet in de beschikbare folders",
    ]
    if any(signaal in answer.lower() for signaal in doorverwijzing_signalen):
        suggestions = []

    # ── 8. Post-processing: blokkeer dosering in antwoord ──
    # Alleen blokkeren als (a) de vraag expliciet over dosering gaat
    # én (b) het antwoord doseringscijfers bevat.
    # Dit voorkomt dat bijwerkingsvragen geblokkeerd worden omdat het
    # antwoord toevallig een getal bevat (bijv. "bij 10% van de patiënten").
    dosage_found = any(re.search(p, answer, re.IGNORECASE) for p in DOSAGE_PATTERNS)
    if dosage_found and is_dosage_question(req.message):
        print(f"⚠️  DOSERING GEDETECTEERD in antwoord op doseringsvraag — geblokkeerd")
        answer = DOSAGE_BLOCK_RESPONSE
    elif dosage_found:
        print(f"ℹ️  Cijfers in antwoord op niet-doseringsvraag — niet geblokkeerd ('{req.message[:50]}')")

    # 4. Format sources
    sources = []
    seen = set()
    for chunk in chunks:
        meta = chunk["metadata"]
        fname = meta.get("filename", "")
        if fname not in seen:
            seen.add(fname)
            sources.append(Source(
                filename=fname,
                title=meta.get("title", fname),
                chunk_preview=chunk["text"][:150] + "...",
                relevance=round(1 - chunk["distance"], 3),  # cosine similarity
            ))

    return ChatResponse(answer=answer, sources=sources, suggestions=suggestions)


# ─────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────

def _get_kokoro():
    """Laad Kokoro-ONNX model (lazy, singleton)."""
    global _kokoro_instance
    if _kokoro_instance is None:
        from kokoro_onnx import Kokoro
        _kokoro_instance = Kokoro(TTS_MODEL_PATH, TTS_VOICES_PATH)
    return _kokoro_instance


@app.get("/api/tts", summary="Zet tekst om naar WAV-audio (Kokoro-ONNX)")
async def tts(
    text: str = Query(..., description="Te spreken tekst", max_length=3000),
    lang: str = Query("nl", description="Taalcode: nl | en | tr | ar | fr"),
):
    """
    Geeft een WAV-audiobestand terug met de ingesproken tekst.
    Gebruikt het lokale Kokoro-82M ONNX model — geen data verlaat de server.
    """
    if not TTS_ENABLED:
        raise HTTPException(status_code=503, detail="TTS is uitgeschakeld op deze server.")

    config = TTS_LANG_CONFIG.get(lang)
    if config is None:
        raise HTTPException(
            status_code=400,
            detail=f"Onbekende taalcode '{lang}'. Kies uit: {list(TTS_LANG_CONFIG.keys())}",
        )

    try:
        import soundfile as sf

        kokoro = _get_kokoro()

        # kokoro-onnx create() retourneert (samples, sample_rate)
        samples, sample_rate = kokoro.create(
            text,
            voice=config["voice"],
            speed=config["speed"],
            lang=config["lang"],
        )

        if samples is None or len(samples) == 0:
            raise HTTPException(status_code=500, detail="Geen audio gegenereerd.")

        buf = io.BytesIO()
        sf.write(buf, samples, samplerate=sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)

        print(f"🔊 TTS [{lang}] — {len(text)} tekens → {len(samples)/sample_rate:.1f}s")

        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={"Cache-Control": "no-cache", "Content-Disposition": "inline; filename=tts.wav"},
        )

    except HTTPException:
        raise
    except Exception as exc:
        print(f"❌ TTS fout: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/stats")
async def stats():
    """Database statistieken."""
    count = collection.count()
    # Haal unieke filenames op
    sample = collection.peek(limit=1)
    return {
        "total_chunks": count,
        "llm_provider": LLM_PROVIDER,
        "llm_model": OLLAMA_MODEL if LLM_PROVIDER == "ollama" else OPENAI_MODEL,
        "embedding_provider": EMBEDDING_PROVIDER,
    }


@app.get("/api/search")
async def search(q: str, top_k: int = 10):
    """Pure vector search (zonder LLM) — handig voor debugging."""
    chunks = retrieve(q, top_k=top_k)
    return [
        {
            "text": c["text"][:300],
            "filename": c["metadata"].get("filename"),
            "title": c["metadata"].get("title"),
            "similarity": round(1 - c["distance"], 3),
        }
        for c in chunks
    ]


# ─────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
