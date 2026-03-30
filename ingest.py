"""
HAGA Folder RAG — Ingestion Pipeline
=====================================
Extraheert tekst uit PDF-folders, splitst in chunks, genereert embeddings,
slaat op in ChromaDB en verrijkt metadata via een lokale LLM.

Gebruik:
    # Standaard (snelle) ingest:
    python ingest.py --folder-dir ./folders/

    # Met LLM metadata-verrijking (titel, fase, leefregels):
    python ingest.py --folder-dir ./folders/ --enrich

    # Enterprise ingest met OpenAI embeddings:
    python ingest.py --folder-dir ./folders/ --enrich --provider openai
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
import chromadb
import requests
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# 1. AI METADATA ENRICHMENT
# ─────────────────────────────────────────────

# Geldige waarden voor fase-classificatie — alles buiten deze set
# wordt door de validatie teruggezet naar "Algemeen".
GELDIGE_FASES = {"Algemeen", "Voorbereiding", "Onderzoek", "Behandeling", "Nazorg"}

# Pad naar de JSON-cache die gegenereerde metadata per folder
# bewaart zodat een herhaalde ingest niet alle LLM-calls opnieuw
# hoeft te doen.
ENRICHMENT_CACHE_PATH = os.getenv("ENRICHMENT_CACHE", "./enrichment_cache.json")


def _laad_cache() -> dict:
    """Laad bestaande enrichment-cache van schijf."""
    try:
        with open(ENRICHMENT_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _sla_cache_op(cache: dict) -> None:
    """Sla de enrichment-cache op naar schijf."""
    with open(ENRICHMENT_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _parse_llm_json(raw: str) -> dict | None:
    """
    Parse de LLM-response als JSON. LLMs wikkelen hun antwoord
    soms in markdown-codeblokken of voegen preamble toe.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```json\s*", "", raw)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Probeer het eerste JSON-object uit de tekst te vissen
        match = re.search(r"\{[^}]+\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _valideer_metadata(parsed: dict, fallback_titel: str) -> dict:
    """
    Valideer en normaliseer de LLM-output naar een betrouwbaar
    metadata-dict met bekende types en waarden.
    """
    titel = str(parsed.get("titel", "")).strip()
    titel = re.sub(r'["\']', "", titel)  # Haal aanhalingstekens weg
    if not titel or len(titel) > 80:
        titel = fallback_titel

    fase = str(parsed.get("fase", "Algemeen")).strip()
    if fase not in GELDIGE_FASES:
        fase = "Algemeen"

    is_leefregel = parsed.get("is_leefregel", False)
    if not isinstance(is_leefregel, bool):
        is_leefregel = str(is_leefregel).lower() in ("true", "1", "ja", "yes")

    return {
        "patient_friendly_title": titel,
        "fase": fase,
        "is_leefregel": is_leefregel,
    }


def genereer_metadata_lokaal(
    tekst_snippet: str,
    fallback_titel: str,
) -> dict:
    """
    Gebruikt de lokale Ollama LLM om gestructureerde metadata
    te genereren op basis van de folder-introductie.

    Returns
    -------
    dict met keys: patient_friendly_title, fase, is_leefregel
    """
    prompt = f"""Je bent een medisch redacteur voor het HagaZiekenhuis.
Analyseer de introductie van deze patiëntenfolder en genereer gestructureerde metadata.

Geef UITSLUITEND een geldig JSON-object terug (geen toelichting, geen markdown).
Gebruik deze exacte keys:
- "titel": Korte, begrijpelijke titel voor een patiënt (max 6 woorden)
- "fase": Kies EXACT ÉÉN uit: ["Algemeen", "Voorbereiding", "Onderzoek", "Behandeling", "Nazorg"]
- "is_leefregel": boolean (true als de folder regels bevat over eten, bewegen, douchen, autorijden, herstel, of andere dagelijkse adviezen)

Folder tekst:
{tekst_snippet[:1500]}"""

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        parsed = _parse_llm_json(raw)
        if parsed:
            return _valideer_metadata(parsed, fallback_titel)

        # JSON parsing mislukt — probeer tenminste een titel te halen
        print(f"  ⚠ JSON parse mislukt, fallback naar titel-only")
        titel = re.sub(r'["\']', "", raw.split("\n")[0].strip())
        return {
            "patient_friendly_title": titel if titel and len(titel) < 80 else fallback_titel,
            "fase": "Algemeen",
            "is_leefregel": False,
        }

    except requests.exceptions.ConnectionError:
        print(f"  ⚠ Ollama niet bereikbaar op {base_url} — draait de server?")
        return {
            "patient_friendly_title": fallback_titel,
            "fase": "Algemeen",
            "is_leefregel": False,
        }
    except Exception as e:
        print(f"  ⚠ LLM fout (fallback): {e}")
        return {
            "patient_friendly_title": fallback_titel,
            "fase": "Algemeen",
            "is_leefregel": False,
        }


# ─────────────────────────────────────────────
# 2. PDF EXTRACTIE
# ─────────────────────────────────────────────


def fix_encoding(text: str) -> str:
    """
    Herstel dubbele UTF-8/Latin-1 encoding corruptie.

    PyMuPDF geeft soms tekst terug die als Latin-1 geïnterpreteerd is
    terwijl het eigenlijk UTF-8 bytes zijn. Resultaat: 'één' → 'Ã©Ã©n'.

    Fix: decodeer de string als Latin-1 terug naar bytes, interpreteer
    die bytes daarna opnieuw als UTF-8.
    """
    CORRUPTION_MARKERS = ["Ã©", "Ã«", "Ã¯", "Ã¶", "Ã¼", "Ã¨", "Ã ", "Ã²", "Ã®"]
    if not any(m in text for m in CORRUPTION_MARKERS):
        return text

    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def extract_text_from_pdf(pdf_path: str) -> dict | None:
    """
    Extraheert tekst + metadata uit een PDF.

    Gebruikt 'blocks' extractiemodus met positie-sortering voor
    correcte leesvolgorde bij meerkolomslayouts en zijbalken.
    """
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page_num, page in enumerate(doc):
            # "blocks" geeft tekstblokken met boundingbox-coördinaten.
            # Sorteer op y-positie (afgerond op 10px voor dezelfde "regel")
            # en dan op x-positie (links→rechts).
            blocks = page.get_text("blocks")
            blocks_sorted = sorted(
                [b for b in blocks if b[6] == 0 and b[4].strip()],
                key=lambda b: (round(b[1] / 10) * 10, b[0]),
            )
            text = "\n".join(b[4].strip() for b in blocks_sorted)

            if text.strip():
                text = fix_encoding(text)
                pages.append({"text": text.strip(), "page": page_num + 1})

        metadata = doc.metadata or {}
        doc.close()

        full_text = "\n\n".join(p["text"] for p in pages)
        return {
            "filename": Path(pdf_path).name,
            "filepath": str(pdf_path),
            "title": metadata.get("title", "") or Path(pdf_path).stem,
            "text": full_text,
            "pages": pages,
            "num_pages": len(pages),
        }
    except Exception as e:
        print(f"  ⚠ Fout bij {pdf_path}: {e}")
        return None


def clean_haga_text(text: str) -> str:
    """
    HAGA-specifieke tekstreiniging:
    - Verwijdert het meertalige disclaimerblok (NL/EN)
    - Verwijdert de feedbackvraag
    - Verwijdert pagina-nummers en folder-ID codes
    - Verwijdert copyright-footers van externe organisaties
    """
    disclaimer_patterns = [
        r"Wat vindt u van deze patiënteninformatie\?.*$",
        r"Spreekt u geen of slecht Nederlands\?.*$",
        r"Do you speak Dutch poorly or not at all\?.*$",
    ]
    for pattern in disclaimer_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Copyright-footers externe organisaties
    text = re.sub(
        r"©\s*Nederlandse Vereniging[^\n]*", "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"©?\s*met dank aan de\s+Nederlandse Vereniging[^\n]*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Folder-ID codes (bijv. "581805062025" aan het eind)
    text = re.sub(r"\n\d{10,15}\s*$", "", text.strip())

    # Losse paginanummers
    text = re.sub(r"^\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    # HagaZiekenhuis en Juliana Kinderziekenhuis footer-tekst
    text = re.sub(
        r"●?\s*Haga\s*Ziekenhuis\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE
    )
    text = re.sub(r"Juliana\s*Kinderziekenhuis\s*$", "", text, flags=re.MULTILINE)

    # Normaliseer whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_folder_id(filename: str) -> str:
    """Haal het folder-ID uit de bestandsnaam (bijv. '5818' uit '5818daglounge...pdf')."""
    match = re.match(r"(\d+)", filename)
    return match.group(1) if match else ""


def scan_pdfs(folder_dir: str) -> Generator[str, None, None]:
    """Vindt alle PDFs in een directory (recursief)."""
    folder_path = Path(folder_dir)
    if not folder_path.exists():
        print(f"❌ Map niet gevonden: {folder_dir}")
        sys.exit(1)

    pdf_files = sorted(folder_path.rglob("*.pdf"))
    print(f"📁 {len(pdf_files)} PDFs gevonden in {folder_dir}")
    yield from pdf_files


# ─────────────────────────────────────────────
# 3. CHUNKING
# ─────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Splitst tekst in overlappende chunks.
    Respecteert alinea-grenzen waar mogelijk.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size * 4:  # ~4 chars per token
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(para) > chunk_size * 4:
                sentences = para.replace(". ", ".\n").split("\n")
                sub_chunk = ""
                for sent in sentences:
                    if len(sub_chunk) + len(sent) > chunk_size * 4:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                        sub_chunk = sent
                    else:
                        sub_chunk += " " + sent
                current_chunk = sub_chunk if sub_chunk else ""
            else:
                current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Voeg overlap toe
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        overlap_chars = chunk_overlap * 4
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap_chars:]
            overlapped.append(prev_tail + "\n" + chunks[i])
        return overlapped

    return chunks


# ─────────────────────────────────────────────
# 4. EMBEDDINGS
# ─────────────────────────────────────────────


class LocalEmbedder:
    """Sentence-transformers embeddings (GPU-accelerated)."""

    def __init__(self, model_name: str = None):
        from sentence_transformers import SentenceTransformer
        import torch

        model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "intfloat/multilingual-e5-large"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Embedding model laden: {model_name} op {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.is_e5 = "e5" in model_name.lower()

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if self.is_e5:
            texts = [f"passage: {t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


class OpenAIEmbedder:
    """OpenAI API embeddings."""

    def __init__(self, model_name: str = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model_name or os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        print(f"🧠 OpenAI embedding model: {self.model}")

    def embed(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend([d.embedding for d in response.data])
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        return all_embeddings


def get_embedder(provider: str = None):
    provider = provider or os.getenv("EMBEDDING_PROVIDER", "local")
    if provider == "openai":
        return OpenAIEmbedder()
    return LocalEmbedder()


# ─────────────────────────────────────────────
# 5. VECTOR STORE (ChromaDB)
# ─────────────────────────────────────────────


def get_chroma_collection(reset: bool = False):
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("COLLECTION_NAME", "haga_folders")
    client = chromadb.PersistentClient(path=persist_dir)

    if reset:
        try:
            client.delete_collection(collection_name)
            print("🗑  Bestaande collectie verwijderd")
        except Exception:
            pass

    return client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────


def make_chunk_id(filename: str, chunk_idx: int) -> str:
    """Deterministic ID zodat re-runs geen duplicaten maken."""
    raw = f"{filename}::chunk_{chunk_idx}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest(
    folder_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    batch_size: int = 64,
    provider: str = None,
    reset: bool = False,
    enrich: bool = False,
):
    start = time.time()
    embedder = get_embedder(provider)
    collection = get_chroma_collection(reset=reset)

    # Laad enrichment-cache (zodat herhaalde runs de LLM niet opnieuw
    # hoeven aan te roepen voor al verwerkte folders)
    enrichment_cache = _laad_cache() if enrich else {}
    cache_hits = 0

    all_chunks, all_metadatas, all_ids = [], [], []
    min_chunk_len = int(os.getenv("MIN_CHUNK_LENGTH", "100"))

    print("\n📄 Stap 1/3: PDFs extracten en chunken...")
    if enrich:
        print("   ✨ LLM metadata-verrijking INGESCHAKELD")
        print(f"   📦 Cache: {len(enrichment_cache)} eerder verrijkte folders")
    pdf_count, short_total, skip_count = 0, 0, 0

    for pdf_path in scan_pdfs(folder_dir):
        doc = extract_text_from_pdf(str(pdf_path))
        if not doc or not doc["text"].strip():
            skip_count += 1
            continue

        cleaned_text = clean_haga_text(doc["text"])
        if not cleaned_text.strip():
            skip_count += 1
            continue

        folder_id = extract_folder_id(doc["filename"])
        fallback_title = doc["title"] or doc["filename"]

        # ── Metadata Enrichment ──────────────────────────
        if enrich:
            # Check cache eerst
            if doc["filename"] in enrichment_cache:
                ai_meta = enrichment_cache[doc["filename"]]
                cache_hits += 1
            else:
                ai_meta = genereer_metadata_lokaal(cleaned_text, fallback_title)
                enrichment_cache[doc["filename"]] = ai_meta
                # Sla cache tussentijds op (elke 50 folders) zodat bij crash
                # het werk niet verloren gaat
                if (pdf_count + 1) % 50 == 0:
                    _sla_cache_op(enrichment_cache)

            patient_title = ai_meta["patient_friendly_title"]
            fase = ai_meta["fase"]
            is_leefregel = ai_meta["is_leefregel"]

            if pdf_count < 5 or pdf_count % 100 == 0:
                print(
                    f"  ✨ [{folder_id}] {patient_title}  "
                    f"│ fase={fase} │ leefregel={is_leefregel}"
                )
        else:
            patient_title = fallback_title
            fase = ""
            is_leefregel = False

        # ── Chunking ─────────────────────────────────────
        chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)
        skipped_short = 0
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < min_chunk_len:
                skipped_short += 1
                continue

            all_chunks.append(chunk)
            metadata = {
                "filename": doc["filename"],
                "title": doc["title"],
                "patient_friendly_title": patient_title,
                "folder_id": folder_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "num_pages": doc["num_pages"],
                "source": doc["filepath"],
            }

            # Voeg enrichment-velden alleen toe als ze zinvol zijn,
            # zodat ChromaDB metadata-filtering er ook op kan werken
            if enrich:
                metadata["fase"] = fase
                metadata["is_leefregel"] = is_leefregel

            all_metadatas.append(metadata)
            all_ids.append(make_chunk_id(doc["filename"], i))

        if skipped_short > 0:
            short_total += skipped_short

        pdf_count += 1
        if not enrich and pdf_count % 100 == 0:
            print(f"  ✓ {pdf_count} PDFs verwerkt ({len(all_chunks)} chunks)")

    # Sla enrichment-cache definitief op
    if enrich and enrichment_cache:
        _sla_cache_op(enrichment_cache)
        print(f"\n  💾 Enrichment-cache opgeslagen: {len(enrichment_cache)} folders")
        print(f"     ({cache_hits} uit cache, {len(enrichment_cache) - cache_hits} nieuw)")

    print(f"\n  ✓ Totaal: {pdf_count} PDFs → {len(all_chunks)} chunks")
    if skip_count > 0:
        print(f"  ⏭  {skip_count} PDFs overgeslagen (geen tekst)")
    if short_total > 0:
        print(f"  🗑  {short_total} te korte chunks gefilterd (<{min_chunk_len} chars)")

    if not all_chunks:
        print("❌ Geen tekst gevonden in de PDFs!")
        return

    # Stap 2: Embeddings genereren
    print(f"\n🔢 Stap 2/3: Embeddings genereren ({len(all_chunks)} chunks)...")
    all_embeddings = embedder.embed(all_chunks, batch_size=batch_size)

    # Stap 3: Opslaan in ChromaDB
    print(f"\n💾 Stap 3/3: Opslaan in ChromaDB...")
    store_batch = 5000
    for i in range(0, len(all_chunks), store_batch):
        end = min(i + store_batch, len(all_chunks))
        collection.upsert(
            ids=all_ids[i:end],
            documents=all_chunks[i:end],
            embeddings=all_embeddings[i:end],
            metadatas=all_metadatas[i:end],
        )
        print(f"  ✓ {end}/{len(all_chunks)} chunks opgeslagen")

    elapsed = time.time() - start
    print(f"\n✅ Klaar! {pdf_count} PDFs → {len(all_chunks)} chunks in {elapsed:.1f}s")
    print(f"   Database: {os.getenv('CHROMA_PERSIST_DIR', './chroma_db')}")

    # Samenvatting enrichment-statistieken
    if enrich:
        fases = {}
        leefregels = 0
        for m in all_metadatas:
            f = m.get("fase", "Algemeen")
            fases[f] = fases.get(f, 0) + 1
            if m.get("is_leefregel"):
                leefregels += 1

        print(f"\n📊 Enrichment statistieken:")
        for f in sorted(fases.keys()):
            bar = "█" * (fases[f] // 20) or "▏"
            print(f"   {f:<16} {fases[f]:>5} chunks  {bar}")
        print(f"   {'Leefregels':<16} {leefregels:>5} chunks")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAGA Folder RAG — Ingestion")
    parser.add_argument("--folder-dir", required=True, help="Map met PDF-folders")
    parser.add_argument("--chunk-size", type=int, default=500, help="Tokens per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap tokens")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--provider", choices=["local", "openai"], default=None)
    parser.add_argument("--reset", action="store_true", help="Verwijder bestaande database")
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Genereer AI metadata (titel, fase, leefregels) via lokale LLM",
    )

    args = parser.parse_args()

    ingest(
        folder_dir=args.folder_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        provider=args.provider,
        reset=args.reset,
        enrich=args.enrich,
    )
