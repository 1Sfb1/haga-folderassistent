"""
STAP 1: Database Sanity Check
================================
Draai dit DIRECT na het chunken om te controleren of je data schoon is.

Controleert:
  ✓ Aantal chunks en unieke folders in de database
  ✓ Gemiddelde chunk-lengte (te kort = context verlies, te lang = ruis)
  ✓ Disclaimervervuiling (meertalig blok dat in elke folder zit)
  ✓ Duplicate chunks (identieke tekst = verspilde embeddings)
  ✓ Lege of ultra-korte chunks
  ✓ Voorbeeld-chunks voor visuele inspectie

Gebruik:
    python 1_sanity_check.py
"""

import os
import sys
from collections import Counter

import chromadb
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "haga_folders")

# Disclaimer-fragmenten die NIET in chunks horen
CONTAMINATION_MARKERS = [
    "Spreekt u geen of slecht Nederlands",
    "Do you speak Dutch poorly",
    "Czy Państwa znajomość języka",
    "Hollandaca dilini hiç konuşamıyor",
    "إذا كنتم لا تتحدثون",
    "folders.hagaziekenhuis.nl/2228",
    "Wat vindt u van deze patiënteninformatie",
]


def main():
    print("=" * 60)
    print("🔍 STAP 1: Database Sanity Check")
    print("=" * 60)

    # ── Connectie ──
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"\n❌ Kan ChromaDB niet openen: {e}")
        print(f"   Verwacht database in: {CHROMA_PERSIST_DIR}")
        print("   Heb je ingest.py al gedraaid?")
        sys.exit(1)

    total_chunks = collection.count()
    print(f"\n📊 Totaal chunks in database: {total_chunks}")

    if total_chunks == 0:
        print("❌ Database is leeg! Draai eerst ingest.py.")
        sys.exit(1)

    # ── Haal een sample op (max 10000 voor analyse) ──
    sample_size = min(total_chunks, 10000)
    print(f"   Analyseer {sample_size} chunks...\n")

    results = collection.get(
        limit=sample_size,
        include=["documents", "metadatas"],
    )

    documents = results["documents"]
    metadatas = results["metadatas"]

    # ── 1. Basis statistieken ──
    print("─" * 40)
    print("1️⃣  BASIS STATISTIEKEN")
    print("─" * 40)

    filenames = set(m.get("filename", "") for m in metadatas)
    print(f"   Unieke folders (PDFs):  {len(filenames)}")
    print(f"   Totaal chunks:          {total_chunks}")
    print(f"   Gem. chunks per folder: {total_chunks / max(len(filenames), 1):.1f}")

    lengths = [len(doc) for doc in documents]
    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    print(f"   Chunk-lengte (chars):   min={min_len}, gem={avg_len:.0f}, max={max_len}")

    # Beoordeling
    if avg_len < 200:
        print("   ⚠ WAARSCHUWING: Gemiddelde chunk is erg kort — verhoog chunk_size")
    elif avg_len > 4000:
        print("   ⚠ WAARSCHUWING: Gemiddelde chunk is erg lang — verlaag chunk_size")
    else:
        print("   ✓ Chunk-lengte ziet er goed uit")

    # ── 2. Lege / ultra-korte chunks ──
    print(f"\n{'─' * 40}")
    print("2️⃣  LEGE & ULTRA-KORTE CHUNKS")
    print("─" * 40)

    empty = [i for i, doc in enumerate(documents) if len(doc.strip()) < 20]
    short = [i for i, doc in enumerate(documents) if 20 <= len(doc.strip()) < 100]

    print(f"   Leeg (<20 chars):       {len(empty)}")
    print(f"   Erg kort (20-100):      {len(short)}")

    if empty:
        print(f"   ⚠ {len(empty)} lege chunks gevonden — deze vervuilen je retrieval")
        for idx in empty[:3]:
            print(f"     → [{metadatas[idx].get('filename', '?')}]: '{documents[idx][:50]}'")
    else:
        print("   ✓ Geen lege chunks")

    # ── 3. Disclaimervervuiling ──
    print(f"\n{'─' * 40}")
    print("3️⃣  DISCLAIMER-VERVUILING")
    print("─" * 40)

    contaminated = []
    for i, doc in enumerate(documents):
        for marker in CONTAMINATION_MARKERS:
            if marker.lower() in doc.lower():
                contaminated.append((i, marker))
                break

    contamination_rate = len(contaminated) / len(documents) * 100
    print(f"   Chunks met disclaimer-tekst: {len(contaminated)} ({contamination_rate:.1f}%)")

    if contamination_rate > 5:
        print("   ⚠ PROBLEEM: Disclaimertekst zit nog in >5% van de chunks")
        print("   → Dit vervuilt je embeddings — alle folders gaan op elkaar lijken")
        print("   → Fix: verbeter clean_haga_text() in ingest.py en draai opnieuw")
        print(f"\n   Voorbeelden:")
        for idx, marker in contaminated[:3]:
            fname = metadatas[idx].get("filename", "?")
            # Toon context rond de marker
            pos = documents[idx].lower().find(marker.lower())
            snippet = documents[idx][max(0, pos - 30):pos + 60]
            print(f"     → [{fname}]: ...{snippet}...")
    elif contaminated:
        print(f"   ⚡ Minimale vervuiling ({len(contaminated)} chunks) — acceptabel")
    else:
        print("   ✓ Geen disclaimervervuiling gevonden — clean_haga_text() werkt goed!")

    # ── 4. Duplicate chunks ──
    print(f"\n{'─' * 40}")
    print("4️⃣  DUPLICATE CHUNKS")
    print("─" * 40)

    # Hash eerste 200 chars van elk document
    hashes = Counter(doc[:200] for doc in documents)
    duplicates = {h: c for h, c in hashes.items() if c > 1}
    total_dupes = sum(c - 1 for c in duplicates.values())

    print(f"   Exacte duplicaten:      {total_dupes}")
    if total_dupes > len(documents) * 0.05:
        print(f"   ⚠ >5% duplicaten — mogelijk dezelfde folders meerdere keren geïngest")
        for text_start, count in list(duplicates.items())[:3]:
            print(f"     → {count}x: '{text_start[:80]}...'")
    else:
        print("   ✓ Weinig tot geen duplicaten")

    # ── 5. Top folders (meeste chunks) ──
    print(f"\n{'─' * 40}")
    print("5️⃣  FOLDERS MET MEESTE CHUNKS (top 10)")
    print("─" * 40)

    folder_counts = Counter(m.get("filename", "?") for m in metadatas)
    for fname, count in folder_counts.most_common(10):
        print(f"   {count:4d} chunks  ←  {fname}")

    # ── 6. Voorbeeld chunks (visuele inspectie) ──
    print(f"\n{'─' * 40}")
    print("6️⃣  VOORBEELD CHUNKS (eerste 3)")
    print("─" * 40)

    for i in range(min(3, len(documents))):
        fname = metadatas[i].get("filename", "?")
        title = metadatas[i].get("title", "?")
        chunk_idx = metadatas[i].get("chunk_index", "?")
        print(f"\n   📄 {fname} (chunk {chunk_idx})")
        print(f"   Titel: {title}")
        print(f"   Lengte: {len(documents[i])} chars")
        print(f"   Preview:")
        preview = documents[i][:300].replace("\n", " ↵ ")
        print(f"   {preview}...")

    # ── SAMENVATTING ──
    print(f"\n{'=' * 60}")
    print("📋 SAMENVATTING")
    print("=" * 60)

    issues = 0
    if len(empty) > 0:
        print("   ⚠ Lege chunks verwijderen of chunk-logica fixen")
        issues += 1
    if contamination_rate > 5:
        print("   ⚠ Disclaimervervuiling aanpakken in clean_haga_text()")
        issues += 1
    if total_dupes > len(documents) * 0.05:
        print("   ⚠ Duplicaten verwijderen of ingestion opnieuw draaien met --reset")
        issues += 1
    if avg_len < 200 or avg_len > 4000:
        print("   ⚠ Chunk-grootte aanpassen")
        issues += 1

    if issues == 0:
        print("   ✅ Alles ziet er goed uit! Ga door naar stap 2.")
    else:
        print(f"\n   {issues} probleem(en) gevonden — fix deze voordat je verder gaat.")
        print("   Na het fixen: python ingest.py --folder-dir ./folders/ --reset")


if __name__ == "__main__":
    main()
