"""
HAGA Folder RAG — Enrichment Verificatie
==========================================
Controleert of de LLM-gegenereerde metadata (patient_friendly_title,
fase, is_leefregel) correct in ChromaDB is opgeslagen na ingest.

Gebruik:
    python verify_enrichment.py
    python verify_enrichment.py --sample 20
"""

import argparse
import json
import os
import sys
from collections import Counter

import chromadb
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Verificeer enrichment metadata in ChromaDB")
    parser.add_argument("--sample", type=int, default=10, help="Aantal voorbeeldchunks om te tonen")
    args = parser.parse_args()

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("COLLECTION_NAME", "haga_folders")

    print("=" * 70)
    print("🔍 HAGA Enrichment Verificatie")
    print("=" * 70)

    # ── 1. Database laden ──
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(name=collection_name)
        total_chunks = collection.count()
        print(f"\n📦 Database: {persist_dir}")
        print(f"   Collectie: {collection_name}")
        print(f"   Totaal chunks: {total_chunks}")
    except Exception as e:
        print(f"❌ Database niet gevonden: {e}")
        sys.exit(1)

    # ── 2. Metadata ophalen (in batches voor grote databases) ──
    print(f"\n📊 Metadata analyseren...")

    batch_size = 5000
    all_metadatas = []
    offset = 0

    # ChromaDB .get() met limit/offset
    while offset < total_chunks:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )
        all_metadatas.extend(result["metadatas"])
        offset += batch_size

    print(f"   {len(all_metadatas)} chunks metadata opgehaald")

    # ── 3. Check welke enrichment-velden aanwezig zijn ──
    has_title = 0
    has_fase = 0
    has_leefregel = 0
    unique_folders = set()
    fase_counter = Counter()
    leefregel_count = 0
    missing_enrichment = []

    for meta in all_metadatas:
        fname = meta.get("filename", "?")
        unique_folders.add(fname)

        if meta.get("patient_friendly_title"):
            has_title += 1
        if meta.get("fase"):
            has_fase += 1
            fase_counter[meta["fase"]] += 1
        if "is_leefregel" in meta:
            has_leefregel += 1
            if meta["is_leefregel"]:
                leefregel_count += 1

        # Track folders zonder enrichment
        if not meta.get("fase") and not meta.get("patient_friendly_title"):
            if fname not in [m[0] for m in missing_enrichment]:
                missing_enrichment.append((fname, meta.get("title", "?")))

    # ── 4. Resultaten tonen ──
    print(f"\n{'─' * 70}")
    print(f"📋 ENRICHMENT DEKKING")
    print(f"{'─' * 70}")
    print(f"   Unieke folders:          {len(unique_folders)}")
    print(f"   patient_friendly_title:  {has_title}/{len(all_metadatas)} chunks ({100*has_title/len(all_metadatas):.1f}%)")
    print(f"   fase:                    {has_fase}/{len(all_metadatas)} chunks ({100*has_fase/len(all_metadatas):.1f}%)")
    print(f"   is_leefregel:            {has_leefregel}/{len(all_metadatas)} chunks ({100*has_leefregel/len(all_metadatas):.1f}%)")

    if has_fase == 0 and has_title == 0:
        print(f"\n   ⚠️  GEEN enrichment metadata gevonden!")
        print(f"      Heb je `python ingest.py --folder-dir ./folders/ --enrich` gedraaid?")
        print(f"      Zonder --enrich worden fase en is_leefregel niet gegenereerd.")
        sys.exit(1)

    # ── 5. Fase-verdeling ──
    print(f"\n{'─' * 70}")
    print(f"📊 FASE-VERDELING (chunks)")
    print(f"{'─' * 70}")

    total_fase = sum(fase_counter.values()) or 1
    for fase in ["Algemeen", "Voorbereiding", "Onderzoek", "Behandeling", "Nazorg"]:
        count = fase_counter.get(fase, 0)
        pct = 100 * count / total_fase
        bar = "█" * int(pct / 2) or "▏"
        print(f"   {fase:<16} {count:>5} ({pct:5.1f}%)  {bar}")

    onbekend = sum(v for k, v in fase_counter.items() if k not in {"Algemeen", "Voorbereiding", "Onderzoek", "Behandeling", "Nazorg"})
    if onbekend > 0:
        print(f"   {'⚠ Onbekend':<16} {onbekend:>5}  ← LLM heeft ongeldige fases gegenereerd")

    # ── 6. Leefregels ──
    print(f"\n{'─' * 70}")
    print(f"📋 LEEFREGELS")
    print(f"{'─' * 70}")
    print(f"   Chunks met is_leefregel=true:  {leefregel_count}")
    print(f"   Chunks met is_leefregel=false: {has_leefregel - leefregel_count}")
    if leefregel_count > 0:
        pct = 100 * leefregel_count / has_leefregel
        print(f"   Percentage leefregels:         {pct:.1f}%")

    # ── 7. Voorbeelden tonen ──
    print(f"\n{'─' * 70}")
    print(f"📝 VOORBEELDEN (eerste {args.sample} unieke folders)")
    print(f"{'─' * 70}")

    seen_folders = set()
    examples_shown = 0

    for meta in all_metadatas:
        fname = meta.get("filename", "?")
        if fname in seen_folders:
            continue
        seen_folders.add(fname)

        titel = meta.get("patient_friendly_title", "—")
        fase = meta.get("fase", "—")
        leefregel = meta.get("is_leefregel", "—")
        folder_id = meta.get("folder_id", "?")

        print(f"\n   [{folder_id}] {fname}")
        print(f"   Originele titel:  {meta.get('title', '—')}")
        print(f"   AI titel:         {titel}")
        print(f"   Fase:             {fase}")
        print(f"   Leefregel:        {leefregel}")

        examples_shown += 1
        if examples_shown >= args.sample:
            break

    # ── 8. Samenvatting ──
    print(f"\n{'=' * 70}")
    all_good = has_fase > 0 and has_title > 0 and has_leefregel > 0
    if all_good:
        print(f"✅ Enrichment verificatie GESLAAGD")
        print(f"   Alle 3 metadata-velden zijn aanwezig in de database.")
    else:
        missing = []
        if has_title == 0:
            missing.append("patient_friendly_title")
        if has_fase == 0:
            missing.append("fase")
        if has_leefregel == 0:
            missing.append("is_leefregel")
        print(f"⚠️  Enrichment ONVOLLEDIG — ontbrekend: {', '.join(missing)}")
    print(f"{'=' * 70}")

    # ── 9. Check cache ──
    cache_path = os.getenv("ENRICHMENT_CACHE", "./enrichment_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
        print(f"\n💾 Enrichment cache: {cache_path}")
        print(f"   {len(cache)} folders in cache")
        cache_vs_db = len(unique_folders) - len(cache)
        if cache_vs_db > 0:
            print(f"   ⚠️  {cache_vs_db} folders in DB maar niet in cache")
    else:
        print(f"\n💾 Geen enrichment cache gevonden op {cache_path}")


if __name__ == "__main__":
    main()
