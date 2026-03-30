"""
HAGA Folder RAG — Enrichment Patch
====================================
Voegt LLM-gegenereerde metadata (titel, fase, is_leefregel) toe aan
bestaande chunks in ChromaDB ZONDER opnieuw te embedden.

Dit script is bedoeld voor de situatie waarin de ingest al is gedraaid
zonder --enrich, en je alleen de metadata wilt bijwerken.

Gebruik:
    python enrich_existing.py
    python enrich_existing.py --dry-run          # alleen tonen, niet schrijven
    python enrich_existing.py --batch-size 100   # meer chunks per DB-update
"""

import argparse
import json
import os
import re
import sys
import time

import chromadb
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Import enrichment-functies uit ingest.py ──
from ingest import (
    GELDIGE_FASES,
    ENRICHMENT_CACHE_PATH,
    _laad_cache,
    _sla_cache_op,
    genereer_metadata_lokaal,
)


def main():
    parser = argparse.ArgumentParser(description="Enrichment patch voor bestaande ChromaDB")
    parser.add_argument("--dry-run", action="store_true", help="Toon wat er zou gebeuren zonder te schrijven")
    parser.add_argument("--batch-size", type=int, default=500, help="Chunks per DB-update batch")
    parser.add_argument("--limit", type=int, default=0, help="Max aantal folders om te verrijken (0=alles)")
    args = parser.parse_args()

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("COLLECTION_NAME", "haga_folders")

    print("=" * 70)
    print("✨ HAGA Enrichment Patch — Metadata toevoegen aan bestaande chunks")
    print("=" * 70)

    if args.dry_run:
        print("   ⚠️  DRY RUN — er wordt niets geschreven\n")

    # ── 1. Database laden ──
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(name=collection_name)
        total_chunks = collection.count()
        print(f"\n📦 Database: {total_chunks} chunks in '{collection_name}'")
    except Exception as e:
        print(f"❌ Database niet gevonden: {e}")
        sys.exit(1)

    # ── 2. Alle metadata ophalen ──
    print("📊 Alle metadata ophalen...")
    all_ids = []
    all_metadatas = []
    all_documents = []
    offset = 0
    fetch_batch = 5000

    while offset < total_chunks:
        result = collection.get(
            limit=fetch_batch,
            offset=offset,
            include=["metadatas", "documents"],
        )
        all_ids.extend(result["ids"])
        all_metadatas.extend(result["metadatas"])
        all_documents.extend(result["documents"])
        offset += fetch_batch

    print(f"   {len(all_ids)} chunks opgehaald")

    # ── 3. Groepeer per folder (filename) ──
    # We hebben per folder maar één LLM-call nodig, niet per chunk
    folders: dict[str, dict] = {}  # filename → {chunk_indices, first_text, title}
    for i, meta in enumerate(all_metadatas):
        fname = meta.get("filename", "")
        if fname not in folders:
            folders[fname] = {
                "indices": [],
                "first_text": all_documents[i] or "",
                "title": meta.get("title", fname),
                "has_enrichment": bool(meta.get("fase")),
            }
        folders[fname]["indices"].append(i)

    # Filter: alleen folders die nog geen enrichment hebben
    to_enrich = {k: v for k, v in folders.items() if not v["has_enrichment"]}
    already_done = len(folders) - len(to_enrich)

    print(f"\n   📁 Unieke folders: {len(folders)}")
    print(f"   ✅ Al verrijkt:    {already_done}")
    print(f"   🔄 Te verrijken:   {len(to_enrich)}")

    if not to_enrich:
        print("\n✅ Alle folders zijn al verrijkt! Niets te doen.")
        return

    if args.limit > 0:
        # Beperk tot --limit folders
        filenames = list(to_enrich.keys())[:args.limit]
        to_enrich = {k: to_enrich[k] for k in filenames}
        print(f"   ⚡ Beperkt tot {args.limit} folders (--limit)")

    # ── 4. Laad enrichment cache ──
    cache = _laad_cache()
    cache_hits = 0

    # ── 5. Genereer metadata per folder ──
    print(f"\n🤖 LLM metadata genereren voor {len(to_enrich)} folders...")
    start = time.time()

    updated_ids = []
    updated_metadatas = []
    folder_count = 0
    errors = 0

    for fname, info in to_enrich.items():
        folder_count += 1

        # Check cache
        if fname in cache:
            ai_meta = cache[fname]
            cache_hits += 1
        else:
            # Genereer via Ollama
            ai_meta = genereer_metadata_lokaal(info["first_text"], info["title"])
            cache[fname] = ai_meta

            # Tussentijds opslaan
            if folder_count % 50 == 0:
                _sla_cache_op(cache)

        # Toon voortgang
        if folder_count <= 5 or folder_count % 100 == 0 or folder_count == len(to_enrich):
            titel = ai_meta["patient_friendly_title"][:40]
            fase = ai_meta["fase"]
            lr = "✓" if ai_meta["is_leefregel"] else "✗"
            print(
                f"  [{folder_count:>4}/{len(to_enrich)}] {fname[:45]:<45} "
                f"→ {titel:<40} │ {fase:<14} │ lr={lr}"
            )

        # Update metadata voor alle chunks van deze folder
        for idx in info["indices"]:
            chunk_id = all_ids[idx]
            meta = all_metadatas[idx].copy()
            meta["patient_friendly_title"] = ai_meta["patient_friendly_title"]
            meta["fase"] = ai_meta["fase"]
            meta["is_leefregel"] = ai_meta["is_leefregel"]

            updated_ids.append(chunk_id)
            updated_metadatas.append(meta)

    # Sla cache definitief op
    _sla_cache_op(cache)
    elapsed_enrich = time.time() - start
    print(f"\n   ⏱  Enrichment: {elapsed_enrich:.1f}s ({cache_hits} uit cache)")

    # ── 6. Schrijf updates naar ChromaDB ──
    if args.dry_run:
        print(f"\n🏁 DRY RUN voltooid — {len(updated_ids)} chunks zouden bijgewerkt worden")
        print(f"   Cache opgeslagen met {len(cache)} folders")
        return

    print(f"\n💾 {len(updated_ids)} chunks bijwerken in ChromaDB...")
    start_write = time.time()

    for i in range(0, len(updated_ids), args.batch_size):
        end = min(i + args.batch_size, len(updated_ids))
        batch_ids = updated_ids[i:end]
        batch_metas = updated_metadatas[i:end]

        collection.update(
            ids=batch_ids,
            metadatas=batch_metas,
        )
        print(f"  ✓ {end}/{len(updated_ids)} chunks bijgewerkt")

    elapsed_write = time.time() - start_write
    total = time.time() - start

    # ── 7. Samenvatting ──
    print(f"\n{'=' * 70}")
    print(f"✅ Enrichment patch voltooid!")
    print(f"   Folders verrijkt:  {folder_count} ({cache_hits} uit cache)")
    print(f"   Chunks bijgewerkt: {len(updated_ids)}")
    print(f"   Tijd enrichment:   {elapsed_enrich:.1f}s")
    print(f"   Tijd DB-update:    {elapsed_write:.1f}s")
    print(f"   Cache:             {len(cache)} folders opgeslagen")
    print(f"\n   Draai verify_enrichment.py om het resultaat te controleren.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
