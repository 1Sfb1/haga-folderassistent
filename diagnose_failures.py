"""
Diagnose: Similarity scores voor gefaalde RAG-vragen
=====================================================
Toont de top-25 resultaten (met scores) voor de twee vragen die falen,
zodat je kunt zien of TOP_K=20 ze zou oppakken.

Gebruik:
    python diagnose_failures.py
"""

import os
import sys
from dotenv import load_dotenv
import chromadb

load_dotenv()

# ─────────────────────────────────────────────
# GEFAALDE VRAGEN
# ─────────────────────────────────────────────

FAILING_QUESTIONS = [
    {
        "question": "Wat mag ik drinken 3 uur voor mijn operatie op de daglounge?",
        "target_keywords": ["water", "thee", "nuchter", "drinken"],
        "target_source_hint": "daglounge",
    },
    {
        "question": "Kan ik mijn afspraak annuleren via WhatsApp?",
        "target_keywords": ["bellen", "afzeggen", "annuleren", "afspraak"],
        "target_source_hint": "afspraak",
    },
]

TOP_K = 25  # Haal ruim op zodat we kunnen zien waar de juiste bron staat


def main():
    # ── Load DB ──
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("COLLECTION_NAME", "haga_folders")

    print("📦 ChromaDB laden...")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name)
    print(f"   ✓ {collection.count()} chunks\n")

    # ── Load Embedder ──
    print("🧠 Embedding model laden...")
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    if provider == "openai":
        from ingest import OpenAIEmbedder
        embedder = OpenAIEmbedder()
    else:
        from ingest import LocalEmbedder
        embedder = LocalEmbedder()
    print("   ✓ Klaar\n")

    is_e5 = "e5" in os.getenv("EMBEDDING_MODEL", "").lower()

    # ── Diagnose per vraag ──
    for fq in FAILING_QUESTIONS:
        question = fq["question"]
        keywords = fq["target_keywords"]
        source_hint = fq["target_source_hint"]

        print("=" * 70)
        print(f"🔍 VRAAG: {question}")
        print(f"   Verwachte keywords: {keywords}")
        print(f"   Bron-hint: *{source_hint}*")
        print("=" * 70)

        # Embed
        q_text = f"query: {question}" if is_e5 else question
        q_embedding = embedder.embed([q_text], batch_size=1)[0]

        # Query
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        # ── Analyseer resultaten ──
        found_target = False
        keyword_positions = {}  # keyword -> eerste positie waar het voorkomt

        for rank in range(len(results["ids"][0])):
            dist = results["distances"][0][rank]
            sim = 1 - dist
            text = results["documents"][0][rank]
            meta = results["metadatas"][0][rank]
            fname = meta.get("filename", "?")
            cidx = meta.get("chunk_index", "?")
            title = meta.get("title", "?")[:50]
            preview = text[:100].replace("\n", " ")

            # Check of dit de doelbron is
            is_target = source_hint.lower() in fname.lower()
            target_marker = " ← 🎯 TARGET" if is_target else ""
            if is_target:
                found_target = True

            # Check keywords in deze chunk
            text_lower = text.lower()
            kw_found = [kw for kw in keywords if kw.lower() in text_lower]
            kw_marker = f"  [bevat: {', '.join(kw_found)}]" if kw_found else ""

            for kw in kw_found:
                if kw not in keyword_positions:
                    keyword_positions[kw] = rank + 1

            # Kleurcode op basis van similarity
            if sim >= 0.50:
                score_icon = "🟢"
            elif sim >= 0.30:
                score_icon = "🟡"
            else:
                score_icon = "🔴"

            # Print alleen top-10 volledig, daarna alleen als het relevant is
            if rank < 10 or is_target or kw_found:
                print(f"\n   #{rank+1:2d}  {score_icon} [{sim:.3f}]  {fname} (chunk {cidx})")
                print(f"        {preview}...")
                if target_marker or kw_marker:
                    print(f"       {target_marker}{kw_marker}")

        # ── Samenvatting ──
        print(f"\n{'─' * 70}")
        print(f"📊 SAMENVATTING voor: {question[:50]}...")

        if found_target:
            # Vind de rank van de target bron
            for rank in range(len(results["ids"][0])):
                fname = results["metadatas"][0][rank].get("filename", "")
                if source_hint.lower() in fname.lower():
                    sim = 1 - results["distances"][0][rank]
                    print(f"   🎯 Doelbron gevonden op positie #{rank+1} (sim={sim:.3f})")
                    if rank + 1 <= 5:
                        print(f"      ✅ Zou gevonden worden bij TOP_K=5")
                    elif rank + 1 <= 10:
                        print(f"      ⚡ Zou gevonden worden bij TOP_K=10")
                    elif rank + 1 <= 20:
                        print(f"      🔶 Zou gevonden worden bij TOP_K=20")
                    else:
                        print(f"      ❌ Buiten TOP_K=20 — retrieval probleem")
                    break
        else:
            print(f"   ❌ Doelbron (*{source_hint}*) NIET in top-{TOP_K}")

        print(f"\n   Keyword-dekking in top-{TOP_K}:")
        for kw in keywords:
            if kw in keyword_positions:
                print(f"     ✅ '{kw}' eerst gevonden op positie #{keyword_positions[kw]}")
            else:
                print(f"     ❌ '{kw}' NIET gevonden in top-{TOP_K}")

        # Check: welke relevance threshold zou nodig zijn?
        all_sims = [1 - results["distances"][0][i] for i in range(len(results["ids"][0]))]
        print(f"\n   Similarity range: {min(all_sims):.3f} – {max(all_sims):.3f}")
        print(f"   Huidige RELEVANCE_THRESHOLD: {os.getenv('RELEVANCE_THRESHOLD', '0.30')}")

        # Hoeveel chunks overleven threshold=0.30?
        threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))
        surviving = sum(1 for s in all_sims if s >= threshold)
        print(f"   Chunks boven threshold: {surviving}/{TOP_K}")

        print()

    print("\n✅ Diagnose compleet.")
    print("   → Als de doelbron buiten TOP_K=10 maar binnen TOP_K=20 valt,")
    print("     lost het verhogen van TOP_K het probleem op.")
    print("   → Als de doelbron helemaal niet in top-25 zit,")
    print("     is er een dieper retrieval/embedding probleem.")


if __name__ == "__main__":
    main()
