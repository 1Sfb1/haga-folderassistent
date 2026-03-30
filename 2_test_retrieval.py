"""
STAP 2: Retrieval Quality Test
================================
Test de vector-zoeklaag ZONDER LLM. Dit isoleert retrieval-problemen
van generatie-problemen.

Per vraag toont dit script:
  - Top-5 teruggehaalde chunks met similarity score
  - Of de verwachte bron-folder in de resultaten zit
  - Mean Reciprocal Rank (MRR) over alle vragen

Gebruik:
    python 2_test_retrieval.py                  # naïeve retrieval
    python 2_test_retrieval.py --hyde           # HyDE retrieval
    python 2_test_retrieval.py --compare        # HyDE vs naïef naast elkaar
    python 2_test_retrieval.py --top-k 10
"""

import argparse
import json
import os
import sys
from datetime import datetime

import chromadb
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# GOUDEN TEST VRAGEN
# ─────────────────────────────────────────────
# Pas deze aan naar folders die JIJ in je database hebt!
# Format: (vraag, verwachte_folder_substring, verwacht_antwoord_bevat)
#
# De verwachte_folder_substring is een deel van de filename die je
# verwacht terug te zien in de top resultaten.
# Zet op None als je geen specifieke folder verwacht.

GOLDEN_QUESTIONS = [
    # ══════════════════════════════════════════════
    # FEITELIJKE VRAGEN (exact antwoord verwacht)
    # ══════════════════════════════════════════════
    {
        "question": "Wat zijn de openingstijden van de daglounge in Zoetermeer?",
        "expected_source": "daglounge",
        "expected_keywords": ["07.30", "18.00", "maandag"],
        "category": "feitelijk",
    },
    {
        "question": "Wat is het telefoonnummer van de polikliniek Urologie?",
        "expected_source": "bcg",
        "expected_keywords": ["070", "210 6482"],
        "category": "feitelijk",
    },
    {
        "question": "Waar kan ik contact opnemen bij brandwonden buiten kantooruren?",
        "expected_source": "brandwonden",
        "expected_keywords": ["Spoedeisende Hulp", "079"],
        "category": "feitelijk",
    },
    {
        "question": "Wat is het telefoonnummer van de Cardiolounge?",
        "expected_source": "cardioversie",
        "expected_keywords": ["070", "210 1847"],
        "category": "feitelijk",
    },
    {
        "question": "Hoe lang duurt een BAEP onderzoek?",
        "expected_source": "baep",
        "expected_keywords": ["30", "45", "minuten"],
        "category": "feitelijk",
    },

    # ══════════════════════════════════════════════
    # PROCEDURELE VRAGEN (stappen/instructies)
    # ══════════════════════════════════════════════
    {
        "question": "Hoe bereid ik me voor op een BCG blaasspoeling?",
        "expected_source": "bcg",
        "expected_keywords": ["nuchter", "drinken", "vier uur"],
        "category": "procedureel",
    },
    {
        "question": "Wat moet ik doen na het plassen van een BCG blaasspoeling?",
        "expected_source": "bcg",
        "expected_keywords": ["zittend", "wc", "handen"],
        "category": "procedureel",
    },
    {
        "question": "Hoelang moet ik nuchter zijn voor een elektrische cardioversie?",
        "expected_source": "cardioversie",
        "expected_keywords": ["6 uur", "nuchter"],
        "category": "procedureel",
    },
    {
        "question": "Mag ik auto rijden na een elektrische cardioversie?",
        "expected_source": "cardioversie",
        "expected_keywords": ["niet", "autorijden"],
        "category": "procedureel",
    },
    {
        "question": "Hoe lang moet ik het drukverband dragen na een elleboogzenuw operatie?",
        "expected_source": "elleboogzenuw",
        "expected_keywords": ["dagen", "verband"],
        "category": "procedureel",
    },
    {
        "question": "Welke medicijnen mag ik niet gebruiken voor een colonoscopie?",
        "expected_source": "colonoscopie",
        "expected_keywords": ["bloedverdunner", "ijzer", "aspirine"],
        "category": "procedureel",
    },
    {
        "question": "Hoe gebruik ik EMLA zalf bij mijn kind?",
        "expected_source": "emla",
        "expected_keywords": ["zalf", "pleister", "uur"],
        "category": "procedureel",
    },

    # ══════════════════════════════════════════════
    # MEDISCHE VRAGEN (aandoeningen/behandelingen)
    # ══════════════════════════════════════════════
    {
        "question": "Wat is hidradenitis suppurativa?",
        "expected_source": "hidradenitis",
        "expected_keywords": ["huid", "ontsteking", "oksel"],
        "category": "medisch",
    },
    {
        "question": "Hoe werkt een fibroscan?",
        "expected_source": "fibroscan",
        "expected_keywords": ["lever", "littekenweefsel", "trillingen"],
        "category": "medisch",
    },
    {
        "question": "Wat is een intravitreale injectie?",
        "expected_source": "injectie",
        "expected_keywords": ["oogbol", "glasvocht", "injectie"],
        "category": "medisch",
    },
    {
        "question": "Wat zijn de bijwerkingen van methotrexaat bij kinderen?",
        "expected_source": "methotrexaat",
        "expected_keywords": ["misselijkheid", "lever", "bloedwaarden"],
        "category": "medisch",
    },

    # ══════════════════════════════════════════════
    # NAZORG / HERSTEL VRAGEN
    # ══════════════════════════════════════════════
    {
        "question": "Wanneer mag ik weer sporten na een borstoperatie?",
        "expected_source": "borst",
        "expected_keywords": ["weken", "sport", "bewegen"],
        "category": "nazorg",
    },
    {
        "question": "Hoe herken ik tekenen van een wondinfectie?",
        "expected_source": None,
        "expected_keywords": ["rood", "warm", "zwelling", "pus"],
        "category": "nazorg",
    },
    {
        "question": "Mag ik douchen na de operatie?",
        "expected_source": None,
        "expected_keywords": ["douchen", "wond", "droog"],
        "category": "nazorg",
    },

    # ══════════════════════════════════════════════
    # CROSS-FOLDER VRAGEN (meerdere folders relevant)
    # ══════════════════════════════════════════════
    {
        "question": "Welke voeding is goed na een darmoperatie?",
        "expected_source": "darm",
        "expected_keywords": ["voeding", "vezels", "vocht"],
        "category": "cross-folder",
    },
    {
        "question": "Hoe kan ik pijn bestrijden na een ingreep?",
        "expected_source": None,
        "expected_keywords": ["pijnstiller", "paracetamol"],
        "category": "cross-folder",
    },

    # ══════════════════════════════════════════════
    # SPECIFIEKE APPS & HULPMIDDELEN
    # ══════════════════════════════════════════════
    {
        "question": "Wat is de Virtual Fracture Care app?",
        "expected_source": "fracture",
        "expected_keywords": ["botbreuk", "app", "oefeningen"],
        "category": "hulpmiddel",
    },
    {
        "question": "Wat is een BAEP onderzoek?",
        "expected_source": "baep",
        "expected_keywords": ["gehoorzenuw", "koptelefoon", "elektroden"],
        "category": "medisch",
    },
]


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

def load_embedder():
    """Laad het embedding model."""
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    if provider == "openai":
        from ingest import OpenAIEmbedder
        return OpenAIEmbedder()
    else:
        from ingest import LocalEmbedder
        return LocalEmbedder()


def retrieve_naive(collection, embedder, query: str, top_k: int = 5) -> list[dict]:
    """Naïeve vector search — embed de ruwe query direct."""
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
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "similarity": round(1 - results["distances"][0][i], 3),
        })
    return chunks


# ─────────────────────────────────────────────
# EVALUATIE
# ─────────────────────────────────────────────

def evaluate_retrieval(results: list[dict], test_case: dict) -> dict:
    """Evalueer retrieval resultaten voor één test case."""
    expected_source = test_case.get("expected_source")
    expected_keywords = test_case.get("expected_keywords", [])

    # Check 1: Zit de verwachte bron in de resultaten?
    source_found = False
    source_rank = None
    if expected_source:
        for rank, r in enumerate(results, 1):
            fname = r["metadata"].get("filename", "").lower()
            if expected_source.lower() in fname:
                source_found = True
                source_rank = rank
                break

    # Check 2: Bevatten de resultaten de verwachte keywords?
    all_text = " ".join(r["text"].lower() for r in results)
    keywords_found = [kw for kw in expected_keywords if kw.lower() in all_text]
    keywords_missing = [kw for kw in expected_keywords if kw.lower() not in all_text]

    # Check 3: Similarity scores
    top_similarity = results[0]["similarity"] if results else 0
    avg_similarity = sum(r["similarity"] for r in results) / len(results) if results else 0

    return {
        "source_found": source_found,
        "source_rank": source_rank,
        "keywords_found": keywords_found,
        "keywords_missing": keywords_missing,
        "keyword_recall": len(keywords_found) / max(len(expected_keywords), 1),
        "top_similarity": top_similarity,
        "avg_similarity": avg_similarity,
    }


def print_results_block(results: list[dict], test: dict, label: str = "") -> dict:
    """Print resultaten voor één vraag en geef evaluatie terug."""
    if label:
        print(f"   ── {label} ──")
    for rank, r in enumerate(results, 1):
        fname = r["metadata"].get("filename", "?")
        sim = r["similarity"]
        preview = r["text"][:100].replace("\n", " ")
        marker = ""
        if test["expected_source"] and test["expected_source"].lower() in fname.lower():
            marker = " ← ✓"
        print(f"   #{rank}  [{sim:.3f}]  {fname}{marker}")
        print(f"         {preview}...")
        print()
    return evaluate_retrieval(results, test)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HAGA RAG — Retrieval Test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save", action="store_true", help="Sla resultaten op als JSON")
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="Gebruik HyDE retrieval (extra LLM call per vraag)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Vergelijk HyDE vs naïef naast elkaar",
    )
    args = parser.parse_args()

    use_hyde = args.hyde or args.compare

    print("=" * 60)
    mode = "HyDE" if args.hyde and not args.compare else ("HyDE vs Naïef" if args.compare else "Naïef")
    print(f"🔍 STAP 2: Retrieval Quality Test — {mode}")
    print("=" * 60)

    # Load database
    try:
        client = chromadb.PersistentClient(path=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
        collection = client.get_collection(name=os.getenv("COLLECTION_NAME", "haga_folders"))
        print(f"📦 Database geladen: {collection.count()} chunks")
    except Exception as e:
        print(f"❌ Database niet gevonden: {e}")
        sys.exit(1)

    # Load embedder
    print("🧠 Embedding model laden...")
    embedder = load_embedder()
    print("   ✓ Klaar")

    # Laad HyDE retriever indien nodig
    hyde_retriever = None
    if use_hyde:
        try:
            from hyde_retriever import HyDERetriever
            hyde_retriever = HyDERetriever(embedder)
            print("🔮 HyDE retriever geladen")
        except ImportError:
            print("❌ hyde_retriever.py niet gevonden. Zet het in dezelfde map.")
            sys.exit(1)
    print()

    # ── Run tests ──
    all_evaluations_naive = []
    all_evaluations_hyde = []
    rr_naive = []
    rr_hyde = []

    for i, test in enumerate(GOLDEN_QUESTIONS, 1):
        print(f"{'─' * 60}")
        print(f"📝 Vraag {i}/{len(GOLDEN_QUESTIONS)}: {test['question']}")
        print(f"   Categorie: {test['category']}")
        if test["expected_source"]:
            print(f"   Verwachte bron: *{test['expected_source']}*")
        print()

        if args.compare:
            # ── Vergelijkingsmodus: beide methoden naast elkaar ──

            # Naïef
            naive_results = retrieve_naive(collection, embedder, test["question"], top_k=args.top_k)
            naive_eval = print_results_block(naive_results, test, label="NAÏEF")
            all_evaluations_naive.append({**test, **naive_eval})
            if test["expected_source"]:
                rr_naive.append(1.0 / naive_eval["source_rank"] if naive_eval["source_rank"] else 0.0)

            # HyDE
            hyde_result = hyde_retriever.retrieve_full(collection, test["question"], top_k=args.top_k)
            hyde_chunks = [
                {"text": c.text, "metadata": c.metadata, "similarity": c.similarity}
                for c in hyde_result.chunks
            ]
            print(f"   Hypothetisch doc ({hyde_result.hyde_latency_ms}ms): "
                  f"{hyde_result.hypothetical_doc[:120]}...")
            print()
            hyde_eval = print_results_block(hyde_chunks, test, label="HYDE")
            all_evaluations_hyde.append({**test, **hyde_eval})
            if test["expected_source"]:
                rr_hyde.append(1.0 / hyde_eval["source_rank"] if hyde_eval["source_rank"] else 0.0)

            # Wie wint voor deze vraag?
            naive_rr = 1.0 / naive_eval["source_rank"] if naive_eval["source_rank"] else 0.0
            hyde_rr = 1.0 / hyde_eval["source_rank"] if hyde_eval["source_rank"] else 0.0
            if hyde_rr > naive_rr:
                print("   ✅ HyDE scoort beter")
            elif naive_rr > hyde_rr:
                print("   📉 Naïef scoort beter")
            else:
                print("   🟰 Gelijkspel")

        elif args.hyde:
            # ── HyDE-only modus ──
            hyde_result = hyde_retriever.retrieve_full(collection, test["question"], top_k=args.top_k)
            print(f"   🧠 Hypothetisch document ({hyde_result.hyde_latency_ms}ms):")
            print(f"   {hyde_result.hypothetical_doc[:200]}")
            print()
            hyde_chunks = [
                {"text": c.text, "metadata": c.metadata, "similarity": c.similarity}
                for c in hyde_result.chunks
            ]
            evaluation = print_results_block(hyde_chunks, test)
            all_evaluations_hyde.append({**test, **evaluation})
            if test["expected_source"]:
                rr_hyde.append(1.0 / evaluation["source_rank"] if evaluation["source_rank"] else 0.0)

        else:
            # ── Naïef-only modus (origineel gedrag) ──
            results = retrieve_naive(collection, embedder, test["question"], top_k=args.top_k)
            evaluation = print_results_block(results, test)
            all_evaluations_naive.append({**test, **evaluation})
            if test["expected_source"]:
                rr_naive.append(1.0 / evaluation["source_rank"] if evaluation["source_rank"] else 0.0)

        print()

    # ── SAMENVATTING ──
    print("=" * 60)
    print("📋 SAMENVATTING RETRIEVAL TEST")
    print("=" * 60)

    def print_metrics(evaluations: list[dict], reciprocal_ranks: list[float], label: str):
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
        avg_recall = sum(e["keyword_recall"] for e in evaluations) / len(evaluations)
        avg_sim = sum(e["top_similarity"] for e in evaluations) / len(evaluations)
        failures = [e for e in evaluations if e.get("expected_source") and not e["source_found"]]

        print(f"\n  [{label}]")
        print(f"   Mean Reciprocal Rank (MRR):  {mrr:.3f}  ", end="")
        print("✅" if mrr > 0.8 else ("⚡" if mrr > 0.5 else "❌"))
        print(f"   Gem. keyword recall:         {avg_recall:.0%}")
        print(f"   Gem. top similarity:         {avg_sim:.3f}")

        if failures:
            print(f"   Gemiste bronnen ({len(failures)}):")
            for f in failures:
                print(f"     → '{f['question'][:45]}...' (verwacht: {f['expected_source']})")

        categories = set(e["category"] for e in evaluations)
        print(f"   Per categorie:")
        for cat in sorted(categories):
            cat_evals = [e for e in evaluations if e["category"] == cat]
            cat_recall = sum(e["keyword_recall"] for e in cat_evals) / len(cat_evals)
            print(f"     {cat:15s}: recall = {cat_recall:.0%}")

        return mrr, avg_recall, avg_sim

    if args.compare:
        mrr_n, rec_n, sim_n = print_metrics(all_evaluations_naive, rr_naive, "NAÏEF")
        mrr_h, rec_h, sim_h = print_metrics(all_evaluations_hyde, rr_hyde, "HYDE")

        print(f"\n  Verschil (HyDE − Naïef):")
        print(f"   MRR:      {mrr_h - mrr_n:+.3f}")
        print(f"   Recall:   {rec_h - rec_n:+.0%}")
        print(f"   Sim:      {sim_h - sim_n:+.3f}")

        if mrr_h > mrr_n:
            print("\n  ✅ HyDE verbetert retrieval — implementeer in de server")
        elif mrr_n > mrr_h:
            print("\n  ⚠ Naïef presteert beter — HyDE prompt verfijnen")
        else:
            print("\n  🟰 Vergelijkbaar — overweeg latency/kosten tradeoff")

    elif args.hyde:
        print_metrics(all_evaluations_hyde, rr_hyde, "HYDE")
    else:
        print_metrics(all_evaluations_naive, rr_naive, "NAÏEF")

    print()

    # Save results
    if args.save:
        output = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "top_k": args.top_k,
        }
        if all_evaluations_naive:
            mrr = sum(rr_naive) / len(rr_naive) if rr_naive else 0
            output["naive"] = {"mrr": mrr, "results": all_evaluations_naive}
        if all_evaluations_hyde:
            mrr = sum(rr_hyde) / len(rr_hyde) if rr_hyde else 0
            output["hyde"] = {"mrr": mrr, "results": all_evaluations_hyde}

        outfile = f"retrieval_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(outfile, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"💾 Resultaten opgeslagen: {outfile}")


if __name__ == "__main__":
    main()
