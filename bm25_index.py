"""
HAGA Folder RAG — BM25 Index
==============================
Sparse keyword-index als aanvulling op de dense vector search.

Waarom BM25 naast dense vectors?
  Dense vectors (multilingual-e5-large) zijn sterk in semantische gelijkenis
  maar missen exacte keyword-matches. Bij feitelijke vragen zoals:
    "telefoonnummer urologie"    → BM25 matcht direct op "urologie" + "telefoonnummer"
    "openingstijden daglounge"   → BM25 matcht direct op beide termen
    "079 346 25 47"              → BM25 vindt het exacte nummer, dense zoekt semantisch

  Dense vectors zijn beter bij:
    "wat moet ik doen na de ingreep?" → semantische match met "nazorg na operatie"

  Reciprocal Rank Fusion (RRF) combineert beide rangordes optimaal.

Gebruik:
    from bm25_index import BM25Index, reciprocal_rank_fusion

    # Eenmalig bouwen + opslaan:
    idx = BM25Index()
    idx.build(collection)          # haalt alle chunks uit ChromaDB
    idx.save("./bm25_index.pkl")   # ~5MB voor 6000 chunks

    # Daarna laden (snel, <0.5s):
    idx = BM25Index.load("./bm25_index.pkl")

    # Zoeken:
    bm25_results = idx.search("telefoonnummer urologie", top_k=20)

    # Fusie met dense resultaten:
    final = reciprocal_rank_fusion(dense_results, bm25_results, top_k=10)

Dependency:
    pip install rank-bm25
"""

import os
import pickle
import re
import time
from pathlib import Path


# ─────────────────────────────────────────────
# DUTCH TOKENIZER
# ─────────────────────────────────────────────

# Nederlandse stopwoorden — verwijder voor BM25 zodat "de folder" niet matcht
# op "de" maar op "folder". Bewust kort gehouden: te agressief filteren
# verwijdert medisch relevante korte woorden (bijv. "mg", "ml").
NL_STOPWORDS = {
    "de", "het", "een", "en", "van", "in", "is", "op", "dat", "die",
    "voor", "met", "aan", "er", "maar", "om", "te", "zijn", "ze", "ook",
    "als", "bij", "uit", "dan", "nog", "tot", "of", "door", "over", "naar",
    "heeft", "dit", "uw", "ik", "we", "hij", "zij", "hun", "hem", "haar",
    "ons", "zich", "wat", "wie", "hoe", "waar", "wanneer", "zal", "zou",
    "wordt", "werd", "waren", "was", "na", "al", "wel", "zo", "geen",
    "alle", "veel", "deze", "worden", "geweest", "toch", "want",
}


def tokenize(text: str) -> list[str]:
    """
    Tokeniseer Nederlandse tekst voor BM25.

    - Lowercase
    - Split op niet-alfanumerieke tekens (behalve koppelteken binnen woorden)
    - Verwijder stopwoorden
    - Verwijder tokens korter dan 2 tekens (behalve cijfers zoals "2x")

    Telefoonnummers (079, 6482) worden bewaard als losse tokens zodat
    exacte nummers direct gevonden worden.
    """
    text = text.lower()
    # Splits op spaties en leestekens, behoud cijferreeksen intact
    tokens = re.findall(r"[a-z\u00e0-\u017e]+|\d+", text)
    return [
        t for t in tokens
        if t not in NL_STOPWORDS and (len(t) >= 2 or t.isdigit())
    ]


# ─────────────────────────────────────────────
# BM25 INDEX
# ─────────────────────────────────────────────

class BM25Index:
    """
    BM25 sparse index over de volledige HAGA-foldercollectie.

    Persisteert naar disk zodat rebuild niet bij elke serverstart nodig is.
    Bij 6000 chunks duurt bouwen ~2s, laden ~0.1s.
    """

    def __init__(self):
        self._bm25 = None          # rank_bm25.BM25Okapi object
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._built_at: float = 0.0
        self._num_docs: int = 0

    # ──────────────────────────────────────────
    # Bouwen
    # ──────────────────────────────────────────

    def build(self, collection) -> None:
        """
        Haal alle chunks uit ChromaDB en bouw de BM25-index.

        ChromaDB heeft geen ingebouwde paginering — we halen alles op
        in één batch. Voor 6000-15000 chunks is dat prima (<500MB RAM).
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 niet geïnstalleerd. Voer uit: pip install rank-bm25"
            )

        t0 = time.perf_counter()
        print("🔍 BM25: alle chunks ophalen uit ChromaDB...")

        # Haal alles op — limit=None geeft alle documenten
        total = collection.count()
        # ChromaDB heeft een interne limiet, haal in batches van 5000
        batch_size = 5000
        all_docs: list[str] = []
        all_metas: list[dict] = []

        for offset in range(0, total, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            all_docs.extend(result["documents"])
            all_metas.extend(result["metadatas"])
            print(f"   {min(offset + batch_size, total)}/{total} chunks geladen")

        print(f"   ✓ {len(all_docs)} chunks opgehaald in {time.perf_counter()-t0:.1f}s")

        print("🔍 BM25: tokeniseren en index bouwen...")
        t1 = time.perf_counter()

        tokenized = [tokenize(doc) for doc in all_docs]
        self._bm25 = BM25Okapi(tokenized)
        self._texts = all_docs
        self._metas = all_metas
        self._built_at = time.time()
        self._num_docs = len(all_docs)

        elapsed = time.perf_counter() - t1
        print(f"   ✓ BM25 index klaar: {self._num_docs} docs in {elapsed:.1f}s")

    # ──────────────────────────────────────────
    # Persistentie
    # ──────────────────────────────────────────

    def save(self, path: str) -> None:
        """Sla de index op als pickle-bestand."""
        payload = {
            "bm25": self._bm25,
            "texts": self._texts,
            "metas": self._metas,
            "built_at": self._built_at,
            "num_docs": self._num_docs,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = Path(path).stat().st_size / 1_048_576
        print(f"💾 BM25 index opgeslagen: {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Laad een eerder opgeslagen index van disk."""
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            payload = pickle.load(f)
        idx = cls()
        idx._bm25 = payload["bm25"]
        idx._texts = payload["texts"]
        idx._metas = payload["metas"]
        idx._built_at = payload.get("built_at", 0.0)
        idx._num_docs = payload.get("num_docs", len(idx._texts))
        elapsed = time.perf_counter() - t0
        print(f"   ✓ BM25 index geladen: {idx._num_docs} docs in {elapsed:.2f}s")
        return idx

    @classmethod
    def load_or_build(cls, path: str, collection, rebuild: bool = False) -> "BM25Index":
        """
        Laad van disk als beschikbaar, bouw anders opnieuw.

        Parameters
        ----------
        path : str
            Pad naar het pickle-bestand (bijv. "./bm25_index.pkl").
        collection : chromadb.Collection
            ChromaDB collectie — alleen nodig als (her)bouwen.
        rebuild : bool
            Als True: altijd herbouwen, ook als het bestand bestaat.
        """
        if not rebuild and Path(path).exists():
            try:
                return cls.load(path)
            except Exception as e:
                print(f"   ⚠ BM25 laden mislukt ({e}), herbouwen...")

        idx = cls()
        idx.build(collection)
        idx.save(path)
        return idx

    # ──────────────────────────────────────────
    # Zoeken
    # ──────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Zoek de top-k chunks via BM25 keyword-score.

        Geeft dezelfde dict-structuur terug als retrieve() in server.py:
            [{"text": ..., "metadata": ..., "similarity": ..., "distance": ...}]

        De "similarity" is hier een genormaliseerde BM25-score (0-1),
        niet een cosine-similarity. Gebruik hem alleen voor ranking,
        niet als absolute drempel — BM25 en dense scores zijn niet
        direct vergelijkbaar.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index niet gebouwd. Roep eerst build() aan.")

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Top-k indices gesorteerd op score (hoogste eerst)
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        max_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 1.0
        if max_score == 0:
            return []  # Geen enkele match — query-termen komen niet voor

        results = []
        for idx in top_indices:
            raw_score = float(scores[idx])
            if raw_score <= 0:
                break  # BM25-scores zijn gesorteerd, stoppen bij nul
            # Normaliseer naar 0-1 relatief aan de beste match
            normalized = raw_score / max_score
            results.append({
                "text": self._texts[idx],
                "metadata": self._metas[idx],
                "similarity": round(normalized, 4),
                "distance": round(1 - normalized, 4),
                "_source": "bm25",
            })

        return results

    def __repr__(self) -> str:
        status = f"{self._num_docs} docs" if self._bm25 else "niet gebouwd"
        return f"BM25Index({status})"


# ─────────────────────────────────────────────
# RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    top_k: int = 10,
    k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[dict]:
    """
    Combineer dense en BM25 resultaten via Reciprocal Rank Fusion.

    RRF-formule (Cormack et al. 2009):
        score(doc) = Σ  weight / (k + rank(doc, methode))

    k=60 is de standaardwaarde uit de literatuur — hogere k maakt de
    fusie minder gevoelig voor exacte rangorde, lagere k geeft de
    top-rankers meer gewicht.

    Parameters
    ----------
    dense_results : list[dict]
        Resultaten van de dense vector search, gesorteerd op similarity.
    bm25_results : list[dict]
        Resultaten van BM25 search, gesorteerd op BM25-score.
    top_k : int
        Aantal resultaten om terug te geven na fusie.
    k : int
        RRF smoothing constant (default 60).
    dense_weight : float
        Gewicht van de dense rangorde in de RRF-score.
    bm25_weight : float
        Gewicht van de BM25-rangorde in de RRF-score.

    Returns
    -------
    list[dict]
        Gefuseerde en gehersorteerde chunks, met een "_rrf_score" veld
        voor diagnostiek.
    """
    scores: dict[str, dict] = {}

    def uid(chunk: dict) -> str:
        meta = chunk.get("metadata", {})
        return f"{meta.get('filename', '')}::{meta.get('chunk_index', '')}"

    # Dense bijdrage
    for rank, chunk in enumerate(dense_results, start=1):
        key = uid(chunk)
        if key not in scores:
            scores[key] = {"chunk": chunk, "rrf_score": 0.0}
        scores[key]["rrf_score"] += dense_weight / (k + rank)

    # BM25 bijdrage
    for rank, chunk in enumerate(bm25_results, start=1):
        key = uid(chunk)
        if key not in scores:
            scores[key] = {"chunk": chunk, "rrf_score": 0.0}
        scores[key]["rrf_score"] += bm25_weight / (k + rank)

    # Sorteer op RRF-score
    ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    result = []
    for item in ranked[:top_k]:
        chunk = item["chunk"].copy()
        chunk["_rrf_score"] = round(item["rrf_score"], 6)
        # Verwijder interne BM25-marker zodat de rest van de pipeline
        # geen verschil ziet met normale retrieve()-output
        chunk.pop("_source", None)
        result.append(chunk)

    return result


# ─────────────────────────────────────────────
# CLI — bouw of test de index stand-alone
# ─────────────────────────────────────────────

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="HAGA BM25 Index — bouwen, testen, vergelijken"
    )
    parser.add_argument("--build", action="store_true",
                        help="Bouw de BM25 index opnieuw (overschrijft bestaand bestand)")
    parser.add_argument("--query", "-q",
                        help="Test een zoekopdracht op de index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-path", default=None,
                        help="Pad naar het index-bestand (default: uit .env of ./bm25_index.pkl)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    index_path = args.index_path or os.getenv("BM25_INDEX_PATH", "./bm25_index.pkl")

    if args.build:
        import chromadb
        print("📦 ChromaDB laden...")
        client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        )
        col = client.get_collection(os.getenv("COLLECTION_NAME", "haga_folders"))
        print(f"   ✓ {col.count()} chunks")

        idx = BM25Index()
        idx.build(col)
        idx.save(index_path)
        print(f"\n✅ Index klaar: {index_path}")
        return

    if args.query:
        if not Path(index_path).exists():
            print(f"❌ Index niet gevonden: {index_path}")
            print("   Bouw eerst de index: python bm25_index.py --build")
            sys.exit(1)

        idx = BM25Index.load(index_path)
        print(f"\n🔍 Query: '{args.query}'")
        print("─" * 60)

        results = idx.search(args.query, top_k=args.top_k)
        if not results:
            print("  Geen resultaten.")
        for i, r in enumerate(results, 1):
            fname = r["metadata"].get("filename", "?")
            title = r["metadata"].get("title", "")[:50]
            sim = r["similarity"]
            preview = r["text"][:120].replace("\n", " ")
            print(f"\n  #{i} [{sim:.3f}] {fname}")
            print(f"       {title}")
            print(f"       {preview}...")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
