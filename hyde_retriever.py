"""
HAGA Folder RAG — HyDE Retriever
=================================
Hypothetical Document Embeddings (HyDE) voor verbeterde retrieval.

Kernidee:
  Naïef:   embed(vraag)         → zoek in vector store
  HyDE:    embed(llm(vraag))    → zoek in vector store

Het LLM genereert eerst een hypothetisch antwoord in foldertal.
Dat antwoord leeft semantisch veel dichter bij de echte chunks dan
de oorspronkelijke patiëntenvraag.

Referentie: Gao et al. (2022) — "Precise Zero-Shot Dense Retrieval
without Relevance Labels" https://arxiv.org/abs/2212.10496

Drop-in gebruik in je server:

    # Vervang dit:
    from server import retrieve
    results = retrieve(collection, embedder, query, top_k=10)

    # Door dit:
    from hyde_retriever import HyDERetriever
    hyde = HyDERetriever(embedder)
    results = hyde.retrieve(collection, query, top_k=10)

Stand-alone test:
    python hyde_retriever.py --query "Mag ik rijden na mijn oogoperatie?"
    python hyde_retriever.py --compare   # HyDE vs naïef voor alle golden questions
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ─────────────────────────────────────────────
# HYPOTHETISCH DOCUMENT GENERATOR
# ─────────────────────────────────────────────

# Systeem-prompt die het LLM instrueert een folderachtig antwoord te schrijven.
# Kritisch: het model moet dezelfde register en woordenschat gebruiken als de
# HAGA-folders, anders is het embedding voordeel beperkt.
HYDE_SYSTEM_PROMPT = """\
Je bent een medisch tekstschrijver voor het HagaZiekenhuis.
Schrijf een korte passage (3-5 zinnen) zoals die in een officiële patiëntenfolder
zou staan. Gebruik zakelijk maar toegankelijk Nederlands (B1-niveau).
Gebruik medische vaktermen die in folders voorkomen.
Schrijf ALLEEN de passage — geen titel, geen bullet points, geen inleiding.
Als de vraag gaat over een procedure: beschrijf voorbereiding, verloop of nazorg.
Als de vraag gaat over een aandoening: beschrijf symptomen of behandeling.
"""

HYDE_USER_TEMPLATE = "Schrijf een folderpassage die antwoord geeft op: {query}"


@dataclass
class RetrievalResult:
    """Resultaat van één chunk-retrieval."""
    text: str
    metadata: dict
    similarity: float
    rank: int


@dataclass
class HyDEResult:
    """Volledig HyDE-retrieval resultaat inclusief diagnostische info."""
    query: str
    hypothetical_doc: str
    chunks: list[RetrievalResult]
    hyde_latency_ms: int          # tijd voor LLM-generatie
    embed_latency_ms: int         # tijd voor embedding
    retrieval_latency_ms: int     # tijd voor ChromaDB query
    total_latency_ms: int
    used_fallback: bool = False   # True als HyDE faalde en naïef werd gebruikt


# ─────────────────────────────────────────────
# HYDE RETRIEVER
# ─────────────────────────────────────────────

class HyDERetriever:
    """
    HyDE-gebaseerde retriever voor de HAGA-foldercollectie.

    Parameters
    ----------
    embedder : LocalEmbedder | OpenAIEmbedder
        Het embedding-model — zelfde instantie als gebruikt in ingest.py.
    llm_client : OpenAI | None
        OpenAI-client voor hypothetische documentgeneratie.
        Als None, wordt een nieuwe client aangemaakt op basis van .env.
    model : str
        OpenAI-model voor generatie. gpt-4o-mini is goedkoop en snel genoeg.
    max_tokens : int
        Maximale lengte van het hypothetische document.
        150-200 tokens is de sweet spot: genoeg semantische dekking,
        niet zo lang dat het ruis introduceert.
    temperature : float
        Lagere temperatuur = consistentere output.
        0.3-0.5 werkt goed voor medische tekst.
    fallback_on_error : bool
        Als True: val terug op naïeve embedding als HyDE mislukt.
        Aanbevolen voor productie.
    """

    def __init__(
        self,
        embedder,
        llm_client: Optional[OpenAI] = None,
        model: str = None,
        max_tokens: int = 200,
        temperature: float = 0.3,
        fallback_on_error: bool = True,
    ):
        self.embedder = embedder
        self.client = llm_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fallback_on_error = fallback_on_error

        # Detecteer of het embedder-model E5 is (heeft prefix nodig)
        self._is_e5 = False
        model_name = getattr(embedder, "model", None)
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL", "")
        if hasattr(model_name, "lower"):
            self._is_e5 = "e5" in model_name.lower()

    # ──────────────────────────────────────────
    # Publieke API
    # ──────────────────────────────────────────

    def retrieve(
        self,
        collection: chromadb.Collection,
        query: str,
        top_k: int = 10,
        relevance_threshold: float = None,
    ) -> list[dict]:
        """
        Drop-in vervanging voor de naïeve retrieve()-functie.

        Geeft dezelfde dict-structuur terug als de originele retrieve():
            [{"text": ..., "metadata": ..., "similarity": ..., "distance": ...}]

        Parameters
        ----------
        collection : chromadb.Collection
            Geladen ChromaDB collectie.
        query : str
            Originele gebruikersvraag.
        top_k : int
            Aantal chunks om terug te geven.
        relevance_threshold : float | None
            Gooit chunks weg onder deze similarity-score.
            Als None: gebruik RELEVANCE_THRESHOLD uit .env (default 0.30).
        """
        result = self.retrieve_full(collection, query, top_k, relevance_threshold)
        return [
            {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "similarity": chunk.similarity,
                "distance": round(1 - chunk.similarity, 3),
            }
            for chunk in result.chunks
        ]

    def retrieve_full(
        self,
        collection: chromadb.Collection,
        query: str,
        top_k: int = 10,
        relevance_threshold: float = None,
    ) -> HyDEResult:
        """
        Zelfde als retrieve() maar geeft ook diagnostische info terug
        (latency, hypothetisch document, fallback-flag).
        Nuttig voor logging en A/B testen.
        """
        threshold = relevance_threshold or float(
            os.getenv("RELEVANCE_THRESHOLD", "0.30")
        )
        t_start = time.perf_counter()

        # Stap 1: Genereer hypothetisch document
        t0 = time.perf_counter()
        hypothetical_doc, used_fallback = self._generate_hypothetical_doc(query)
        hyde_latency = int((time.perf_counter() - t0) * 1000)

        # Stap 2: Embed het hypothetische document
        t1 = time.perf_counter()
        query_vector = self._embed_for_retrieval(hypothetical_doc)
        embed_latency = int((time.perf_counter() - t1) * 1000)

        # Stap 3: Query ChromaDB
        t2 = time.perf_counter()
        raw = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        retrieval_latency = int((time.perf_counter() - t2) * 1000)

        total_latency = int((time.perf_counter() - t_start) * 1000)

        # Bouw resultatenlijst
        chunks = []
        for rank, (doc, meta, dist) in enumerate(zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ), start=1):
            similarity = round(1 - dist, 3)
            if similarity < threshold:
                break  # Gesorteerd op similarity, stoppen bij eerste drop
            chunks.append(RetrievalResult(
                text=doc,
                metadata=meta,
                similarity=similarity,
                rank=rank,
            ))

        return HyDEResult(
            query=query,
            hypothetical_doc=hypothetical_doc,
            chunks=chunks,
            hyde_latency_ms=hyde_latency,
            embed_latency_ms=embed_latency,
            retrieval_latency_ms=retrieval_latency,
            total_latency_ms=total_latency,
            used_fallback=used_fallback,
        )

    # ──────────────────────────────────────────
    # Interne hulpfuncties
    # ──────────────────────────────────────────

    def _generate_hypothetical_doc(self, query: str) -> tuple[str, bool]:
        """
        Laat het LLM een hypothetisch folderantwoord genereren.

        Geeft (tekst, used_fallback) terug.
        Bij fout en fallback_on_error=True: geeft de originele query terug.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                    {"role": "user", "content": HYDE_USER_TEMPLATE.format(query=query)},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            hyp_doc = response.choices[0].message.content.strip()
            if not hyp_doc:
                raise ValueError("Leeg antwoord van LLM")
            return hyp_doc, False

        except Exception as e:
            if self.fallback_on_error:
                print(f"  ⚠ HyDE generatie mislukt ({e}), fallback naar originele query")
                return query, True
            raise

    def _embed_for_retrieval(self, text: str) -> list[float]:
        """
        Embed tekst voor retrieval — voegt E5-prefix toe indien nodig.

        E5-modellen verwachten:
          - "query: ..." voor retrieval-queries (ook hypothetische documenten!)
          - "passage: ..." voor de chunks in de index

        Dit is de sleutel: ook het hypothetische document wordt als "query:"
        geprefixed, niet als "passage:", want het speelt de rol van de query.
        """
        if self._is_e5:
            text = f"query: {text}"
        return self.embedder.embed([text], batch_size=1)[0]


# ─────────────────────────────────────────────
# NAÏEVE RETRIEVE (voor vergelijking)
# ─────────────────────────────────────────────

def retrieve_naive(
    collection: chromadb.Collection,
    embedder,
    query: str,
    top_k: int = 10,
    relevance_threshold: float = None,
) -> list[dict]:
    """
    Originele naïeve retrieval — embed de ruwe query direct.
    Gebruikt voor A/B vergelijking met HyDE.
    """
    threshold = relevance_threshold or float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))

    is_e5 = "e5" in os.getenv("EMBEDDING_MODEL", "").lower()
    q_text = f"query: {query}" if is_e5 else query
    q_embedding = embedder.embed([q_text], batch_size=1)[0]

    raw = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        similarity = round(1 - dist, 3)
        if similarity < threshold:
            break
        results.append({
            "text": doc,
            "metadata": meta,
            "similarity": similarity,
            "distance": round(dist, 3),
        })
    return results


# ─────────────────────────────────────────────
# A/B VERGELIJKINGSTEST
# ─────────────────────────────────────────────

COMPARE_QUESTIONS = [
    {
        "question": "Mag ik autorijden na mijn oogoperatie?",
        "expected_keyword": "autorijden",
        "expected_source": "oog",
    },
    {
        "question": "Hoe lang moet ik nuchter zijn voor de operatie?",
        "expected_keyword": "nuchter",
        "expected_source": None,
    },
    {
        "question": "Wat moet ik meenemen naar het ziekenhuis?",
        "expected_keyword": "meenemen",
        "expected_source": None,
    },
    {
        "question": "Wanneer mag ik weer douchen na de ingreep?",
        "expected_keyword": "douchen",
        "expected_source": None,
    },
    {
        "question": "Hoe herken ik tekenen van een wondinfectie?",
        "expected_keyword": "infectie",
        "expected_source": None,
    },
    {
        "question": "Wat zijn de bijwerkingen van methotrexaat bij kinderen?",
        "expected_keyword": "methotrexaat",
        "expected_source": "methotrexaat",
    },
    {
        "question": "Hoe gebruik ik EMLA zalf voor mijn kind?",
        "expected_keyword": "emla",
        "expected_source": "emla",
    },
    {
        "question": "Mag ik alcohol drinken na de bevalling?",
        "expected_keyword": "alcohol",
        "expected_source": "bevalling",
    },
]


def run_comparison(collection, embedder, top_k: int = 5) -> None:
    """
    Vergelijkt HyDE vs naïeve retrieval op een vaste vragenset.
    Toont per vraag:
      - Het hypothetische document
      - Top-3 chunks voor beide methoden
      - Similarity scores
      - Welke methode de verwachte bron/keyword het hoger rankt
    """
    hyde = HyDERetriever(embedder)

    hyde_wins = 0
    naive_wins = 0
    ties = 0

    print("\n" + "=" * 70)
    print("⚔  HyDE vs Naïef — A/B Vergelijking")
    print("=" * 70)

    for i, tc in enumerate(COMPARE_QUESTIONS, 1):
        q = tc["question"]
        kw = tc["expected_keyword"]
        src = tc["expected_source"]

        print(f"\n{'─' * 70}")
        print(f"📝 Vraag {i}/{len(COMPARE_QUESTIONS)}: {q}")

        # ── HyDE ──
        t0 = time.perf_counter()
        hyde_result = hyde.retrieve_full(collection, q, top_k=top_k)
        hyde_time = int((time.perf_counter() - t0) * 1000)

        print(f"\n🧠 Hypothetisch document ({hyde_result.hyde_latency_ms}ms):")
        print(f"   {hyde_result.hypothetical_doc[:300]}")
        if hyde_result.used_fallback:
            print("   ⚠ Fallback: originele query gebruikt")

        # ── Naïef ──
        t1 = time.perf_counter()
        naive_chunks = retrieve_naive(collection, embedder, q, top_k=top_k)
        naive_time = int((time.perf_counter() - t1) * 1000)

        # ── Scoor beide methoden ──
        def score_results(chunks_list: list[dict]) -> tuple[float, int | None]:
            """Geeft (top_similarity, rank_van_verwachte_bron)."""
            top_sim = chunks_list[0]["similarity"] if chunks_list else 0.0
            rank = None
            for r, chunk in enumerate(chunks_list, 1):
                fname = chunk["metadata"].get("filename", "").lower()
                text_lower = chunk["text"].lower()
                if (src and src.lower() in fname) or kw.lower() in text_lower:
                    rank = r
                    break
            return top_sim, rank

        hyde_chunks_as_dicts = [
            {"text": c.text, "metadata": c.metadata, "similarity": c.similarity}
            for c in hyde_result.chunks
        ]
        hyde_sim, hyde_rank = score_results(hyde_chunks_as_dicts)
        naive_sim, naive_rank = score_results(naive_chunks)

        # ── Druk resultaten af ──
        print(f"\n   {'Methode':<12} {'Top sim':>8}  {'Kw. rank':>9}  {'Tijd':>6}")
        print(f"   {'─' * 42}")

        hyde_rank_str = f"#{hyde_rank}" if hyde_rank else "—"
        naive_rank_str = f"#{naive_rank}" if naive_rank else "—"
        print(f"   {'HyDE':<12} {hyde_sim:>8.3f}  {hyde_rank_str:>9}  {hyde_time:>4}ms")
        print(f"   {'Naïef':<12} {naive_sim:>8.3f}  {naive_rank_str:>9}  {naive_time:>4}ms")

        # Wie wint?
        def rank_val(r):
            return r if r is not None else 999

        if rank_val(hyde_rank) < rank_val(naive_rank):
            print("   ✅ HyDE rankt het beter")
            hyde_wins += 1
        elif rank_val(naive_rank) < rank_val(hyde_rank):
            print("   📉 Naïef rankt het beter")
            naive_wins += 1
        else:
            print("   🟰 Gelijkspel (of beide missen)")
            ties += 1

        # Top-3 HyDE chunks
        print(f"\n   HyDE top-3:")
        for chunk in hyde_result.chunks[:3]:
            fname = chunk.metadata.get("filename", "?")[:45]
            preview = chunk.text[:80].replace("\n", " ")
            print(f"     [{chunk.rank}] {chunk.similarity:.3f}  {fname}")
            print(f"          {preview}...")

    # ── Eindstand ──
    total = len(COMPARE_QUESTIONS)
    print(f"\n{'=' * 70}")
    print(f"📊 Eindstand over {total} vragen:")
    print(f"   HyDE wint:  {hyde_wins}/{total}  ({hyde_wins/total*100:.0f}%)")
    print(f"   Naïef wint: {naive_wins}/{total}  ({naive_wins/total*100:.0f}%)")
    print(f"   Gelijkspel: {ties}/{total}")
    if hyde_wins > naive_wins:
        print("\n   ✅ HyDE is de betere retrieval-strategie voor dit corpus.")
    elif naive_wins > hyde_wins:
        print("\n   ⚠ Naïef scoort beter — heroverweeg HyDE prompt of model.")
    else:
        print("\n   🟰 Beide methoden presteren vergelijkbaar.")
    print()


# ─────────────────────────────────────────────
# LAADHELPERS (hergebruik van ingest.py)
# ─────────────────────────────────────────────

def _load_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    return client.get_collection(
        name=os.getenv("COLLECTION_NAME", "haga_folders")
    )


def _load_embedder():
    """Laadt het embedding model consistent met ingest.py."""
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    if provider == "openai":
        from ingest import OpenAIEmbedder
        return OpenAIEmbedder()
    from ingest import LocalEmbedder
    return LocalEmbedder()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="HAGA RAG — HyDE Retriever (test & vergelijking)"
    )
    parser.add_argument(
        "--query", "-q",
        help="Enkele vraag om HyDE op te testen",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Voer A/B vergelijking uit: HyDE vs naïef over standaard vragenset",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Aantal chunks om op te halen (default: 5)",
    )
    parser.add_argument(
        "--show-doc",
        action="store_true",
        help="Toon het hypothetische document bij --query",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model voor HyDE generatie (default: uit .env)",
    )
    args = parser.parse_args()

    if not args.query and not args.compare:
        parser.print_help()
        sys.exit(0)

    # Laad database en embedder
    print("📦 Database laden...")
    try:
        collection = _load_collection()
        print(f"   ✓ {collection.count()} chunks geladen")
    except Exception as e:
        print(f"   ❌ Database niet gevonden: {e}")
        sys.exit(1)

    print("🧠 Embedding model laden...")
    embedder = _load_embedder()
    print("   ✓ Klaar")

    if args.compare:
        run_comparison(collection, embedder, top_k=args.top_k)
        return

    # ── Enkele vraag ──
    hyde = HyDERetriever(embedder, model=args.model)

    print(f"\n🔍 Vraag: {args.query}")
    print("─" * 60)

    result = hyde.retrieve_full(collection, args.query, top_k=args.top_k)

    if args.show_doc or True:  # Altijd tonen bij enkel-vraag modus
        print(f"\n🧠 Hypothetisch document ({result.hyde_latency_ms}ms):")
        print(f"   {result.hypothetical_doc}")
        if result.used_fallback:
            print("   ⚠ Fallback: originele query gebruikt")

    print(f"\n⏱  Latency: LLM {result.hyde_latency_ms}ms | "
          f"embed {result.embed_latency_ms}ms | "
          f"retrieval {result.retrieval_latency_ms}ms | "
          f"totaal {result.total_latency_ms}ms")

    print(f"\n📄 Top {len(result.chunks)} resultaten:")
    for chunk in result.chunks:
        fname = chunk.metadata.get("filename", "?")
        title = chunk.metadata.get("title", "?")
        print(f"\n  [{chunk.rank}] similarity={chunk.similarity:.3f}")
        print(f"      {fname}")
        print(f"      {title[:60]}")
        preview = chunk.text[:200].replace("\n", " ")
        print(f"      {preview}...")


if __name__ == "__main__":
    main()
