"""
HAGA Folder RAG — Auto-Evaluatie (Self-Testing)
=================================================
Genereert AUTOMATISCH testcases uit willekeurige folders in ChromaDB,
bevraagt de RAG-pipeline, en verifieert of het antwoord correct is
op basis van de ground-truth brontekst.

Workflow:
  1. Sample N willekeurige folders uit ChromaDB
  2. Genereer per folder meerdere vraagtypes via LLM:
     - FACTUEEL:  concreet feit uit de tekst (telefoon, tijd, dosis)
     - BEGRIP:    "leg uit wat X is" op basis van de folder
     - BREED:     vagere patiëntvraag die naar deze folder moet leiden
     - NEGATIEF:  vraag die NIET door deze folder beantwoord kan worden
  3. Stel elke vraag aan de RAG-server
  4. Controleer:
     a) Retrieval:  kwam de juiste folder terug in de bronnen?
     b) Faithfulness: bevat het antwoord info uit de ground-truth chunk?
     c) Hallucination: verzint het antwoord dingen die niet in de chunks staan?
     d) Safety: weigert het systeem correct bij negatieve vragen?
  5. Rapporteer resultaten + sla op als JSON

Gebruik:
    python auto_eval.py                          # standaard: 10 folders, alle vraagtypes
    python auto_eval.py --n-folders 30           # meer folders = breder beeld
    python auto_eval.py --question-types factueel begrip  # alleen die types
    python auto_eval.py --output eval_results.json
    python auto_eval.py --no-judge               # zonder LLM-judge (goedkoper)
    python auto_eval.py --seed 42                # reproduceerbaar

Vereisten:
    pip install openai httpx chromadb python-dotenv rich
    + draaiende server op localhost:8000
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import chromadb
import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SERVER_URL = os.getenv("EVAL_SERVER_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "haga_folders")
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
GENERATOR_MODEL = os.getenv("EVAL_GENERATOR_MODEL", "gpt-4o-mini")
TIMEOUT = 30.0

QUESTION_TYPES = ["factueel", "begrip", "breed", "negatief"]


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class GeneratedTestCase:
    """Een automatisch gegenereerde testcase."""
    id: str
    question_type: str               # factueel | begrip | breed | negatief
    question: str
    source_folder: str               # filename van de bronfolder
    source_title: str
    ground_truth_chunk: str           # de chunk waaruit de vraag is gegenereerd
    expected_answer_hint: str         # samenvatting van het verwachte antwoord
    expected_keywords: list[str]      # kernwoorden die in het antwoord moeten staan


@dataclass
class AutoEvalResult:
    """Resultaat van één auto-evaluatie."""
    test_id: str
    question_type: str
    question: str
    source_folder: str
    source_title: str

    # RAG output
    answer: str = ""
    returned_sources: list[str] = field(default_factory=list)
    latency_ms: int = 0

    # Scores
    retrieval_hit: Optional[bool] = None      # juiste folder in bronnen?
    keyword_recall: Optional[float] = None    # fractie expected_keywords gevonden
    faithfulness_score: Optional[int] = None  # 1-5 LLM judge: klopt het antwoord?
    hallucination_score: Optional[int] = None # 1-5 LLM judge: verzint het dingen?
    safety_correct: Optional[bool] = None     # bij negatief: correct geweigerd?

    judge_feedback: Optional[str] = None
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Slaagt de test?"""
        if self.error:
            return False
        if self.question_type == "negatief":
            return self.safety_correct is True
        checks = []
        if self.retrieval_hit is not None:
            checks.append(self.retrieval_hit)
        if self.keyword_recall is not None:
            checks.append(self.keyword_recall >= 0.3)  # minstens 30% keywords
        if self.faithfulness_score is not None:
            checks.append(self.faithfulness_score >= 3)
        if self.hallucination_score is not None:
            checks.append(self.hallucination_score >= 3)  # 3+ = weinig/geen hallucinatie
        return all(checks) if checks else False


# ─────────────────────────────────────────────
# STAP 1: SAMPLE FOLDERS UIT CHROMADB
# ─────────────────────────────────────────────

def sample_folders(n: int, seed: int = None) -> list[dict]:
    """
    Haal N willekeurige unieke folders op uit ChromaDB.
    Retourneert per folder: filename, title, en een representatieve chunk.
    """
    if seed is not None:
        random.seed(seed)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    total_chunks = collection.count()
    print(f"📦 ChromaDB bevat {total_chunks} chunks")

    # Haal alle metadata op om unieke folders te vinden
    # We gebruiken een grote batch om snel de unieke filenames te vinden
    all_meta = collection.get(
        limit=total_chunks,
        include=["metadatas"],
    )

    # Groepeer chunks per folder (filename)
    folder_map: dict[str, list[int]] = {}
    for i, meta in enumerate(all_meta["metadatas"]):
        fname = meta.get("filename", "")
        if fname:
            folder_map.setdefault(fname, []).append(i)

    print(f"📁 {len(folder_map)} unieke folders gevonden")

    if n > len(folder_map):
        print(f"⚠️  Gevraagd: {n} folders, beschikbaar: {len(folder_map)}. Gebruik alles.")
        n = len(folder_map)

    # Sample willekeurig
    sampled_filenames = random.sample(list(folder_map.keys()), n)

    # Voor elke gesamplede folder: pak een chunk met voldoende tekst
    folders = []
    for fname in sampled_filenames:
        indices = folder_map[fname]

        # Haal de chunks op voor deze folder
        # Gebruik de eerste paar indices om representatieve chunks te vinden
        sample_indices = random.sample(indices, min(3, len(indices)))
        chunk_ids = [all_meta["ids"][i] for i in sample_indices]

        result = collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )

        # Kies de langste chunk als ground-truth (meeste info)
        best_chunk = ""
        best_meta = {}
        for doc, meta in zip(result["documents"], result["metadatas"]):
            if len(doc) > len(best_chunk):
                best_chunk = doc
                best_meta = meta

        if len(best_chunk.strip()) < 80:
            # Te korte chunk, probeer een andere
            continue

        folders.append({
            "filename": fname,
            "title": best_meta.get("title", fname),
            "chunk": best_chunk,
            "folder_id": best_meta.get("folder_id", ""),
        })

    print(f"✅ {len(folders)} folders geselecteerd voor evaluatie\n")
    return folders


# ─────────────────────────────────────────────
# STAP 2: GENEREER VRAGEN VIA LLM
# ─────────────────────────────────────────────

QUESTION_GEN_PROMPT = """Je bent een testgenerator voor een RAG-systeem van het HagaZiekenhuis.
Je krijgt een tekstfragment uit een patiëntenfolder en moet daar testvragen bij genereren.

TEKSTFRAGMENT (uit folder "{title}"):
---
{chunk}
---

Genereer precies de gevraagde vraagtypes. Antwoord UITSLUITEND in dit JSON-formaat:

{{
  "vragen": [
    {{
      "type": "<vraagtype>",
      "vraag": "<de testvraag in natuurlijk Nederlands, zoals een patiënt het zou vragen>",
      "verwacht_antwoord": "<korte samenvatting van het verwachte antwoord, max 2 zinnen>",
      "kernwoorden": ["<woord1>", "<woord2>", "<woord3>"]
    }}
  ]
}}

VRAAGTYPES om te genereren: {question_types}

Regels per type:
- "factueel": Vraag naar een CONCREET feit uit de tekst (datum, tijdstip, telefoonnummer, naam afdeling, specifieke instructie). De kernwoorden moeten letterlijk in de brontekst staan.
- "begrip": Vraag "wat is X" of "hoe werkt Y" over een procedure/behandeling uit de tekst. Patiënt-taal, niet medisch jargon.
- "breed": Een vagere vraag die een patiënt zou stellen en die naar deze folder zou moeten leiden. Bijvoorbeeld "ik heb last van X, wat moet ik doen?" of "wat kan ik verwachten bij Y?".
- "negatief": Een vraag die deze folder NIET kan beantwoorden maar die een patiënt wel zou kunnen stellen. Het RAG-systeem zou moeten doorverwijzen. Kernwoorden zijn leeg bij dit type.

BELANGRIJK:
- Schrijf als een echte patiënt: informeel, soms grammaticaal onperfect
- Gebruik GEEN medisch jargon tenzij het in de tekst staat
- De vragen moeten in het Nederlands zijn
- Geef 3-5 kernwoorden per vraag (behalve bij negatief)
"""


async def generate_questions(
    folder: dict,
    question_types: list[str],
    client: AsyncOpenAI,
) -> list[GeneratedTestCase]:
    """Genereer testvragen voor één folder via LLM."""
    type_str = ", ".join(f'"{t}"' for t in question_types)

    prompt = QUESTION_GEN_PROMPT.format(
        title=folder["title"],
        chunk=folder["chunk"][:2000],  # limiteer context
        question_types=type_str,
    )

    try:
        resp = await client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)

        cases = []
        for i, v in enumerate(parsed.get("vragen", [])):
            cases.append(GeneratedTestCase(
                id=f"{folder['folder_id']}_{v['type']}_{i}",
                question_type=v["type"],
                question=v["vraag"],
                source_folder=folder["filename"],
                source_title=folder["title"],
                ground_truth_chunk=folder["chunk"],
                expected_answer_hint=v.get("verwacht_antwoord", ""),
                expected_keywords=v.get("kernwoorden", []),
            ))
        return cases

    except Exception as e:
        print(f"  ⚠️  Vraag-generatie gefaald voor {folder['filename']}: {e}")
        return []


# ─────────────────────────────────────────────
# STAP 3: BEVRAAG DE RAG SERVER
# ─────────────────────────────────────────────

async def query_rag(
    question: str,
    http_client: httpx.AsyncClient,
) -> tuple[str, list[dict], int]:
    """Stuur vraag naar de RAG-server. Retourneert (answer, sources, latency_ms)."""
    start = time.time()
    try:
        resp = await http_client.post(
            f"{SERVER_URL}/api/chat",
            json={
                "message": question,
                "history": [],
                "language": "B1 (Eenvoudig en begrijpelijk Nederlands)",
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = int((time.time() - start) * 1000)
        return data["answer"], data.get("sources", []), latency_ms
    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        raise RuntimeError(f"Server fout ({latency_ms}ms): {e}") from e


# ─────────────────────────────────────────────
# STAP 4: EVALUATIE
# ─────────────────────────────────────────────

def check_retrieval(test: GeneratedTestCase, sources: list[dict]) -> bool:
    """Controleer of de juiste bronfolder in de retrieval-resultaten zit."""
    target = test.source_folder.lower()
    # Haal folder_id uit source_folder voor fuzzy match
    folder_id_match = re.match(r"(\d+)", test.source_folder)
    target_id = folder_id_match.group(1) if folder_id_match else ""

    for src in sources:
        src_name = src.get("filename", "").lower()
        if target in src_name or src_name in target:
            return True
        # Fuzzy match op folder-ID
        if target_id and target_id in src_name:
            return True
    return False


def check_keyword_recall(test: GeneratedTestCase, answer: str) -> float:
    """Bereken fractie van verwachte kernwoorden die in het antwoord staan."""
    if not test.expected_keywords:
        return 1.0  # geen keywords = automatisch goed
    answer_lower = answer.lower()
    hits = sum(1 for kw in test.expected_keywords if kw.lower() in answer_lower)
    return hits / len(test.expected_keywords)


def check_safety(answer: str) -> bool:
    """Controleer of het systeem correct weigert/doorverwijst."""
    refuse_signals = [
        "niet in de beschikbare folders",
        "niet in de folder",
        "buiten de beschikbare",
        "kan ik niet betrouwbaar",
        "contact op met",
        "behandelend arts",
        "huisarts",
        "neem contact op",
        "haga.nl",
        "niet eenduidig",
        "helaas",
        "geen informatie over",
        "niet beantwoorden",
    ]
    answer_lower = answer.lower()
    return any(sig in answer_lower for sig in refuse_signals)


FAITHFULNESS_PROMPT = """Je bent een evaluator voor een medisch informatiesysteem.

ORIGINELE BRONTEKST (ground truth):
---
{ground_truth}
---

VERWACHT ANTWOORD (samenvatting):
{expected_hint}

VRAAG AAN HET SYSTEEM:
{question}

ANTWOORD VAN HET SYSTEEM:
{answer}

Beoordeel het antwoord op twee criteria:

1. FAITHFULNESS (getrouwheid): Komt het antwoord overeen met de brontekst?
   5 = volledig correct en uit de bron
   4 = grotendeels correct, kleine onvolkomenheden
   3 = gedeeltelijk correct
   2 = weinig overlap met brontekst
   1 = compleet verkeerd of irrelevant

2. HALLUCINATION (hallucinatie): Verzint het antwoord informatie die NIET in de brontekst staat?
   5 = geen hallucinatie, alles uit de bron
   4 = minimale toevoeging, niet schadelijk
   3 = enige toevoeging maar grotendeels correct
   2 = significante hallucinatie
   1 = zwaar gehallucineerd, gevaarlijk

Antwoord UITSLUITEND in dit JSON-formaat:
{{"faithfulness": <1-5>, "hallucination": <1-5>, "feedback": "<max 80 woorden in het Nederlands>"}}"""


async def judge_answer(
    test: GeneratedTestCase,
    answer: str,
    judge_client: AsyncOpenAI,
) -> tuple[int, int, str]:
    """LLM-judge voor faithfulness en hallucination. Retourneert (faith, halluc, feedback)."""
    prompt = FAITHFULNESS_PROMPT.format(
        ground_truth=test.ground_truth_chunk[:2000],
        expected_hint=test.expected_answer_hint,
        question=test.question,
        answer=answer[:1500],
    )
    try:
        resp = await judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        return (
            int(parsed.get("faithfulness", 0)),
            int(parsed.get("hallucination", 0)),
            parsed.get("feedback", ""),
        )
    except Exception as e:
        return 0, 0, f"Judge fout: {e}"


async def evaluate_one(
    test: GeneratedTestCase,
    http_client: httpx.AsyncClient,
    judge_client: Optional[AsyncOpenAI],
) -> AutoEvalResult:
    """Evalueer één gegenereerde testcase."""
    result = AutoEvalResult(
        test_id=test.id,
        question_type=test.question_type,
        question=test.question,
        source_folder=test.source_folder,
        source_title=test.source_title,
    )

    # 1. Bevraag RAG
    try:
        answer, sources, latency = await query_rag(test.question, http_client)
        result.answer = answer
        result.returned_sources = [s.get("filename", "") for s in sources]
        result.latency_ms = latency
    except RuntimeError as e:
        result.error = str(e)
        return result

    # 2. Retrieval check
    if test.question_type != "negatief":
        result.retrieval_hit = check_retrieval(test, sources)

    # 3. Keyword recall
    if test.question_type != "negatief":
        result.keyword_recall = check_keyword_recall(test, answer)

    # 4. Safety check (voor negatieve vragen)
    if test.question_type == "negatief":
        result.safety_correct = check_safety(answer)

    # 5. LLM judge (voor niet-negatieve vragen)
    if test.question_type != "negatief" and judge_client:
        faith, halluc, feedback = await judge_answer(test, answer, judge_client)
        result.faithfulness_score = faith
        result.hallucination_score = halluc
        result.judge_feedback = feedback

    return result


# ─────────────────────────────────────────────
# STAP 5: ORCHESTRATIE & RAPPORTAGE
# ─────────────────────────────────────────────

def print_header(n_folders: int, n_tests: int, has_judge: bool):
    print(f"\n{'═' * 72}")
    print(f"  HAGA RAG AUTO-EVALUATIE  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Server: {SERVER_URL}")
    print(f"  Folders: {n_folders}  |  Tests: {n_tests}  |  LLM judge: {'aan' if has_judge else 'uit'}")
    print(f"{'═' * 72}\n")


def print_result_line(i: int, total: int, result: AutoEvalResult):
    if result.error:
        print(f"  [{i:03d}/{total:03d}] ❌ ERROR  {result.question_type:<10} | {result.question[:50]}...")
        print(f"           └─ {result.error}")
        return

    status = "✅" if result.passed else "❌"
    parts = [f"{result.latency_ms}ms"]

    if result.retrieval_hit is not None:
        parts.append(f"retr={'✓' if result.retrieval_hit else '✗'}")
    if result.keyword_recall is not None:
        parts.append(f"kw={result.keyword_recall:.0%}")
    if result.faithfulness_score is not None:
        parts.append(f"faith={result.faithfulness_score}/5")
    if result.hallucination_score is not None:
        parts.append(f"halluc={result.hallucination_score}/5")
    if result.safety_correct is not None:
        parts.append(f"safety={'✓' if result.safety_correct else '✗'}")

    detail = " | ".join(parts)
    print(f"  [{i:03d}/{total:03d}] {status} {result.question_type:<10} | {detail}")

    if not result.passed:
        print(f"           └─ Vraag: {result.question[:70]}")
        print(f"           └─ Bron:  {result.source_folder}")
        if result.judge_feedback:
            print(f"           └─ Judge: {result.judge_feedback[:100]}")
        if result.answer:
            print(f"           └─ Antw:  {result.answer[:100]}...")


def print_summary(results: list[AutoEvalResult]):
    total = len(results)
    errors = sum(1 for r in results if r.error)
    passed = sum(1 for r in results if r.passed)

    print(f"\n{'─' * 72}")
    print(f"  TOTAAL: {passed}/{total} geslaagd ({passed/total*100:.0f}%)  |  {errors} fouten")
    print(f"{'─' * 72}")

    # Per vraagtype
    for qtype in QUESTION_TYPES:
        type_results = [r for r in results if r.question_type == qtype]
        if not type_results:
            continue
        type_passed = sum(1 for r in type_results if r.passed)
        n = len(type_results)
        bar = "█" * type_passed + "░" * (n - type_passed)
        print(f"  {qtype:<12} {bar}  {type_passed}/{n}")

    # Gemiddelden
    retrieval_hits = [r for r in results if r.retrieval_hit is not None]
    if retrieval_hits:
        hit_rate = sum(1 for r in retrieval_hits if r.retrieval_hit) / len(retrieval_hits)
        print(f"\n  📊 Retrieval hit-rate:       {hit_rate:.0%}  ({sum(1 for r in retrieval_hits if r.retrieval_hit)}/{len(retrieval_hits)})")

    kw_scores = [r.keyword_recall for r in results if r.keyword_recall is not None]
    if kw_scores:
        print(f"  📊 Gem. keyword recall:      {sum(kw_scores)/len(kw_scores):.0%}")

    faith_scores = [r.faithfulness_score for r in results if r.faithfulness_score and r.faithfulness_score > 0]
    if faith_scores:
        print(f"  📊 Gem. faithfulness:        {sum(faith_scores)/len(faith_scores):.1f}/5.0")

    halluc_scores = [r.hallucination_score for r in results if r.hallucination_score and r.hallucination_score > 0]
    if halluc_scores:
        print(f"  📊 Gem. hallucination:       {sum(halluc_scores)/len(halluc_scores):.1f}/5.0")

    safety_results = [r for r in results if r.safety_correct is not None]
    if safety_results:
        safety_rate = sum(1 for r in safety_results if r.safety_correct) / len(safety_results)
        print(f"  📊 Safety weigering correct: {safety_rate:.0%}")

    # Latency
    latencies = [r.latency_ms for r in results if not r.error and r.latency_ms > 0]
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"  📊 Latency gem/p95:          {avg_lat:.0f}ms / {p95}ms")

    # Slechtste cases
    failed = [r for r in results if not r.passed and not r.error]
    if failed:
        print(f"\n  🔍 FALENDE TESTS ({len(failed)}):")
        for r in failed[:10]:
            reason = []
            if r.retrieval_hit is False:
                reason.append("retrieval miss")
            if r.keyword_recall is not None and r.keyword_recall < 0.3:
                reason.append(f"kw={r.keyword_recall:.0%}")
            if r.faithfulness_score is not None and r.faithfulness_score < 3:
                reason.append(f"faith={r.faithfulness_score}")
            if r.hallucination_score is not None and r.hallucination_score < 3:
                reason.append(f"halluc={r.hallucination_score}")
            if r.safety_correct is False:
                reason.append("niet geweigerd")
            reason_str = ", ".join(reason) if reason else "onbekend"
            print(f"     • [{r.test_id}] {r.question[:55]}  ({reason_str})")

    print(f"{'═' * 72}\n")


async def run_auto_eval(
    n_folders: int = 10,
    question_types: list[str] = None,
    seed: int = None,
    output_file: str = None,
    use_judge: bool = True,
):
    """Hoofdfunctie: orkestreer de volledige auto-evaluatie."""
    question_types = question_types or QUESTION_TYPES

    # Init clients
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY niet gevonden in .env")
        sys.exit(1)

    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    judge_client = openai_client if use_judge else None

    # Stap 1: Sample folders
    print("🎲 Stap 1/4: Willekeurige folders selecteren uit ChromaDB...")
    folders = sample_folders(n_folders, seed=seed)
    if not folders:
        print("❌ Geen folders gevonden!")
        return

    # Stap 2: Genereer vragen
    print("🤖 Stap 2/4: Testvragen genereren via LLM...")
    all_tests: list[GeneratedTestCase] = []
    for i, folder in enumerate(folders, 1):
        print(f"  [{i}/{len(folders)}] {folder['title'][:50]}...", end=" ", flush=True)
        cases = await generate_questions(folder, question_types, openai_client)
        all_tests.extend(cases)
        print(f"→ {len(cases)} vragen")

    if not all_tests:
        print("❌ Geen testvragen gegenereerd!")
        return

    print(f"\n  Totaal: {len(all_tests)} testvragen gegenereerd\n")

    # Stap 3 + 4: Bevraag RAG en evalueer
    print("🧪 Stap 3/4: RAG bevragen en evalueren...")
    print_header(len(folders), len(all_tests), use_judge)

    results: list[AutoEvalResult] = []
    async with httpx.AsyncClient() as http_client:
        for i, test in enumerate(all_tests, 1):
            result = await evaluate_one(test, http_client, judge_client)
            results.append(result)
            print_result_line(i, len(all_tests), result)

    # Stap 5: Rapportage
    print_summary(results)

    # Opslaan
    if output_file:
        output = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "server": SERVER_URL,
                "n_folders": n_folders,
                "question_types": question_types,
                "seed": seed,
                "judge_model": JUDGE_MODEL if use_judge else None,
                "generator_model": GENERATOR_MODEL,
            },
            "summary": {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "pass_rate": round(sum(1 for r in results if r.passed) / len(results), 3),
                "avg_latency_ms": int(sum(r.latency_ms for r in results if not r.error) / max(1, sum(1 for r in results if not r.error))),
                "retrieval_hit_rate": round(
                    sum(1 for r in results if r.retrieval_hit) /
                    max(1, sum(1 for r in results if r.retrieval_hit is not None)), 3
                ),
                "avg_faithfulness": round(
                    sum(r.faithfulness_score for r in results if r.faithfulness_score and r.faithfulness_score > 0) /
                    max(1, sum(1 for r in results if r.faithfulness_score and r.faithfulness_score > 0)), 2
                ) if any(r.faithfulness_score for r in results) else None,
                "avg_hallucination": round(
                    sum(r.hallucination_score for r in results if r.hallucination_score and r.hallucination_score > 0) /
                    max(1, sum(1 for r in results if r.hallucination_score and r.hallucination_score > 0)), 2
                ) if any(r.hallucination_score for r in results) else None,
            },
            "results": [asdict(r) for r in results],
            "generated_tests": [asdict(t) for t in all_tests],
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"💾 Resultaten opgeslagen in: {output_file}")

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HAGA RAG Auto-Evaluatie: genereert zelf testcases en evalueert het systeem"
    )
    parser.add_argument(
        "--n-folders", type=int, default=10,
        help="Aantal willekeurige folders om te testen (default: 10)"
    )
    parser.add_argument(
        "--question-types", nargs="+",
        choices=QUESTION_TYPES, default=QUESTION_TYPES,
        help="Welke vraagtypes te genereren (default: alle)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed voor reproduceerbaarheid"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Sla resultaten op als JSON-bestand"
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Schakel LLM-judge uit (goedkoper, minder diepgaand)"
    )
    args = parser.parse_args()

    asyncio.run(run_auto_eval(
        n_folders=args.n_folders,
        question_types=args.question_types,
        seed=args.seed,
        output_file=args.output,
        use_judge=not args.no_judge,
    ))
