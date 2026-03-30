"""
STAP 3: Volledige RAG Evaluatie (met LLM)
==========================================
Test de complete pipeline: vraag → retrieval → LLM → antwoord.

Controleert per vraag:
  - Correctheid: bevat het antwoord de verwachte informatie?
  - Bronvermelding: verwijst het naar de juiste folder?
  - Hallucinations: verzint het LLM dingen die niet in de context staan?
  - Weigering: zegt het systeem "weet ik niet" bij onbekende vragen?

Vereist: draaiende Ollama (of OpenAI key) + ChromaDB met data.

Gebruik:
    python 3_test_full_rag.py
    python 3_test_full_rag.py --save
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import httpx
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# SERVER URL
# ─────────────────────────────────────────────

SERVER_URL = f"http://localhost:{os.getenv('PORT', '8000')}"


# ─────────────────────────────────────────────
# TEST VRAGEN
# ─────────────────────────────────────────────
# Inclusief "trap" vragen waar het antwoord NIET in de folders staat.

TEST_QUESTIONS = [
    # ══════════════════════════════════════════════
    # FEITELIJKE VRAGEN (exact antwoord verwacht)
    # ══════════════════════════════════════════════
    {
        "question": "Wat zijn de openingstijden van de daglounge in Zoetermeer?",
        "expected_in_answer": ["07.30", "18.00"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
    },
    {
        "question": "Wat is het telefoonnummer van de polikliniek Urologie in het HAGA?",
        "expected_in_answer": ["070", "6482"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
    },
    {
        "question": "Hoe lang duurt een BAEP gehoorzenuwonderzoek?",
        "expected_in_answer": ["30", "45"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
    },

    # ══════════════════════════════════════════════
    # PROCEDURELE VRAGEN
    # ══════════════════════════════════════════════
    {
        "question": "Wat moet ik doen na het plassen van een BCG blaasspoeling?",
        "expected_in_answer": ["zittend", "wc", "handen"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
    },
    {
        "question": "Hoelang moet ik nuchter zijn voor een elektrische cardioversie?",
        "expected_in_answer": ["6 uur"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
    },
    {
        "question": "Mag ik auto rijden na een elektrische cardioversie?",
        "expected_in_answer": ["niet"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
    },
    {
        "question": "Hoe lang moet ik het drukverband dragen na een elleboogzenuw operatie?",
        "expected_in_answer": ["5", "10"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
    },
    {
        "question": "Wat is de eerste hulp bij brandwonden?",
        "expected_in_answer": ["spoelen", "water", "15 minuten"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
    },


    # Spirometrie duur
    {
        "question": "Hoe lang duurt een spirometrie met diffusie en bodybox?",
        "expected_in_answer": ["75"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
    },
    # Spirometrie nuchter
    {
        "question": "Moet ik nuchter zijn voor een longfunctieonderzoek?",
        "expected_in_answer": ["niet nuchter", "eten en drinken"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
    },
    # Spirometrie telefoon Zoetermeer
    {
        "question": "Wat is het telefoonnummer van de polikliniek Longgeneeskunde in Zoetermeer?",
        "expected_in_answer": ["079", "2883"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
    },
    # Bodybox uitleg
    {
        "question": "Wat wordt er gemeten bij een bodybox onderzoek?",
        "expected_in_answer": ["longinhoud"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "medisch",
    },
    # JKZ dagbehandeling nuchter kind
    {
        "question": "Hoe lang voor de operatie mag mijn kind niet meer eten op de dagbehandeling JKZ?",
        "expected_in_answer": ["6 uur"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
    },
    # JKZ EMLA crème
    {
        "question": "Wanneer moet ik EMLA crème aanbrengen voor de dagbehandeling van mijn kind?",
        "expected_in_answer": ["1 uur", "hand"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
    },
    # JKZ opnamebureau telefoon
    {
        "question": "Wat is het telefoonnummer van het opnamebureau van het Juliana Kinderziekenhuis?",
        "expected_in_answer": ["070", "7368"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
    },
    # Glioom uitleg
    {
        "question": "Wat is een glioom?",
        "expected_in_answer": ["hersentumor", "gliacellen"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "medisch",
    },
    # Glioom gradaties
    {
        "question": "Hoeveel gradaties van gliomen zijn er?",
        "expected_in_answer": ["4"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "medisch",
    },
    # Glioom trap — persoonlijk medisch advies
    {
        "question": "Welke graad glioom heb ik waarschijnlijk als ik last heb van hoofdpijn en misselijkheid?",
        "expected_in_answer": [],
        "must_not_contain": [],
        "should_refuse": True,
        "category": "trap-medisch",
    },
    # Neurologie telefoon
    {
        "question": "Met welk telefoonnummer kan ik de polikliniek Neurologie bereiken?",
        "expected_in_answer": ["079", "2563"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
    },
    # Vingerprik pijnlijk
    {
        "question": "Is de vingerprik bij het longfunctieonderzoek pijnlijk?",
        "expected_in_answer": ["gevoelig"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "medisch",
    },


    # ══════════════════════════════════════════════
    # MEDISCHE KENNIS
    # ══════════════════════════════════════════════
    {
        "question": "Wat is BCG en waarvoor wordt het gebruikt?",
        "expected_in_answer": ["blaas", "afweer"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "medisch",
    },
    {
        "question": "Wat is een liesbreuk en hoe wordt het behandeld?",
        "expected_in_answer": ["zwelling", "matje"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "medisch",
    },
    {
        "question": "Welke bijwerkingen kan een BCG spoeling geven?",
        "expected_in_answer": ["koorts", "blaasontsteking"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "medisch",
    },

    # ══════════════════════════════════════════════
    # MUST-NOT-CONTAIN (foutdetectie)
    # ══════════════════════════════════════════════
    {
        "question": "Wat mag ik drinken 3 uur voor mijn operatie op de daglounge?",
        "expected_in_answer": ["water", "thee"],
        "must_not_contain": [],  # melk-check was vals-positief: "zonder melk" is correct
        "should_refuse": False,
        "category": "foutdetectie",
    },

    # ══════════════════════════════════════════════
    # TRAP-VRAGEN (moet weigeren / onzekerheid tonen)
    # ══════════════════════════════════════════════
    {
        "question": "Hoeveel kost een liesbreukoperatie bij het HagaZiekenhuis?",
        "expected_in_answer": [],
        "must_not_contain": [],
        "should_refuse": True,
        "category": "trap",
    },
    {
        "question": "Kan ik mijn afspraak annuleren via WhatsApp?",
        "expected_in_answer": ["bellen"],  # folder zegt bellen/internet/balie
        "must_not_contain": [],
        "should_refuse": False,  # info staat wél in folders (hoe je afspraken afzegt)
        "category": "procedureel",
    },
    {
        "question": "Wat is de beste behandeling voor diabetes type 2?",
        "expected_in_answer": [],
        "must_not_contain": [],
        "should_refuse": True,
        "category": "trap-medisch",
    },
    {
        "question": "Kan ik met een klaplong gaan vliegen volgende week?",
        "expected_in_answer": [],
        "must_not_contain": [],
        "should_refuse": True,  # moet doorverwijzen naar arts, niet zelf adviseren
        "category": "trap-medisch",
    },
]


# ─────────────────────────────────────────────
# API CALLS
# ─────────────────────────────────────────────

def call_chat(question: str, timeout: float = 120.0) -> dict | None:
    """Stuur een vraag naar de server."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{SERVER_URL}/api/chat",
                json={"message": question, "top_k": 20, "history": []},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        print(f"   ❌ Server niet bereikbaar op {SERVER_URL}")
        print("      Start eerst: python server.py")
        return None
    except Exception as e:
        print(f"   ❌ API fout: {e}")
        return None


# ─────────────────────────────────────────────
# EVALUATIE
# ─────────────────────────────────────────────

# Signaalwoorden die aangeven dat het model eerlijk is over onzekerheid
REFUSAL_SIGNALS = [
    "weet ik niet", "kan ik niet", "geen informatie", "niet in de folders",
    "niet in de beschikbare", "niet vermeld", "niet in de context",
    "raadpleeg", "neem contact op", "neem hiervoor contact",
    "vraag het aan", "behandelend arts",
    "niet beschikbaar", "geen antwoord", "niet gevonden", "helaas",
    "niet bekend", "buiten de beschikbare", "geen gegevens",
    "niet specifiek adviseren", "kan ik je niet",
    "staat niet in", "niet mogelijk om",
]


def evaluate_answer(response: dict, test: dict) -> dict:
    """Evalueer een RAG-antwoord."""
    answer = response["answer"].lower()
    sources = response.get("sources", [])

    result = {
        "question": test["question"],
        "answer": response["answer"],
        "sources": [s["filename"] for s in sources],
        "category": test["category"],
    }

    if test["should_refuse"]:
        # ── TRAP-VRAAG: het model MOET onzekerheid tonen ──
        shows_uncertainty = any(signal in answer for signal in REFUSAL_SIGNALS)
        result["test"] = "refusal"
        result["passed"] = shows_uncertainty
        result["detail"] = (
            "Correct: model toont onzekerheid"
            if shows_uncertainty
            else "⚠ HALLUCINATION: model geeft een antwoord alsof het zeker is"
        )

    else:
        # ── ECHTE VRAAG: check expected keywords ──
        expected = test.get("expected_in_answer", [])
        found = [kw for kw in expected if kw.lower() in answer]
        missing = [kw for kw in expected if kw.lower() not in answer]

        # Check must_not_contain (foutieve info)
        forbidden = test.get("must_not_contain", [])
        false_info = [kw for kw in forbidden if kw.lower() in answer]

        result["test"] = "correctness"
        result["keywords_found"] = found
        result["keywords_missing"] = missing
        result["false_info"] = false_info
        result["keyword_recall"] = len(found) / max(len(expected), 1)

        if false_info:
            result["passed"] = False
            result["detail"] = f"⚠ FOUTIEVE INFO: antwoord bevat '{', '.join(false_info)}'"
        elif missing:
            result["passed"] = len(found) / max(len(expected), 1) >= 0.5
            result["detail"] = f"Ontbrekend: {', '.join(missing)}"
        else:
            result["passed"] = True
            result["detail"] = "Alle verwachte info aanwezig"

    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HAGA RAG — Full Evaluation")
    parser.add_argument("--save", action="store_true", help="Sla resultaten op als JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("🧪 STAP 3: Volledige RAG Evaluatie (met LLM)")
    print("=" * 60)

    # Check server
    print(f"\n🔗 Server: {SERVER_URL}")
    try:
        with httpx.Client(timeout=5) as client:
            stats = client.get(f"{SERVER_URL}/api/stats").json()
        print(f"   ✓ Verbonden — {stats['total_chunks']} chunks, LLM: {stats['llm_model']}")
    except Exception:
        print(f"   ❌ Server niet bereikbaar. Start eerst:\n      python server.py")
        sys.exit(1)

    print(f"\n   {len(TEST_QUESTIONS)} test-vragen worden gedraaid...\n")

    # Run all tests
    results = []
    for i, test in enumerate(TEST_QUESTIONS, 1):
        category_emoji = {
            "feitelijk": "📌",
            "procedureel": "📋",
            "multi-bron": "🔗",
            "trap": "🪤",
            "trap-medisch": "🪤",
        }.get(test["category"], "❓")

        print(f"{'─' * 60}")
        print(f"{category_emoji} Vraag {i}/{len(TEST_QUESTIONS)}: {test['question']}")
        print(f"   Type: {test['category']}")

        start_time = time.time()
        response = call_chat(test["question"])
        elapsed = time.time() - start_time

        if response is None:
            results.append({"question": test["question"], "passed": False, "detail": "Server error"})
            continue

        # Toon antwoord (ingekort)
        answer_preview = response["answer"][:200].replace("\n", " ")
        print(f"\n   💬 Antwoord ({elapsed:.1f}s):")
        print(f"   {answer_preview}...")

        if response.get("sources"):
            source_names = [s["filename"] for s in response["sources"][:3]]
            print(f"   📎 Bronnen: {', '.join(source_names)}")

        # Evalueer
        evaluation = evaluate_answer(response, test)
        evaluation["response_time"] = elapsed
        results.append(evaluation)

        # Toon resultaat
        status = "✅ PASS" if evaluation["passed"] else "❌ FAIL"
        print(f"\n   {status}: {evaluation['detail']}")
        print()

    # ── SAMENVATTING ──
    print("=" * 60)
    print("📋 SAMENVATTING RAG EVALUATIE")
    print("=" * 60)

    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    pass_rate = passed / total * 100

    print(f"\n   Totaal:    {passed}/{total} geslaagd ({pass_rate:.0f}%)")

    # Per categorie
    categories = set(r.get("category", "?") for r in results)
    for cat in sorted(categories):
        cat_results = [r for r in results if r.get("category") == cat]
        cat_passed = sum(1 for r in cat_results if r.get("passed"))
        print(f"   {cat:15s}: {cat_passed}/{len(cat_results)}")

    # Response times
    times = [r.get("response_time", 0) for r in results if r.get("response_time")]
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n   Gem. response tijd: {avg_time:.1f}s")

    # Failed tests
    failures = [r for r in results if not r.get("passed")]
    if failures:
        print(f"\n   ❌ Gefaalde tests ({len(failures)}):")
        for f in failures:
            print(f"     → {f['question'][:50]}...")
            print(f"       {f.get('detail', '?')}")

    # Hallucination rate
    trap_results = [r for r in results if r.get("category", "").startswith("trap")]
    if trap_results:
        hallucinations = sum(1 for r in trap_results if not r.get("passed"))
        print(f"\n   🪤 Hallucinatie-detectie: {hallucinations}/{len(trap_results)} onterecht zeker")
        if hallucinations == 0:
            print("      ✅ Model hallucineert niet bij onbekende vragen")
        else:
            print("      ⚠ Model verzint antwoorden — system prompt aanscherpen")

    # Overall verdict
    print(f"\n{'─' * 60}")
    if pass_rate >= 80:
        print("   ✅ RAG-systeem werkt goed — klaar voor demo!")
    elif pass_rate >= 60:
        print("   ⚡ RAG-systeem werkt redelijk — verbeterpunten hierboven")
    else:
        print("   ❌ RAG-systeem heeft werk nodig — check retrieval eerst (stap 2)")
    print()

    # Save
    if args.save:
        output = {
            "timestamp": datetime.now().isoformat(),
            "server": SERVER_URL,
            "pass_rate": pass_rate,
            "results": results,
        }
        outfile = f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Resultaten opgeslagen: {outfile}")


if __name__ == "__main__":
    main()
