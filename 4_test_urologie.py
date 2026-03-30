"""
Urologie Folders — Volledige RAG Evaluatie
==========================================
Testset gebaseerd op 10 urologie-folders (1248-1258):
  1248  Niersteenvergruizing (ESWL)
  1249  Prostaatoperatie TURP
  1250  Sterilisatie van de man (vasectomie)
  1251  Stressincontinentie (Altis sling)
  1252  Thuis met een katheter
  1253  Thuis met een katheterventiel
  1254  Thuis met een nefrostomiekatheter
  1255  Thuis met een suprapubische katheter
  1257  Blaasoperatie TUR-blaas
  1258  Kijkoperatie urineleider of nier

Alle expected_in_answer-waarden zijn letterlijk uit de PDF-tekst overgenomen.

Gebruik:
    python 4_test_urologie.py
    python 4_test_urologie.py --save
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

SERVER_URL = f"http://localhost:{os.getenv('PORT', '8000')}"

# ─────────────────────────────────────────────
# REFUSAL SIGNALS
# ─────────────────────────────────────────────
# Als het model een trap-vraag correct afwijst, moet het
# minstens één van deze signaalzinnen bevatten.

REFUSAL_SIGNALS = [
    "niet betrouwbaar beantwoorden",
    "niet in de beschikbare",
    "staat niet in",
    "raadpleeg uw arts",
    "behandelend arts",
    "neem contact op",
    "kan ik niet",
    "valt buiten",
    "geen informatie",
    "niet gevonden",
    "weet ik niet",
]


# ─────────────────────────────────────────────
# TEST VRAGEN
# ─────────────────────────────────────────────

TEST_QUESTIONS = [

    # ══════════════════════════════════════════════
    # FEITELIJKE VRAGEN — contactgegevens & tijden
    # ══════════════════════════════════════════════

    {
        "question": "Wat is het telefoonnummer van de polikliniek Urologie in Den Haag?",
        "expected_in_answer": ["070", "6482"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
        "source_hint": "1255/1257/1258/1249/1251",
    },
    {
        "question": "Wat is het telefoonnummer van de Spoedeisende Hulp in Den Haag voor urologie-klachten buiten kantooruren?",
        "expected_in_answer": ["070", "2060"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
        "source_hint": "1255/1257/1252",
    },
    {
        "question": "Op welke dagen en tijden is de polikliniek Urologie bereikbaar?",
        "expected_in_answer": ["maandag", "vrijdag", "8.00", "16.30"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
        "source_hint": "1258",
    },
    {
        "question": "Hoe lang duurt een niersteenvergruizing (ESWL-behandeling)?",
        "expected_in_answer": ["half uur"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
        "source_hint": "1248",
    },
    {
        "question": "Hoe lang duurt een TUR-blaas operatie?",
        "expected_in_answer": ["30", "60"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
        "source_hint": "1257",
    },
    {
        "question": "Hoe lang duurt de stressincontinentie operatie met de Altis sling?",
        "expected_in_answer": ["20"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
        "source_hint": "1251",
    },
    {
        "question": "Hoe lang duurt een vasectomie (sterilisatie van de man)?",
        "expected_in_answer": ["15"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
        "source_hint": "1250",
    },
    {
        "question": "Hoe vaak moet een gewone urinekatheter vervangen worden?",
        "expected_in_answer": ["6", "8", "weken"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
        "source_hint": "1252",
    },
    {
        "question": "Hoe vaak moet een nefrostomiekatheter vervangen worden?",
        "expected_in_answer": ["4", "6", "maanden"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
        "source_hint": "1254",
    },
    {
        "question": "Na hoeveel maanden na de vasectomie wordt er een spermaonderzoek gedaan?",
        "expected_in_answer": ["3"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "feitelijk",
        "source_hint": "1250",
    },

    # ══════════════════════════════════════════════
    # PROCEDURELE VRAGEN — voorbereiding & nazorg
    # ══════════════════════════════════════════════

    {
        "question": "Hoeveel moet ik per dag drinken als ik een katheter heb thuis?",
        "expected_in_answer": ["1,5", "liter"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
        "source_hint": "1252/1255/1254",
    },
    {
        "question": "Hoe vaak per dag moet ik de huid rondom mijn suprapubische katheter schoonmaken?",
        "expected_in_answer": ["twee keer", "2 keer"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
        "source_hint": "1255",
    },
    {
        "question": "Mag ik zeep gebruiken bij het schoonmaken van de katheterinsteekopening?",
        "expected_in_answer": ["geen zeep", "niet", "zeep"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
        "source_hint": "1255/1252/1254",
    },
    {
        "question": "Moet ik nuchter zijn voor een niersteenvergruizing?",
        "expected_in_answer": ["niet nuchter", "hoeft niet nuchter"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
        "source_hint": "1248",
    },
    {
        "question": "Welke pijnstiller moet ik innemen voor een ESWL-behandeling en hoeveel?",
        "expected_in_answer": ["diclofenac", "paracetamol"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
        "source_hint": "1248",
    },
    {
        "question": "Hoeveel dagen voor een vasectomie moet ik mijn balzak scheren?",
        "expected_in_answer": ["3"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
        "source_hint": "1250",
    },
    {
        "question": "Wat zijn de leefregels voor de eerste 2 weken na een TUR-blaas operatie?",
        "expected_in_answer": ["seksuele gemeenschap", "alcohol", "fietsen"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
        "source_hint": "1257",
    },
    {
        "question": "Hoe lang moet ik minimaal 5 keer per dag plassen na de Altis stressincontinentie operatie?",
        "expected_in_answer": ["2 weken"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
        "source_hint": "1251",
    },
    {
        "question": "Hoe vaak moet ik overdag mijn katheterventiel openzetten om de blaas leeg te maken?",
        "expected_in_answer": ["4 uur"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
        "source_hint": "1253",
    },
    {
        "question": "Wanneer moet ik na een niersteenvergruizing contact opnemen met de polikliniek Urologie?",
        "expected_in_answer": ["koorts", "38,5", "pijn"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
        "source_hint": "1248",
    },
    {
        "question": "Hoe lang mag de pleister bij een nefrostomiekatheter blijven zitten?",
        "expected_in_answer": ["week"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
        "source_hint": "1254",
    },
    {
        "question": "Wat moet ik doen als mijn katheter er per ongeluk uitvalt?",
        "expected_in_answer": ["2 uur", "contact"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "procedureel",
        "source_hint": "1255",
    },

    # ══════════════════════════════════════════════
    # MEDISCHE VRAGEN — wat is / hoe werkt
    # ══════════════════════════════════════════════

    {
        "question": "Wat is een suprapubische katheter?",
        "expected_in_answer": ["buikwand", "blaas", "slangetje"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "medisch",
        "source_hint": "1255",
    },
    {
        "question": "Wat is een dubbel-J katheter en waarvoor wordt hij gebruikt?",
        "expected_in_answer": ["nier", "blaas", "urineleider"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "medisch",
        "source_hint": "1258",
    },
    {
        "question": "Wat betekent TURP bij een prostaatoperatie?",
        "expected_in_answer": ["Trans Urethrale Resectie", "prostaat"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "medisch",
        "source_hint": "1249",
    },
    {
        "question": "Wat is stressincontinentie?",
        "expected_in_answer": ["urineverlies", "inspanning", "plasbuis"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "medisch",
        "source_hint": "1251",
    },
    {
        "question": "Wat is het effect van een prostaatoperatie op de zaadlozing?",
        "expected_in_answer": ["blaas", "droog", "zaadlozing"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "medisch",
        "source_hint": "1249",
    },
    {
        "question": "Hoe groot is de kans dat de stressincontinentie operatie succesvol is?",
        "expected_in_answer": ["90"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "medisch",
        "source_hint": "1251",
    },

    # ══════════════════════════════════════════════
    # NAZORG / KLACHTEN
    # ══════════════════════════════════════════════

    {
        "question": "Hoe lang kan de urine rood van kleur zijn na een TUR-blaas operatie?",
        "expected_in_answer": ["6 weken"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "nazorg",
        "source_hint": "1257",
    },
    {
        "question": "Kan ik autorijden na een niersteenvergruizing?",
        "expected_in_answer": ["niet", "autorijden"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "nazorg",
        "source_hint": "1248",
    },
    {
        "question": "Wat moet ik doen als er gedurende 2-3 uur geen urine in mijn opvangzak loopt?",
        "expected_in_answer": ["contact", "knikken", "opvangzak"],
        "must_not_contain": [],
        "should_refuse": False,
        "category": "nazorg",
        "source_hint": "1252/1255/1254",
    },

    # ══════════════════════════════════════════════
    # TRAP VRAGEN — antwoord staat NIET in de folders
    # ══════════════════════════════════════════════

    {
        "question": "Wat is de beste medicatie voor het behandelen van een prostaatontsteking?",
        "expected_in_answer": [],
        "must_not_contain": [],
        "should_refuse": True,
        "category": "trap",
        "source_hint": None,
    },
    {
        "question": "Kan een vasectomie worden teruggedraaid en wat zijn de slagingspercentages?",
        "expected_in_answer": [],
        "must_not_contain": [],
        "should_refuse": True,
        "category": "trap",
        "source_hint": None,
    },
    {
        "question": "Welke voedingsmiddelen moet ik vermijden als ik last heb van nierstenen?",
        "expected_in_answer": [],
        "must_not_contain": [],
        "should_refuse": True,
        "category": "trap",
        "source_hint": None,
    },
]


# ─────────────────────────────────────────────
# API AANROEP
# ─────────────────────────────────────────────

def call_chat(question: str) -> dict | None:
    """Stuur een vraag naar de RAG-server en retourneer het antwoord."""
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{SERVER_URL}/api/chat",
                json={"message": question, "history": []},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"   ⚠ Server-fout: {e}")
        return None


# ─────────────────────────────────────────────
# EVALUATIE
# ─────────────────────────────────────────────

def evaluate_answer(response: dict, test: dict) -> dict:
    """Evalueer één antwoord op basis van de testverwachtingen."""
    answer = response["answer"].lower()
    sources = response.get("sources", [])

    result = {
        "question": test["question"],
        "answer": response["answer"],
        "sources": [s["filename"] for s in sources],
        "category": test["category"],
        "source_hint": test.get("source_hint"),
    }

    if test["should_refuse"]:
        # Trap-vraag: model MOET weigeren of doorverwijzen
        shows_uncertainty = any(signal in answer for signal in REFUSAL_SIGNALS)
        result["test"] = "refusal"
        result["passed"] = shows_uncertainty
        result["detail"] = (
            "✓ Correct geweigerd"
            if shows_uncertainty
            else "⚠ HALLUCINATION: model geeft antwoord alsof het zeker is"
        )

    else:
        # Echte vraag: check of verwachte trefwoorden aanwezig zijn
        expected = test.get("expected_in_answer", [])
        found = [kw for kw in expected if kw.lower() in answer]
        missing = [kw for kw in expected if kw.lower() not in answer]

        forbidden = test.get("must_not_contain", [])
        false_info = [kw for kw in forbidden if kw.lower() in answer]

        result["test"] = "correctness"
        result["keywords_found"] = found
        result["keywords_missing"] = missing
        result["false_info"] = false_info
        result["keyword_recall"] = len(found) / max(len(expected), 1)

        if false_info:
            result["passed"] = False
            result["detail"] = f"⚠ FOUTIEVE INFO: '{', '.join(false_info)}'"
        elif missing:
            # Slaagt als minstens de helft van de trefwoorden aanwezig is
            result["passed"] = result["keyword_recall"] >= 0.5
            result["detail"] = f"Ontbrekend: {', '.join(missing)}"
        else:
            result["passed"] = True
            result["detail"] = "Alle verwachte info aanwezig"

    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HAGA RAG — Urologie Evaluatie (folders 1248-1258)"
    )
    parser.add_argument("--save", action="store_true", help="Sla resultaten op als JSON")
    args = parser.parse_args()

    print("=" * 62)
    print("🧪 Urologie RAG Evaluatie — folders 1248-1258")
    print("=" * 62)

    # Servercheck
    print(f"\n🔗 Server: {SERVER_URL}")
    try:
        with httpx.Client(timeout=5) as client:
            stats = client.get(f"{SERVER_URL}/api/stats").json()
        print(f"   ✓ Verbonden — {stats['total_chunks']} chunks | "
              f"LLM: {stats['llm_model']} | "
              f"hybrid: {stats.get('hybrid_enabled', '?')}")
    except Exception:
        print(f"   ❌ Server niet bereikbaar. Start eerst:\n      python server.py")
        sys.exit(1)

    print(f"\n   {len(TEST_QUESTIONS)} testvragen — "
          f"{sum(1 for t in TEST_QUESTIONS if not t['should_refuse'])} inhoud, "
          f"{sum(1 for t in TEST_QUESTIONS if t['should_refuse'])} trap\n")

    # ── Tests uitvoeren ──
    results = []
    category_emoji = {
        "feitelijk":   "📌",
        "procedureel": "📋",
        "medisch":     "🏥",
        "nazorg":      "🩹",
        "trap":        "🪤",
    }

    for i, test in enumerate(TEST_QUESTIONS, 1):
        emoji = category_emoji.get(test["category"], "❓")
        print(f"{'─' * 62}")
        print(f"{emoji} [{i:02d}/{len(TEST_QUESTIONS)}] {test['question']}")
        if test.get("source_hint"):
            print(f"   Verwachte folder(s): {test['source_hint']}")

        t0 = time.time()
        response = call_chat(test["question"])
        elapsed = time.time() - t0

        if response is None:
            results.append({
                "question": test["question"],
                "passed": False,
                "category": test["category"],
                "detail": "Server error",
                "response_time": elapsed,
            })
            print("   ❌ Geen antwoord ontvangen")
            continue

        # Antwoord preview
        preview = response["answer"][:200].replace("\n", " ")
        print(f"\n   💬 ({elapsed:.1f}s) {preview}...")

        # Bronnen
        if response.get("sources"):
            src_names = [s["filename"][:40] for s in response["sources"][:3]]
            print(f"   📎 {' | '.join(src_names)}")

        # Evalueer
        evaluation = evaluate_answer(response, test)
        evaluation["response_time"] = elapsed
        results.append(evaluation)

        status = "✅ PASS" if evaluation["passed"] else "❌ FAIL"
        print(f"   {status}: {evaluation['detail']}")

    # ── SAMENVATTING ──
    print(f"\n{'=' * 62}")
    print("📋 SAMENVATTING RAG EVALUATIE — Urologie")
    print("=" * 62)

    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    pass_rate = passed / total * 100
    print(f"\n   Totaal:    {passed}/{total} geslaagd ({pass_rate:.0f}%)")

    # Per categorie
    categories = sorted(set(r.get("category", "?") for r in results))
    for cat in categories:
        cat_r = [r for r in results if r.get("category") == cat]
        cat_p = sum(1 for r in cat_r if r.get("passed"))
        bar = "█" * cat_p + "░" * (len(cat_r) - cat_p)
        print(f"   {cat:12s}: {bar}  {cat_p}/{len(cat_r)}")

    # Response tijd
    times = [r["response_time"] for r in results if "response_time" in r]
    if times:
        print(f"\n   Gem. response tijd: {sum(times)/len(times):.1f}s")

    # Failures
    failures = [r for r in results if not r.get("passed")]
    if failures:
        print(f"\n   ❌ Gefaalde tests ({len(failures)}):")
        for f in failures:
            print(f"     → {f['question'][:55]}...")
            print(f"       {f.get('detail', '?')}")

    # Hallucinatie-check
    trap_r = [r for r in results if r.get("category") == "trap"]
    if trap_r:
        halluc = sum(1 for r in trap_r if not r.get("passed"))
        print(f"\n   🪤 Hallucinatie-detectie: {halluc}/{len(trap_r)} onterecht zeker")
        if halluc == 0:
            print("      ✅ Model hallucineert niet bij vragen buiten de folders")
        else:
            print("      ⚠ Model verzint antwoorden — system prompt aanscherpen")

    # Verdict
    print(f"\n{'─' * 62}")
    if pass_rate >= 85:
        print("   ✅ Urologie retrieval werkt uitstekend!")
    elif pass_rate >= 70:
        print("   ⚡ Urologie retrieval werkt redelijk — zie failures hierboven")
    else:
        print("   ❌ Urologie retrieval heeft werk nodig — check of folders geïngesteerd zijn")
    print()

    # Opslaan
    if args.save:
        output = {
            "timestamp": datetime.now().isoformat(),
            "server": SERVER_URL,
            "suite": "urologie-1248-1258",
            "pass_rate": round(pass_rate, 1),
            "passed": passed,
            "total": total,
            "results": results,
        }
        outfile = f"rag_eval_urologie_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Resultaten opgeslagen: {outfile}")


if __name__ == "__main__":
    main()
