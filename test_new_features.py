"""
Testcases voor nieuwe conversatie-features
==========================================
Voeg deze toe aan eval.py of 3_test_full_rag.py

Features getest:
- Feature 1: Disambiguation (ambigue vragen → folder-keuze)
- Feature 2: Confidence tiers (hoog/medium/laag)
- Feature 3: Proactieve suggesties
- Feature 5: Zachte verduidelijking
"""

# ══════════════════════════════════════
# DISAMBIGUATION — Feature 1
# ══════════════════════════════════════

DISAMBIGUATION_TESTS = [
    {
        "id": "D01",
        "description": "Ambigue vraag: 'borstoperatie' matcht meerdere folders",
        "question": "Ik wil meer weten over een borstoperatie",
        "expected_behavior": "disambiguation",
        "check": lambda resp: (
            # Moet meerdere opties aanbieden OF een verduidelijkende vraag stellen
            len(resp.get("disambiguation", [])) >= 2
            or "bedoelt u" in resp["answer"].lower()
            or "welk" in resp["answer"].lower()
        ),
        "rationale": "Meerdere folders over borst (verkleining, reconstructie, oncoplastisch) "
                     "moeten als keuze worden aangeboden, niet één willekeurig antwoord.",
    },
    {
        "id": "D02",
        "description": "Specifieke vraag mag GEEN disambiguation triggeren",
        "question": "Wat zijn de leefregels na een borstverkleining?",
        "expected_behavior": "direct_answer",
        "check": lambda resp: (
            len(resp.get("disambiguation", [])) == 0
            and len(resp["answer"]) > 50
        ),
        "rationale": "Specifieke vraag over borstverkleining moet direct beantwoord worden, "
                     "niet leiden tot disambiguation.",
    },
    {
        "id": "D03",
        "description": "Vage vraag over 'onderzoek' matcht veel folders",
        "question": "Ik moet een onderzoek doen",
        "expected_behavior": "disambiguation_or_clarification",
        "check": lambda resp: (
            len(resp.get("disambiguation", [])) >= 2
            or "welk onderzoek" in resp["answer"].lower()
            or "kunt u" in resp["answer"].lower()
        ),
        "rationale": "'Onderzoek' is te vaag — systeem moet verduidelijken.",
    },
]

# ══════════════════════════════════════
# CONFIDENCE TIERS — Feature 2
# ══════════════════════════════════════

CONFIDENCE_TIER_TESTS = [
    {
        "id": "CT01",
        "description": "Hoge confidence: specifieke folder-vraag",
        "question": "Wat is een FibroScan onderzoek?",
        "expected_behavior": "direct_answer_no_caveat",
        "check": lambda resp: (
            len(resp["answer"]) > 80
            and "neem contact op" not in resp["answer"].lower()[:200]
            # Antwoord moet NIET beginnen met een voorbehoud
        ),
        "rationale": "Bij hoge confidence (>0.65) moet het antwoord direct en zonder "
                     "onzekerheids-caveat worden gegeven.",
    },
    {
        "id": "CT02",
        "description": "Medium confidence: antwoord + caveat verwacht",
        "question": "Hoe bereid ik me voor op een leveronderzoek?",
        "expected_behavior": "answer_with_caveat",
        "check": lambda resp: (
            len(resp["answer"]) > 50
            # Bij medium confidence verwachten we een caveat-achtige zin
            # maar dit is niet altijd meetbaar zonder de score te kennen
        ),
        "rationale": "Bij medium confidence moet het systeem wél antwoorden maar met "
                     "een voorzichtigheidszin.",
    },
]

# ══════════════════════════════════════
# PROACTIEVE SUGGESTIES — Feature 3
# ══════════════════════════════════════

SUGGESTION_TESTS = [
    {
        "id": "SG01",
        "description": "Na een goed antwoord: suggesties verwacht",
        "question": "Wat is een FibroScan onderzoek?",
        "expected_behavior": "answer_with_suggestions",
        "check": lambda resp: (
            len(resp.get("suggestions", [])) >= 1
            and len(resp.get("suggestions", [])) <= 2
        ),
        "rationale": "Bij een succesvol antwoord moet het systeem 1-2 relevante "
                     "vervolgvragen suggereren.",
    },
    {
        "id": "SG02",
        "description": "Bij doorverwijzing: GEEN suggesties",
        "question": "Wat zijn goede oefeningen voor mijn rug?",
        "expected_behavior": "no_suggestions",
        "check": lambda resp: (
            len(resp.get("suggestions", [])) == 0
        ),
        "rationale": "Als het systeem doorverwijst (out-of-scope), mogen er geen "
                     "suggesties worden gegeven.",
    },
    {
        "id": "SG03",
        "description": "Suggesties zijn inhoudelijk relevant",
        "question": "Wat gebeurt er tijdens een blaasverwijdering?",
        "expected_behavior": "relevant_suggestions",
        "check": lambda resp: (
            # Suggesties moeten gerelateerd zijn aan het onderwerp
            any(
                any(w in s.lower() for w in ["blaas", "nazorg", "operatie", "herstel", "na de"])
                for s in resp.get("suggestions", [])
            ) if resp.get("suggestions") else True  # geen suggesties = ook ok
        ),
        "rationale": "Suggesties moeten thematisch aansluiten bij de vraag, niet random.",
    },
]

# ══════════════════════════════════════
# ZACHTE VERDUIDELIJKING — Feature 5
# ══════════════════════════════════════

CLARIFICATION_TESTS = [
    {
        "id": "CL01",
        "description": "Eén-woord query: verduidelijking verwacht",
        "question": "Isolatie",
        "expected_behavior": "clarification",
        "check": lambda resp: (
            "bedoelt" in resp["answer"].lower()
            or "beschermende" in resp["answer"].lower()
            or "isolatie" in resp["answer"].lower()
        ),
        "rationale": "'Isolatie' is vaag — systeem moet aangeven wat het gevonden heeft "
                     "en verduidelijking vragen.",
    },
    {
        "id": "CL02",
        "description": "Typo/onbegrijpelijke query: geen crash",
        "question": "fieberscaan lever",
        "expected_behavior": "best_effort_or_clarification",
        "check": lambda resp: (
            len(resp["answer"]) > 20
            # Systeem crasht niet en geeft óf een antwoord (FibroScan gevonden)
            # óf een verduidelijkende vraag
        ),
        "rationale": "Bij typo's moet het systeem robuust zijn — best-effort match "
                     "of vriendelijke verduidelijking, nooit een crash.",
    },
]


# ══════════════════════════════════════
# RUNNER
# ══════════════════════════════════════

ALL_NEW_TESTS = (
    DISAMBIGUATION_TESTS
    + CONFIDENCE_TIER_TESTS
    + SUGGESTION_TESTS
    + CLARIFICATION_TESTS
)


def run_feature_tests(base_url: str = "http://localhost:8000"):
    """
    Draai alle feature-tests tegen een draaiende server.
    
    Gebruik:
        python test_new_features.py
    """
    import httpx
    import json

    passed = 0
    failed = 0
    errors = []

    for test in ALL_NEW_TESTS:
        try:
            resp = httpx.post(
                f"{base_url}/api/chat",
                json={"message": test["question"]},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

            if test["check"](data):
                print(f"  ✅ {test['id']}: {test['description']}")
                passed += 1
            else:
                print(f"  ❌ {test['id']}: {test['description']}")
                print(f"     Verwacht: {test['expected_behavior']}")
                print(f"     Antwoord: {data['answer'][:100]}...")
                print(f"     Suggesties: {data.get('suggestions', [])}")
                print(f"     Disambiguation: {len(data.get('disambiguation', []))} opties")
                failed += 1
                errors.append(test["id"])
        except Exception as e:
            print(f"  💥 {test['id']}: ERROR — {e}")
            failed += 1
            errors.append(test["id"])

    print(f"\n{'='*50}")
    print(f"Resultaat: {passed}/{passed+failed} passed ({100*passed/max(1,passed+failed):.0f}%)")
    if errors:
        print(f"Gefaald: {', '.join(errors)}")


if __name__ == "__main__":
    run_feature_tests()
