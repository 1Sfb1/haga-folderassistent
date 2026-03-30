"""
Validatietests voor de 3 server-fixes.
Gebruik: kopieer deze TestCase-blokken in eval.py → get_test_cases()

Elke test gebruikt een ANDERE folder dan de originele faaltests,
zodat we overfitting uitsluiten.
"""

# ══════════════════════════════════════
# FIX 1: DOSAGE GUARD TE BREED (R03-fix)
# Doel: antwoorden met getallen maar ZONDER doseringsvraag mogen NIET geblokkeerd worden.
# Bron: folder "Voeding bij chronische nierinsufficiëntie" — vol met getallen (mg, mmol, %)
# ══════════════════════════════════════

"""
TestCase(
    id="R07",
    suite="retrieval",
    description="Bijwerkingen/risico's met getallen — dosage guard mag NIET triggeren",
    question="Wat zijn de gevaren van te veel kalium bij nierproblemen?",
    expected_contains=["hart"],              # folder noemt expliciet hartklachten/hartstilstand
    expected_absent=["doseringsadvies", "bijsluiter", "apotheker"],  # blokkeer-respons mag NIET
    must_have_source=True,
    llm_judge_criteria="Beschrijft het antwoord concrete risico's van een te hoog kaliumgehalte "
                       "(zoals hartklachten of hartstilstand) op basis van de folder, "
                       "ZONDER dat het antwoord is vervangen door een doseringsweigering?",
),
TestCase(
    id="R08",
    suite="retrieval",
    description="Voedingsbeperking met streefwaarden — getallen in antwoord zijn legitiem",
    question="Hoeveel natrium per dag mag ik eten bij een verminderde nierfunctie?",
    expected_absent=["over het algemeen", "doorgaans"],
    must_have_source=True,
    llm_judge_criteria="Geeft het antwoord een concreet natriumadvies uit de folder "
                       "(2000-2400 mg of 5-6 gram zout) in plaats van door te verwijzen naar "
                       "een arts vanwege 'doseringsregels'? "
                       "LET OP: dit is een voedingsadvies, geen medicijndosering — "
                       "het antwoord MOET de getallen bevatten.",
),
"""

# ══════════════════════════════════════
# FIX 2: rewrite_query_with_context NIET AANGEROEPEN (C03-fix)
# Doel: "daarna", "dan", "erna" in een vervolgvraag moet context ophalen uit eerder gesprek.
# Bron: folder "Rugoperatie/Spondylodese" — autorijden, fietsen, werken staan er concreet in.
# ══════════════════════════════════════

"""
TestCase(
    id="C05",
    suite="conversation",
    description="Meerstaps context — 'daarna' over rugoperatie autorijden",
    question="Mag ik daarna zelf autorijden?",
    history=[
        {"role": "user",      "content": "Ik heb een spondylodese operatie gehad."},
        {"role": "assistant", "content": "Een spondylodese is een rugoperatie waarbij wervels aan "
                                         "elkaar worden vastgezet. Het herstel duurt enkele weken "
                                         "tot maanden."},
        {"role": "user",      "content": "Wanneer mag ik dan weer fietsen?"},
        {"role": "assistant", "content": "Volgens de folder mag u 2 weken na ontslag uit het "
                                         "ziekenhuis weer fietsen, mits u geen doof gevoel in uw "
                                         "been heeft."},
    ],
    expected_contains=["2 weken", "week"],   # folder: "De eerste 2 weken niet zelf autorijden"
    expected_absent=["kan ik niet betrouwbaar"],
    must_have_source=False,
    llm_judge_criteria="Begrijpt het systeem dat 'daarna' verwijst naar herstel na de spondylodese "
                       "en geeft het concrete informatie over autorijden na deze ingreep "
                       "(eerste 2 weken niet zelf rijden)?",
),
TestCase(
    id="C06",
    suite="conversation",
    description="Vervolgvraag 'dan' na sondevoeding-gesprek",
    question="Hoe lang mag die sonde er dan in blijven?",
    history=[
        {"role": "user",      "content": "Mijn kind krijgt een voedingssonde via de neus."},
        {"role": "assistant", "content": "Een voedingssonde is een slangetje dat via de neus naar "
                                         "de maag wordt ingebracht om uw kind te voeden."},
    ],
    expected_contains=["zes weken", "6 weken", "weken"],  # folder: "zes weken in de neus blijven"
    expected_absent=["kan ik niet betrouwbaar"],
    must_have_source=False,
    llm_judge_criteria="Begrijpt het systeem dat 'die sonde' en 'dan' verwijzen naar de "
                       "eerder besproken voedingssonde, en noemt het de maximale verblijfsduur "
                       "van zes weken?",
),
"""

# ══════════════════════════════════════
# FIX 3: FALLBACK WIJST NAAR ARTS I.P.V. HAGA.NL (C02-fix)
# Doel: logistieke vragen (parkeren, OV, openingstijden) → verwijzing naar haga.nl, NIET naar arts.
# Bron: folder "Dagcentrum Volwassenen" bevat parkeerinfo — maar die is verouderd/specifiek.
# ══════════════════════════════════════

"""
TestCase(
    id="C07",
    suite="conversation",
    description="Logistieke vraag na medische context — verwijzing naar haga.nl niet naar arts",
    question="Hoe kom ik met de tram bij het ziekenhuis?",
    history=[
        {"role": "user",      "content": "Ik krijg volgende week een dagbehandeling."},
        {"role": "assistant", "content": "Voor een dagbehandeling wordt u voor één dag opgenomen. "
                                         "U moet nuchter komen en iemand meenemen die u naar huis "
                                         "begeleidt."},
    ],
    expected_contains=["haga", "website", "9292"],   # folder noemt 9292.nl + HTM tram 6
    expected_absent=["behandelend arts", "behandelaar"],  # arts is NIET de juiste doorverwijzing
    must_have_source=False,
    llm_judge_criteria="Verwijst het systeem voor OV-informatie naar haga.nl of 9292.nl in plaats "
                       "van naar de behandelend arts? De arts is geen juiste bron voor "
                       "reisadvies.",
),
TestCase(
    id="C08",
    suite="conversation",
    description="Topicwisseling naar parkeren — mag BCG-context niet meenemen",
    question="Wat kost parkeren bij het ziekenhuis?",
    history=[
        {"role": "user",      "content": "Wat moet ik doen voor de operatie?"},
        {"role": "assistant", "content": "U moet nuchter zijn vanaf middernacht en geen make-up "
                                         "of sieraden dragen."},
    ],
    expected_absent=["operatie", "nuchter", "behandelend arts"],
    must_have_source=False,
    llm_judge_criteria="Geeft het systeem parkeerinformatie of verwijst het naar haga.nl voor "
                       "actuele parkeerkosten, ZONDER onnodige verwijzing naar de vorige "
                       "operatievraag of naar de behandelend arts?",
),
"""

# ══════════════════════════════════════
# REGRESSION — nieuwe bekende faalgevallen
# ══════════════════════════════════════

"""
TestCase(
    id="REG02",
    suite="regression",
    description="Specifieke herstelregel na rugoperatie — eerder mogelijke retrieval-miss",
    question="Wanneer mag ik douchen na de rugoperatie?",
    expected_contains=["derde dag", "dag"],
    expected_absent=["over het algemeen", "doorgaans"],
    must_have_source=True,
    llm_judge_criteria="Geeft het antwoord de concrete douche-instructie uit de folder: "
                       "douchen zonder douchepleister mag op de derde dag na de operatie? "
                       "Het antwoord moet specifiek zijn, niet algemeen.",
),
TestCase(
    id="REG03",
    suite="regression",
    description="Temperatuur sondevoeding — concrete instructie moet gevonden worden",
    question="Hoe warm moet de sondevoeding zijn als ik die aan mijn kind geef?",
    expected_absent=["over het algemeen", "doorgaans", "kan ik niet betrouwbaar"],
    must_have_source=True,
    llm_judge_criteria="Geeft het antwoord de concrete temperatuurinstructie uit de folder: "
                       "lichaamstemperatuur (~37 graden), controleer met druppel op pols? "
                       "Het antwoord mag NIET alleen doorverwijzen.",
),
"""

print("Testcases klaar — kopieer de TestCase-blokken naar eval.py")
