"""
test_ponte.py — esegue il ponte in mock (no backend live) e verifica gli
invarianti duri dell'handoff. La chiamata di rete allo Specchio è sostituita
da un reader finto che restituisce l'input composto, così si verifica l'esatto
payload che lo Specchio riceverebbe.
"""
import sys, pathlib

# path: package Fractal (src/), specchio_adapter, e il ponte
# Layout integrato (questa cartella è la radice del progetto):
#   ./                                   -> ponte_fractal_specchio.py, test_ponte.py
#   ./fractal_causal_engine/     -> package Fractal (sorgente semplice)
#   ./specchio_di_coscienza/             -> specchio_adapter.py + i .md
HERE = pathlib.Path(__file__).resolve().parent
SPECCHIO = HERE / "specchio_di_coscienza"
sys.path.insert(0, str(HERE))          # ponte_fractal_specchio
sys.path.insert(0, str(SPECCHIO))      # specchio_adapter + .md

from fractal_causal_engine.llm import LLMClient, LLMConfig
import ponte_fractal_specchio as P


MANIFESTAZIONE = (
    "Un uomo parla con voce ferma di un lutto recente, lessico tecnico e "
    "controllato; ma le mani gli tremano e la voce cede su una parola."
)


def reader_finto(input_composto, system_prompt):
    """Sostituisce la rete: registra ciò che lo Specchio riceverebbe."""
    reader_finto.ultimo_input = input_composto
    reader_finto.ultimo_system = system_prompt
    return "[LETTURA SPECCHIO — non eseguita in mock: richiede backend live]"


def main():
    client = LLMClient(LLMConfig(mock=True))

    # forziamo top_n alto per espandere e ottenere candidati anche col mock generico
    ft, records = P.genera(MANIFESTAZIONE, client=client, top_n_espansioni=5)

    # se il mock L1 non ha prodotto item CAUSE/BRIDGE (è generico), espandiamo
    # comunque un parent costruito a mano per esercitare l'estrazione end-to-end.
    if not records:
        from fractal_causal_engine.ft_model import ClassifiedItem, Nature, PredicateType, EpistemicStatus
        from fractal_causal_engine.ft_expander import FractalExpander
        parent = ClassifiedItem(id="p1", quote="la voce cede su una parola",
                                predicate=PredicateType.STATE, nature=Nature.CAUSE,
                                scale="organismo", epistemic_status=EpistemicStatus.TEXT_OBSERVED)
        rec = FractalExpander(client, llm_calls_dir=None).expand(
            parent, original_text=MANIFESTAZIONE, trace=[])
        records = [rec]

    ventaglio = P.estrai_ventaglio(ft, records)
    blocco = P.serializza_ventaglio(ventaglio)
    input_composto = P.componi_input(MANIFESTAZIONE, blocco)

    # system prompt con Regola 8 (usa i .md reali dello Specchio)
    system_prompt = P.monta_system_prompt(
        str(SPECCHIO / "specchio_di_coscienza_nucleo.md"),
        str(SPECCHIO / "specchio_di_coscienza_contratto_di_output.md"),
    )
    lettura = reader_finto(input_composto, system_prompt)

    # ---- INVARIANTI DURI -------------------------------------------------
    errori = []

    # #1 mai magistrale
    if ft.magistrale is not None:
        errori.append("magistrale costruita (deve essere None)")

    # #2 rampa NON appiattita: nessun candidato deve avere tag fuori dai 5 livelli
    livelli = {"text_observed", "domain_knowledge", "causal_model",
               "cross_domain_analogy", "speculative_extension"}
    for c in ventaglio.candidati:
        if c.epistemic not in livelli:
            errori.append(f"tag epistemico fuori rampa: {c.epistemic}")

    # #3 nessun candidato è osservato (solo generati nel ventaglio)
    if any(c.epistemic == "text_observed" for c in ventaglio.candidati):
        errori.append("ventaglio contiene un osservato (deve contenere solo generati)")

    # #4 nessun EFFECT/CONTEXT/INTERPRETATION nel ventaglio
    if any(c.nature not in ("cause", "bridge") for c in ventaglio.candidati):
        errori.append("ventaglio contiene nature non ammesse")

    # #5 la manifestazione è preservata intatta in testa all'input
    if not input_composto.startswith(MANIFESTAZIONE):
        errori.append("la manifestazione non è preservata in testa")

    # #6 il blocco è marcato-generato
    if P.VENTAGLIO_HEADER not in input_composto:
        errori.append("manca l'intestazione marcato-generato")

    # #7 la Regola 8 è nel system prompt
    if "Ventaglio candidato esterno" not in system_prompt:
        errori.append("Regola 8 assente dal system prompt")

    # #8 il sentinella ha tagliato gli esempi human-only del contratto
    if "Esempio schematico" in system_prompt:
        errori.append("il sentinella FINE_PROMPT non ha tagliato gli esempi")

    # ---- COLLEGAMENTO UNLOCKED → VENTAGLIO (fix osservatori scollegati) ----
    from fractal_causal_engine.ft_model import (
        FractalTriadResult, UnlockedReport, DomainConcept, CausalPrinciple,
        CrossDomainAnalogy, EpistemicStatus as ES)
    u = UnlockedReport(
        domain="introspezione",
        domain_knowledge=[DomainConcept(concept=f"concetto {i}",
                                        relation_to_input="evocato dal testo",
                                        suggested_scale="organismo" if i == 0 else "")
                          for i in range(5)],                    # 5 → cap a 3
        causal_principles=[CausalPrinciple(name=f"principio {i}",
                                           description="desc") for i in range(4)],
        cross_domain_analogies=[CrossDomainAnalogy(domain="musica",
                                                   analogy="tema e variazione",
                                                   warning="solo strutturale")],
        open_questions=["quale vincolo ha stretto la scelta?", "q2", "q3"])
    ft_u = FractalTriadResult(unlocked=u)
    v_u = P.estrai_ventaglio(ft_u, [])   # NESSUNA espansione: solo observer

    # #9 con espansioni a zero il ventaglio NON è più vuoto (gli observer entrano)
    if not v_u.candidati:
        errori.append("#9 ventaglio vuoto nonostante il raccolto unlocked")

    # #10 cap per categoria rispettati (3+3+1+2 = 9 col fixture sopra)
    if len(v_u.candidati) != 9:
        errori.append(f"#10 cap unlocked non rispettati: {len(v_u.candidati)} candidati")

    # #11 provenienza dichiarata e rampa enum-valida (serializza non deve rompersi)
    if not all(c.parent_id.startswith("L3A") for c in v_u.candidati):
        errori.append("#11 candidato unlocked senza fonte L3A in parent_id")
    try:
        for c in v_u.candidati:
            ES(c.epistemic)
        blocco_u = P.serializza_ventaglio(v_u)
    except ValueError as exc:
        errori.append(f"#11 rampa fuori enum nei candidati unlocked: {exc}")
        blocco_u = ""

    # #12 la scala suggerita canonica è usata; le non canoniche hanno etichetta di fonte
    scale_u = {c.scale for c in v_u.candidati}
    if "organismo" not in scale_u or not {"dominio", "principio",
                                          "analogia", "domanda_aperta"} <= scale_u:
        errori.append(f"#12 scale dei candidati unlocked inattese: {scale_u}")

    # #13 il trust resta 'bassa' (le ipotesi non sono nessi validati) ma il
    #     motivo dichiara la provenienza
    if v_u.trust != "bassa" or "observer" not in v_u.trust_motivo:
        errori.append(f"#13 trust/motivo: {v_u.trust} · {v_u.trust_motivo}")
    if blocco_u and "ventaglio vuoto" in blocco_u:
        errori.append("#13 il blocco serializzato dichiara vuoto un ventaglio pieno")

    # ---- REPORT ----------------------------------------------------------
    print("=" * 70)
    print("INPUT CHE LO SPECCHIO RICEVEREBBE:")
    print("=" * 70)
    print(input_composto)
    print()
    print("=" * 70)
    print("CODA DEL SYSTEM PROMPT (Regola 8):")
    print("=" * 70)
    print(system_prompt[-len(P.REGOLA_8) - 40:])
    print()
    print("=" * 70)
    print(f"candidati nel ventaglio: {len(ventaglio.candidati)} | trust: {ventaglio.trust}")
    print(f"magistrale: {ft.magistrale}")
    print("=" * 70)

    if errori:
        print("\n❌ INVARIANTI VIOLATI:")
        for e in errori:
            print("   -", e)
        sys.exit(1)
    print("\n✅ TUTTI GLI INVARIANTI DURI RISPETTATI")


if __name__ == "__main__":
    main()
