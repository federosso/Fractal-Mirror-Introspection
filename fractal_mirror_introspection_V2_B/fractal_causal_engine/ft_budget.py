"""
ft_budget.py — budget di token di output per attore: UNICO punto di modifica.

Prima di questo file i budget erano letterali sparsi in 8 moduli, calibrati
per modelli che rispondono in JSON puro. Con un modello *thinking* (che brucia
token di ragionamento prima del JSON) quei tetti troncavano quasi ogni prima
chiamata: 2 retry a raddoppio per attore (rigenerazioni complete), run lenti
e materia persa (parse_failed = item e ipotesi mai arrivati al ventaglio).

Regole:
- max_tokens è un TETTO, non un target: il modello si ferma allo stop token,
  quindi alzare i valori non allunga le risposte buone — elimina i retry.
- Il retry a raddoppio (llm.py, MAX_TRUNCATION_RETRIES) resta come rete di
  sicurezza sopra questi valori.
- MOLTIPLICATORE scala tutto insieme: comodo cambiando modello (1.0 per i
  thinking coi valori sotto; riducibile, es. 0.5, per modelli che rispondono
  in JSON puro senza token di ragionamento).

I valori storici sono annotati accanto a ogni voce.
"""
from __future__ import annotations

MOLTIPLICATORE: float = 1.0

BUDGET: dict[str, int] = {
    # --- L1/L2: lettura del testo -----------------------------------------
    "l1_classifier":        5000,   # era 900
    "l2_locked":            3000,   # era 500 (per scala)

    # --- L3A: observer non bloccati (esplorazione e ipotesi) ---------------
    "l3a1_domain_knowledge": 3000,  # era 500
    "l3a2_causal_principles": 3000, # era 400
    "l3a3_cross_domain":    3000,   # era 400
    "l3a4_open_questions":  3000,   # era 300
    "l3a5_global_synthesis": 3000,  # era 500

    # --- L3B/L4/L5: validazione, regia, espansione --------------------------
    "l3b_crossscale_rilevatore": 3000,  # era 700
    "l3b_crossscale_validator":  3000,  # era 900
    "l5_expander":          3000,   # era 900
    "bridge":               3000,   # era 600
    "director":             3000,   # era 900
    "magistrale":           3000,   # era 1400

    # --- runner tematici (fuori dal loop FMI) --------------------------------
    "thematic_book":        3000,   # era 1000
    "thematic_book_opera":  3000,   # era 1200
}


def budget(chiave: str) -> int:
    """Il tetto di token per l'attore, scalato dal moltiplicatore globale."""
    return max(1, int(BUDGET[chiave] * MOLTIPLICATORE))
