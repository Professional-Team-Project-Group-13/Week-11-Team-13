"""Clinical triage engine.

Combines the model probability with severity, red-flag symptoms, vital signs
and age into a triage band, priority and recommended action — the box labelled
'Clinical Triage Engine' in the architecture.
"""
import config
from ai import risk_model

_SEV_BUMP = {"minimal": 0, "mild": 0, "moderate": 8, "severe": 16}
_ACTION = {
    "Low concern":          "Routine documentation — no action needed",
    "Monitor":              "Monitor; routine clinician review",
    "Elevated — review":    "Clinician review recommended",
    "High — urgent review": "Priority clinician review — escalate",
}


def _band(pct):
    for lo, hi, label, colour, prio in config.TRIAGE_BANDS:
        if lo <= pct < hi:
            return label, colour, prio
    return config.TRIAGE_BANDS[-1][2:5]


def triage(analysis, symptoms="", age=None, vitals=None):
    vitals = vitals or {}
    proba = analysis["proba"]
    pct = proba * 100.0

    # escalation modifiers
    reasons = []
    flags = analysis.get("red_flags") or risk_model.red_flags(symptoms)
    v_alerts = risk_model.vitals_alerts(vitals)

    pct += _SEV_BUMP.get(analysis.get("severity", "mild"), 0)
    if age is not None and age >= 65:
        pct += 6; reasons.append("age ≥ 65")
    if v_alerts:
        pct += 12; reasons.append("abnormal vitals: " + ", ".join(v_alerts))
    if flags:
        pct = max(pct, 90); reasons.append("red-flag symptom: " + ", ".join(flags))
    pct = max(0.0, min(100.0, pct))

    label, colour, priority = _band(pct)
    decision_flag = proba >= config.DECISION_THRESHOLD

    # routing: P1/P2 -> doctor queue, else nurse queue
    route = "doctor" if priority in ("P1", "P2") else "nurse"

    return {
        "triage_pct": round(pct, 1),
        "triage_band": label,
        "band_colour": colour,
        "priority": priority,
        "action": _ACTION[label],
        "route_to": route,
        "decision_flag": decision_flag,
        "escalation_reasons": reasons,
        "vital_alerts": v_alerts,
        "red_flags": flags,
    }
