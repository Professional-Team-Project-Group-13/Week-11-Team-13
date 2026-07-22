"""Severity / risk scoring from text, vitals and age."""
import config


def severity_from_text(text):
    t = (text or "").lower()
    if any(w in t for w in ("severe", "emergency", "hospital", "breathing",
                            "swelling", "collapsed", "anaphylaxis", "seizure")):
        return "severe"
    if any(w in t for w in ("stopped", "vomiting", "unable", "bleeding")):
        return "moderate"
    if any(w in t for w in ("mild", "slight", "a bit", "little")):
        return "mild"
    return "mild"


def vitals_alerts(vitals):
    """Return a list of human-readable vital-sign alerts."""
    alerts = []
    for key, label in [("heart_rate", "Heart rate"), ("systolic_bp", "Systolic BP"),
                       ("temperature", "Temperature"), ("spo2", "SpO₂")]:
        v = vitals.get(key)
        if v in (None, 0):
            continue
        lim = config.VITALS_ALERTS[key]
        if v < lim["low"]:
            alerts.append(f"{label} low ({v})")
        elif v > lim["high"]:
            alerts.append(f"{label} high ({v})")
    return alerts


def red_flags(text):
    t = (text or "").lower()
    return [rf for rf in config.RED_FLAGS if rf in t]
