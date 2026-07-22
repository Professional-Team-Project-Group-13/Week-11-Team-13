"""Clinical PDF report generation for a triaged case (fpdf2)."""
import json
import os
from datetime import datetime

from fpdf import FPDF

import config

TEAL = (14, 124, 116)
INK = (20, 27, 34)
GREY = (90, 112, 118)


def _clean(s):
    """fpdf core fonts are latin-1 only; strip anything outside it."""
    return (str(s) if s is not None else "").encode("latin-1", "replace").decode("latin-1")


def _full(pdf):
    pdf.set_x(pdf.l_margin)
    return pdf.w - pdf.l_margin - pdf.r_margin


class _PDF(FPDF):
    def header(self):
        self.set_fill_color(*TEAL)
        self.rect(0, 0, 210, 22, "F")
        self.set_xy(12, 6)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "SafetyNet AI  -  Clinical Triage Report")
        self.set_y(26)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GREY)
        self.set_x(self.l_margin)
        self.multi_cell(self.w - self.l_margin - self.r_margin, 4, _clean(
            "Research prototype - decision support only, not a medical device. "
            "Reviewed decisions are the responsibility of the qualified clinician. "
            f"Generated {datetime.utcnow().isoformat(timespec='seconds')} UTC."))


def _kv(pdf, k, v):
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 10); pdf.set_text_color(*GREY)
    pdf.cell(46, 7, _clean(k))
    pdf.set_font("Helvetica", "", 10); pdf.set_text_color(*INK)
    avail = pdf.w - pdf.r_margin - pdf.get_x()
    pdf.multi_cell(avail, 7, _clean(v))


def _section(pdf, title):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11); pdf.set_text_color(*TEAL)
    pdf.cell(0, 8, _clean(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*TEAL); pdf.set_line_width(0.3)
    y = pdf.get_y(); pdf.line(12, y, 198, y); pdf.ln(2)


def generate_case_report(case):
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    analysis = {}
    if case.get("analysis_json"):
        try:
            analysis = json.loads(case["analysis_json"])
        except Exception:                          # noqa: BLE001
            analysis = {}

    pdf = _PDF()
    pdf.set_margins(12, 28, 12)
    pdf.set_auto_page_break(True, 16)
    pdf.add_page()
    pdf.set_text_color(*INK)

    _section(pdf, f"Case #{case['id']}  -  {str(case.get('status','')).upper()}")
    _kv(pdf, "Submitted", case.get("created_at", "-"))
    _kv(pdf, "Patient (ref)", case.get("patient_username", "-"))
    _kv(pdf, "Age", case.get("age") or "-")

    _section(pdf, "Presentation")
    _kv(pdf, "Symptoms", case.get("symptoms", "-"))
    _kv(pdf, "Medication", case.get("medication", "-"))
    vit = f"HR {case.get('heart_rate') or '-'}, BP {case.get('systolic_bp') or '-'}, " \
          f"Temp {case.get('temperature') or '-'}, SpO2 {case.get('spo2') or '-'}"
    _kv(pdf, "Vitals", vit)

    _section(pdf, "AI triage assessment")
    _kv(pdf, "P(ADE)", f"{case.get('proba', 0):.2f}   ({case.get('engine','preview')})")
    _kv(pdf, "Prediction", "ADE detected" if case.get("label") else "No ADE")
    _kv(pdf, "Severity", case.get("severity", "-"))
    _kv(pdf, "Triage band", f"{case.get('triage_band','-')}  [{case.get('priority','-')}]")
    _kv(pdf, "Recommended action", case.get("action", "-"))
    _kv(pdf, "Domain", case.get("domain", "-"))
    if case.get("red_flag"):
        _kv(pdf, "Red flag", "YES - urgent escalation")

    drivers = analysis.get("lime") or []
    if drivers:
        _section(pdf, "Key driver words (explainability)")
        line = ", ".join(f"{w} ({v:+.2f})" for w, v in drivers[:8])
        pdf.set_font("Helvetica", "", 10); pdf.set_text_color(*INK)
        pdf.multi_cell(_full(pdf), 6, _clean(line))

    neigh = analysis.get("neighbours") or []
    if neigh:
        _section(pdf, "Similar precedents (FAISS)")
        for n in neigh:
            tag = "ADE" if n.get("label") else "No ADE"
            pdf.set_font("Helvetica", "", 9); pdf.set_text_color(*INK)
            pdf.multi_cell(_full(pdf), 5, _clean(f"[{n.get('sim',0):.2f} - {tag} - "
                                          f"{n.get('domain','')}]  {n.get('text','')}"))

    _section(pdf, "Clinician decision")
    _kv(pdf, "Clinician", case.get("clinician_username", "-"))
    _kv(pdf, "Decision at", case.get("decided_at", "-"))
    _kv(pdf, "Note", case.get("clinician_note", "-"))

    out = os.path.join(config.REPORTS_DIR, f"case_{case['id']}.pdf")
    pdf.output(out)
    return out
