"""
Explainability visuals .


Shared design language across every figure:
  * one clinical-teal theme (matches ui_theme.py)
  * semantic colour: RED = pushes TOWARD ADE (+), GREEN = pushes AWAY (-)
  * a meaning-carrying anchor (0 line / decision threshold) on every plot
  * rich hovertemplates
  * a one-line interpretation caption so each figure reads as clinical evidence
"""

import html as _html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- palette ----------------------------------------------------------------
TEAL, TEAL_DK = "#0F766E", "#115E59"
GREEN, AMBER, RED = "#10B981", "#F59E0B", "#DC2626"
SLATE, SLATE_MID, GRID, PAPER = "#0F172A", "#475569", "#E2E8F0", "#FFFFFF"
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"

# TOWARD ADE (+) is red, AWAY (-) is green — used everywhere for consistency.
def _sign_colour(w):
    return RED if w >= 0 else GREEN


def _theme(fig, height, title=None):
    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=PAPER, height=height,
        font=dict(family=FONT, color=SLATE),
        margin=dict(l=40, r=40, t=70 if title else 40, b=40),
        hoverlabel=dict(font_size=13, font_family=FONT),
    )
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center",
                                     font=dict(size=20, color=TEAL_DK)))
    return fig


def _caption(fig, text, y=-0.16):
    fig.add_annotation(x=0.5, y=y, xref="paper", yref="paper", showarrow=False,
                       text=text, font=dict(size=13, color=SLATE_MID), align="center")
    return fig



# 1. TRIAGE GAUGE  

_BANDS = [
    (0,   20,  "Low concern",          "#DCFCE7", GREEN),
    (20,  50,  "Monitor",              "#CCFBF1", TEAL),
    (50,  80,  "Elevated — review",    "#FEF3C7", AMBER),
    (80, 100,  "High — urgent review", "#FEE2E2", RED),
]
DECISION_THRESHOLD = 50.0  # % — classifier 0.5 operating point


def _norm_sev(severity):
    if isinstance(severity, (int, float)):
        w = severity * 3 if severity <= 1 else min(severity, 3)
        lab = {0: "Minimal", 1: "Mild", 2: "Moderate", 3: "Severe"}[round(min(max(w, 0), 3))]
        return f"{lab} ({severity})", int(round(min(max(w, 0), 3)))
    s = str(severity).strip().lower()
    table = {"minimal": 0, "none": 0, "low": 1, "mild": 1, "moderate": 2,
             "medium": 2, "high": 3, "severe": 3, "serious": 3, "critical": 3}
    return str(severity), table.get(s, 2)


def _band_for(pct):
    for lo, hi, lab, fill, bar in _BANDS:
        if lo <= pct < hi or (hi == 100 and pct >= 100):
            return lab, fill, bar
    return _BANDS[-1][2], _BANDS[-1][3], _BANDS[-1][4]


def _action(band, sev_w):
    m = {
        "Low concern":          ["Routine documentation", "Routine documentation", "Flag for monitoring", "Clinician review"],
        "Monitor":              ["Monitor", "Monitor", "Schedule review", "Priority review"],
        "Elevated — review":    ["Clinician review", "Clinician review", "Priority review", "Escalate"],
        "High — urgent review": ["Priority review", "Priority review", "Escalate", "Escalate — urgent"],
    }
    return m[band][sev_w]


def triage_gauge(proba, severity, title="Triage confidence — clinical case"):
    pct = float(proba) * 100.0
    sev_label, sev_w = _norm_sev(severity)
    band, _fill, bar = _band_for(pct)
    dist = pct - DECISION_THRESHOLD
    margin, side = abs(dist), ("above" if dist >= 0 else "below")
    conf = ("Strong — far from the decision boundary" if margin >= 30 else
            "Moderate — clear of the boundary" if margin >= 15 else
            "Borderline — close to the 0.5 boundary")

    fig = make_subplots(rows=2, cols=1, row_heights=[0.60, 0.40], vertical_spacing=0.06,
                        specs=[[{"type": "indicator"}], [{"type": "table"}]])
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta", value=pct,
        number={"suffix": "%", "font": {"size": 46, "color": SLATE}},
        delta={"reference": DECISION_THRESHOLD, "suffix": " pts",
               "increasing": {"color": RED}, "decreasing": {"color": GREEN}, "font": {"size": 16}},
        gauge={"axis": {"range": [0, 100], "dtick": 10, "tickwidth": 1.4, "tickcolor": SLATE_MID,
                        "ticks": "outside", "ticklen": 7, "tickfont": {"size": 12, "color": SLATE_MID}},
               "bar": {"color": bar, "thickness": 0.28}, "bgcolor": PAPER,
               "borderwidth": 1, "bordercolor": GRID,
               "steps": [{"range": [lo, hi], "color": fl} for lo, hi, _l, fl, _b in _BANDS],
               "threshold": {"line": {"color": SLATE, "width": 3}, "thickness": 0.82,
                             "value": DECISION_THRESHOLD}},
        domain={"x": [0.05, 0.95], "y": [0, 1]}), row=1, col=1)

    rows = [("P(ADE)", f"{pct:.1f}%"), ("Severity", sev_label), ("Triage band", band),
            ("Decision threshold", f"{DECISION_THRESHOLD:.0f}%  (0.5 operating point)"),
            ("Distance to boundary", f"{margin:.1f} pts {side} threshold"),
            ("Confidence", conf), ("Recommended action", _action(band, sev_w))]
    fig.add_trace(go.Table(columnwidth=[34, 66],
        header=dict(values=["<b>Analysis</b>", "<b>Interpretation</b>"], fill_color=TEAL,
                    align="left", font=dict(color="white", size=14), height=32),
        cells=dict(values=[[r[0] for r in rows], [r[1] for r in rows]],
                   fill_color=[["#F8FAFC", "#EEF2F6"] * 4], align="left",
                   font=dict(color=SLATE, size=13), height=28)), row=2, col=1)

    _theme(fig, 620, title)
    fig.add_annotation(x=0.5, y=0.44, xref="paper", yref="paper", showarrow=False,
        text=f"<span style='color:{bar}'>&#9679;</span> <b>{band}</b> &nbsp;·&nbsp; severity <b>{sev_label}</b>",
        font=dict(size=15, color=SLATE))
    return fig



# 2. LIME vs SHAP  

def _diverging_bar(fig, words, row, col):
    """words: list of (token, weight). Sorted by |weight|, largest on top."""
    items = sorted(words, key=lambda kv: abs(kv[1]))        # ascending -> biggest at top
    labels = [w for w, _ in items]
    vals = [v for _, v in items]
    colours = [_sign_colour(v) for v in vals]
    fig.add_trace(go.Bar(
        x=vals, y=labels, orientation="h", marker=dict(color=colours),
        hovertemplate="<b>%{y}</b><br>contribution %{x:+.3f}"
                      "<br>%{customdata}<extra></extra>",
        customdata=["pushes toward ADE" if v >= 0 else "pushes away from ADE" for v in vals],
        showlegend=False), row=row, col=col)
    fig.add_vline(x=0, line=dict(color=SLATE, width=2), row=row, col=col)


def lime_vs_shap(lime_words, shap_words, left_label="LIME", right_label="SHAP"):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.16,
                        subplot_titles=(f"<b>{left_label}</b>", f"<b>{right_label}</b>"))
    _diverging_bar(fig, lime_words, 1, 1)
    _diverging_bar(fig, shap_words, 1, 2)

    for c in (1, 2):
        fig.update_xaxes(title_text="contribution to P(ADE)", zeroline=False,
                         gridcolor=GRID, row=1, col=c)
        fig.update_yaxes(automargin=True, row=1, col=c)

    net_l, net_r = sum(w for _, w in lime_words), sum(w for _, w in shap_words)
    dir_l = "toward ADE" if net_l >= 0 else "away from ADE"
    dir_r = "toward ADE" if net_r >= 0 else "away from ADE"
    _theme(fig, 460, "Driver words — LIME vs SHAP")
    _caption(fig, f"<span style='color:{RED}'>&#9679;</span> toward ADE &nbsp; "
                  f"<span style='color:{GREEN}'>&#9679;</span> away from ADE &nbsp;·&nbsp; "
                  f"net {left_label} {net_l:+.2f} ({dir_l}) &nbsp;·&nbsp; "
                  f"net {right_label} {net_r:+.2f} ({dir_r})", y=-0.22)
    return fig



# 3. AGREEMENT INDICATOR  

def agreement_indicator(agreement):
    pct = float(agreement) * 100.0
    read = ("High — both methods point to the same drivers" if pct >= 60 else
            "Moderate — partial overlap in drivers" if pct >= 30 else
            "Low — methods disagree; interpret with caution")
    bar = GREEN if pct >= 60 else AMBER if pct >= 30 else RED

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number={"suffix": "%", "font": {"size": 44, "color": SLATE}},
        gauge={"axis": {"range": [0, 100], "dtick": 10, "tickwidth": 1.4, "tickcolor": SLATE_MID,
                        "ticks": "outside", "ticklen": 7, "tickfont": {"size": 12, "color": SLATE_MID}},
               "bar": {"color": bar, "thickness": 0.3}, "bgcolor": PAPER,
               "borderwidth": 1, "bordercolor": GRID,
               "steps": [{"range": [0, 30], "color": "#FEE2E2"},
                         {"range": [30, 60], "color": "#FEF3C7"},
                         {"range": [60, 100], "color": "#DCFCE7"}]}))
    _theme(fig, 420, "LIME / SHAP agreement")
    _caption(fig, f"<b>{read}</b>", y=0.02)
    return fig



# 4. TOKEN HIGHLIGHT (HTML)  

def highlight_tokens_html(lime_words, text=None):
    """
    Colour-coded explanation.
      * text given  -> highlight the driver tokens INSIDE the sentence
      * text None   -> render the ranked driver tokens as coloured chips
    Colour = sign (red toward ADE / green away), intensity = |weight|.
    """
    weights = {w.lower(): v for w, v in lime_words}
    mx = max((abs(v) for v in weights.values()), default=1.0) or 1.0

    def style(v):
        alpha = 0.15 + 0.65 * min(abs(v) / mx, 1.0)
        base = "220,38,38" if v >= 0 else "16,185,129"
        return f"background:rgba({base},{alpha:.2f});border-radius:4px;padding:1px 4px;"

    legend = (f"<div style='font:13px {FONT};color:{SLATE_MID};margin:6px 0'>"
              f"<span style='{style(mx)}'>toward ADE</span>&nbsp;&nbsp;"
              f"<span style='{style(-mx)}'>away from ADE</span>&nbsp;&nbsp;"
              f"intensity ∝ contribution</div>")

    if text:
        out = []
        for tok in text.split():
            key = "".join(ch for ch in tok.lower() if ch.isalnum())
            if key in weights:
                v = weights[key]
                out.append(f"<span style='{style(v)}' title='{v:+.3f}'>{_html.escape(tok)}</span>")
            else:
                out.append(_html.escape(tok))
        body = " ".join(out)
        return (f"{legend}<div style='font:16px/1.7 {FONT};color:{SLATE};"
                f"padding:10px 12px;border:1px solid {GRID};border-radius:8px'>{body}</div>")

    chips = "".join(
        f"<span style='{style(v)};margin:3px;display:inline-block;font:14px {FONT};"
        f"color:{SLATE}' title='{v:+.3f}'>{_html.escape(w)} <b>{v:+.2f}</b></span>"
        for w, v in sorted(lime_words, key=lambda kv: abs(kv[1]), reverse=True))
    return f"{legend}<div style='padding:6px 0'>{chips}</div>"


# 
# 5. SIMILAR CASES TABLE (FAISS)  
# 
def _coerce_neighbour(n):
    """Return (text, score, score_label, label, domain) from dict / tuple / object."""
    def g(obj, *keys, default=None):
        for k in keys:
            if isinstance(obj, dict) and k in obj:
                return obj[k]
            if hasattr(obj, k):
                return getattr(obj, k)
        return default

    if isinstance(n, (tuple, list)):
        text = next((x for x in n if isinstance(x, str)), "")
        score = next((x for x in n if isinstance(x, (int, float))), None)
        return text, score, "Score", None, None

    text = g(n, "text", "TEXT", "case", "sentence", default="")
    if g(n, "similarity", "score") is not None:
        score, slabel = g(n, "similarity", "score"), "Similarity"
    elif g(n, "distance", "dist") is not None:
        score, slabel = g(n, "distance", "dist"), "Distance"
    else:
        score, slabel = None, "Score"
    label = g(n, "label", "LABEL", "y")
    domain = g(n, "domain", "DOMAIN", "source")
    return text, score, slabel, label, domain


def _fmt_label(v):
    if v in (1, "1", "ADE", "ade", True):
        return "ADE", "#FEE2E2"
    if v in (0, "0", "No ADE", "no ade", False):
        return "No ADE", "#DCFCE7"
    return (str(v) if v is not None else "—"), "#F8FAFC"


def similar_cases_table(neighbours, k=None):
    rows = list(neighbours)[: (k or len(list(neighbours)))]
    ranks, texts, scores, labels, domains, label_fills = [], [], [], [], [], []
    slabel = "Score"
    for i, n in enumerate(rows, 1):
        text, score, slabel, label, domain = _coerce_neighbour(n)
        ranks.append(str(i))
        snip = (text[:110] + "…") if len(text) > 110 else text
        texts.append(snip or "—")
        scores.append(f"{score:.3f}" if isinstance(score, (int, float)) else "—")
        lab_txt, lab_fill = _fmt_label(label)
        labels.append(lab_txt); label_fills.append(lab_fill)
        domains.append(str(domain) if domain is not None else "—")

    n_ade = sum(1 for l in labels if l == "ADE")
    fig = go.Figure(go.Table(
        columnwidth=[6, 12, 12, 14, 56],
        header=dict(values=["<b>#</b>", f"<b>{slabel}</b>", "<b>Label</b>",
                            "<b>Domain</b>", "<b>Case (retrieved precedent)</b>"],
                    fill_color=TEAL, align="left", font=dict(color="white", size=13), height=32),
        cells=dict(values=[ranks, scores, labels, domains, texts],
                   fill_color=[["#F8FAFC"] * len(ranks), ["#F8FAFC"] * len(ranks),
                               label_fills, ["#F8FAFC"] * len(ranks), ["#FFFFFF"] * len(ranks)],
                   align="left", font=dict(color=SLATE, size=12), height=26)))
    _theme(fig, 120 + 28 * max(len(ranks), 1), "Similar known cases — FAISS (BioBERT space)")
    _caption(fig, f"{n_ade}/{len(ranks)} nearest precedents are ADE-positive — "
                  f"supporting evidence for the model's call", y=-0.05)
    return fig



# 6. GITHUB-SAFE STATIC RENDERING

def use_static_renderer(scale=2, width=900):
    """
    Make every Plotly `fig.show()` emit a static PNG (GitHub-renderable).

    Call this once, early in the notebook. Requires kaleido — pin the classic
    build in Colab with:  !pip install -q "kaleido==0.2.1"
    """
    import plotly.io as pio
    pio.renderers.default = "png"
    # default_scale moved modules across kaleido versions; set whichever exists.
    try:
        pio.defaults.default_scale = scale
        pio.defaults.default_width = width
    except Exception:
        try:
            pio.kaleido.scope.default_scale = scale
        except Exception:
            pass
    return f"Plotly renderer set to static PNG (scale={scale})."


# matplotlib token highlighter 
_RED_RGB = (0.863, 0.149, 0.149)     # #DC2626
_GREEN_RGB = (0.063, 0.725, 0.506)   # #10B981


def _chip_rgba(v, mx):
    alpha = 0.18 + 0.62 * min(abs(v) / mx, 1.0)
    r, g, b = _RED_RGB if v >= 0 else _GREEN_RGB
    return (r, g, b, alpha)


def _measure_px(items, fontsize, dpi):
    """Measure each token's rendered width in pixels (independent of height)."""
    import matplotlib.pyplot as plt
    scratch = plt.figure(figsize=(4, 1), dpi=dpi)
    r = scratch.canvas.get_renderer()
    widths = []
    for disp, _v in items:
        t = scratch.text(0, 0, disp, fontsize=fontsize)
        w = t.get_window_extent(renderer=r).width
        widths.append(w + 12)          # +12px for the rounded chip padding
        t.remove()
    plt.close(scratch)
    return widths


def highlight_tokens_png(lime_words, text=None, width_px=900, title=None,
                         fontsize=15, dpi=100):
    """
    GitHub-safe coloured token explanation as a compact matplotlib Figure.

    Parameters mirror highlight_tokens_html:
      * text given -> highlight the driver tokens INSIDE the sentence
      * text None  -> show the ranked driver tokens as coloured chips

    The figure is measured in pixels and sized to fit its content exactly, so
    it never balloons to multiple pages regardless of sentence length. Returns
    a matplotlib Figure; return it as a cell's last expression to embed a PNG.
    """
    import matplotlib.pyplot as plt

    weights = {w.lower(): v for w, v in lime_words}
    mx = max((abs(v) for v in weights.values()), default=1.0) or 1.0

    def token_key(tok):
        return "".join(ch for ch in tok.lower() if ch.isalnum())

    # Build a flat list of (display_text, value) for either mode.
    if text:
        items = [(tok, weights.get(token_key(tok))) for tok in text.split()]
    else:
        items = [(f"{w}  {v:+.2f}", v)
                 for w, v in sorted(lime_words, key=lambda kv: abs(kv[1]),
                                    reverse=True)]

    # pass 1: measure + wrap into lines (pixel space) 
    pad_x, space = 12, 8
    widths = _measure_px(items, fontsize, dpi)
    max_x = width_px - pad_x
    lines, cur, x = [], [], pad_x
    for (disp, v), w in zip(items, widths):
        if x + w > max_x and cur:
            lines.append(cur)
            cur, x = [], pad_x
        cur.append((disp, v, w))
        x += w + space
    if cur:
        lines.append(cur)

    line_h = int(fontsize * 2.0)
    title_h = int(fontsize * 1.9) if title else 0
    legend_h = line_h + 8
    top_pad, bot_pad = 10, 8
    total_h = top_pad + title_h + len(lines) * line_h + legend_h + bot_pad

    # pass 2: draw at exact size (y measured downward from the top) 
    fig = plt.figure(figsize=(width_px / dpi, total_h / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, width_px)
    ax.set_ylim(0, total_h)

    def yat(top_offset):                # convert "px from top" to axis y
        return total_h - top_offset

    cursor = top_pad
    if title:
        ax.text(pad_x, yat(cursor), title, fontsize=13, fontweight="bold",
                color="#115E59", va="top", ha="left")
        cursor += title_h

    for line in lines:
        x = pad_x
        for disp, v, w in line:
            bbox = (dict(boxstyle="round,pad=0.3", fc=_chip_rgba(v, mx),
                         ec="none") if v is not None else None)
            ax.text(x, yat(cursor), disp, fontsize=fontsize, va="top",
                    ha="left", color="#0F172A", bbox=bbox)
            x += w + space
        cursor += line_h

    # legend strip
    cursor += 4
    lx = pad_x
    for lab, col in [("toward ADE", _chip_rgba(mx, mx)),
                     ("away from ADE", _chip_rgba(-mx, mx))]:
        ax.text(lx, yat(cursor), lab, fontsize=11, va="top", ha="left",
                color="#475569",
                bbox=dict(boxstyle="round,pad=0.3", fc=col, ec="none"))
        lx += _measure_px([(lab, None)], 11, dpi)[0] + 18
    ax.text(lx, yat(cursor), "intensity \u221d contribution", fontsize=11,
            va="top", ha="left", color="#475569")

    plt.close(fig)
    return fig
