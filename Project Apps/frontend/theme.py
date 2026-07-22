"""Shared UI: dark clinical-console theme + HTML component builders."""
import streamlit as st

TEAL = "#2BA69A"
RED = "#F87171"
GREEN = "#34D399"

_SOFT = {
    "#DC2626": ("rgba(248,113,113,.18)", "#FCA5A5"),
    "#0FA36B": ("rgba(52,211,153,.18)", "#6EE7B7"),
    "#E0930C": ("rgba(251,191,36,.18)", "#FCD34D"),
    "#0E7C74": ("rgba(43,166,154,.18)", "#6FD3C9"),
}

CSS = """
<style>
:root{
  --bg:#0A1D21;--panel:rgba(255,255,255,.045);--panel-brd:rgba(255,255,255,.10);
  --ink:#E8F1F1;--ink2:#A9C2C4;--ink3:#7C9799;
  --teal:#2BA69A;--teal-dk:#1C7E77;--cyan:#5FCBC0;
  --red:#F87171;--green:#34D399;--amber:#FBBF24;
  --line:rgba(255,255,255,.10);--field:#10262B;
}
html,body,[class*="css"]{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}
.stApp{background:
  radial-gradient(1100px 500px at 88% -8%,rgba(43,166,154,.10),transparent 60%),
  radial-gradient(900px 500px at 0% 0%,rgba(15,60,64,.35),transparent 55%),
  var(--bg) !important;}
.main .block-container{padding-top:1.4rem;}
.stApp, .main, .block-container, p, span, label, li, .stMarkdown, h1,h2,h3,h4,h5,h6{color:var(--ink);}
.stCaption, small, .note{color:var(--ink3) !important;}
[data-testid="stHeader"]{background:transparent;}
.stTextInput input, .stTextArea textarea, .stNumberInput input,
div[data-baseweb="select"]>div{
  background:var(--field) !important;color:var(--ink) !important;
  border:1px solid var(--line) !important;-webkit-text-fill-color:var(--ink) !important;}
.stTextInput input::placeholder, .stTextArea textarea::placeholder{color:#6C8688 !important;}
.stTextInput label, .stTextArea label, .stNumberInput label,
.stSelectbox label, .stCheckbox label{color:var(--ink2) !important;}
.stButton>button{border-radius:9px;font-weight:600;background:var(--field);
  color:var(--ink);border:1px solid var(--line);}
.stButton>button:hover{border-color:var(--teal);color:#fff;}
div[data-testid="stButton"] button[kind="primary"]{background:var(--teal);border:none;color:#04211d;}
div[data-testid="stButton"] button[kind="primary"]:hover{background:#35c2b3;color:#04211d;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0E353A,#0A2226);
  border-right:1px solid rgba(255,255,255,.06);}
section[data-testid="stSidebar"] *{color:#EAF6F5 !important;}
section[data-testid="stSidebar"] .stButton>button{background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.12);width:100%;}
section[data-testid="stSidebar"] .stButton>button:hover{border-color:var(--cyan);background:rgba(95,203,192,.14);}
div[data-testid="stExpander"]{background:var(--panel);border:1px solid var(--panel-brd);border-radius:12px;}
div[data-testid="stExpander"] summary{color:var(--ink) !important;}
.appbar{background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.02));
  border:1px solid var(--panel-brd);color:var(--ink);border-radius:14px;
  padding:15px 20px;margin-bottom:16px;display:flex;align-items:center;gap:14px;backdrop-filter:blur(8px);}
.appbar .logo{width:34px;height:34px;border-radius:9px;position:relative;flex:none;
  background:linear-gradient(140deg,var(--teal),#0a4b46);}
.appbar .logo::before,.appbar .logo::after{content:"";position:absolute;background:#EAF6F5;border-radius:2px;}
.appbar .logo::before{width:3px;height:16px;left:15.5px;top:9px;}
.appbar .logo::after{width:16px;height:3px;left:9px;top:15.5px;}
.appbar h1{font-size:17px;margin:0;color:#fff;}
.appbar .sub{font-family:ui-monospace,monospace;font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--cyan);}
.appbar .who{margin-left:auto;text-align:right;font-size:12px;color:var(--ink2);}
.appbar .who b{color:#fff;}
.card{background:var(--panel);border:1px solid var(--panel-brd);border-radius:14px;
  padding:16px 18px;margin-bottom:14px;backdrop-filter:blur(8px);
  box-shadow:0 12px 30px -18px rgba(0,0,0,.6);}
.card h3{font-size:14.5px;margin:0 0 10px;color:var(--ink);}
.tag{font-family:ui-monospace,monospace;font-size:10px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--cyan);background:rgba(43,166,154,.14);padding:3px 7px;border-radius:5px;
  border:1px solid rgba(43,166,154,.25);}
.pill{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11.5px;font-weight:600;}
.mono{font-family:ui-monospace,monospace;}
.kpi{background:var(--panel);border:1px solid var(--panel-brd);border-radius:12px;
  padding:12px 14px;text-align:center;backdrop-filter:blur(6px);}
.kpi .n{font-family:ui-monospace,monospace;font-size:26px;font-weight:600;color:var(--cyan);}
.kpi .l{font-size:11px;color:var(--ink3);text-transform:uppercase;letter-spacing:.06em;}
.bar{display:grid;grid-template-columns:82px 1fr 46px;align-items:center;gap:8px;margin:5px 0;font-size:12.5px;}
.bar .tk{text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--ink2);}
.bar .track{position:relative;height:15px;background:#0E2529;border:1px solid var(--line);border-radius:4px;}
.bar .fill{position:absolute;top:0;bottom:0;border-radius:3px;}
.bar .mid{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:#5C7679;}
.bar .num{font-family:ui-monospace,monospace;font-size:11px;color:var(--ink2);}
.sentence{font-size:15px;line-height:2.1;color:var(--ink);}
.sentence .tok{padding:1px 5px;border-radius:4px;color:#fff;font-weight:500;}
.prec{border-collapse:collapse;width:100%;font-size:12.5px;}
.prec th{text-align:left;font-size:10px;text-transform:uppercase;color:var(--ink3);padding:6px 8px;border-bottom:1px solid var(--line);}
.prec td{padding:7px 8px;border-bottom:1px solid var(--line);color:var(--ink2);vertical-align:top;}
.legend{font-family:ui-monospace,monospace;font-size:11px;color:var(--ink3);margin-top:6px;}
.legend .sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin:0 5px 0 12px;vertical-align:-1px;}
.statusbar{border-radius:10px;padding:9px 14px;font-size:12.5px;margin-bottom:12px;}
.statusbar.live{background:rgba(52,211,153,.12);border:1px solid rgba(52,211,153,.3);color:#7bf0c4;}
.statusbar.preview{background:rgba(251,191,36,.10);border:1px solid rgba(251,191,36,.3);color:#f7d271;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def appbar(user):
    who = ""
    if user:
        who = (f'<div class="who">{user["full_name"]}<br>'
               f'<b>{user["role"].title()}</b></div>')
    st.markdown(
        f'<div class="appbar"><div class="logo"></div>'
        f'<div><h1>SafetyNet&nbsp;AI</h1>'
        f'<div class="sub">Cross-domain adverse-event triage</div></div>{who}</div>',
        unsafe_allow_html=True)


def status_banner(live):
    if live:
        st.markdown('<div class="statusbar live">&#9679; <b>BioBERT live</b> &mdash; '
                    'predictions run on the fine-tuned model.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="statusbar preview">&#9650; <b>BioBERT not loaded</b> &mdash; '
                    'set <code>MODEL_DIR</code> in config.py (add the model to '
                    'models/biobert). Running transparent preview meanwhile.</div>',
                    unsafe_allow_html=True)


def priority_colour(priority):
    return {"P1": "#DC2626", "P2": "#E0930C", "P3": "#0E7C74", "P4": "#0FA36B"}.get(priority, "#0E7C74")


def band_pill(label, colour):
    bg, fg = _SOFT.get(colour, ("rgba(255,255,255,.1)", "#E8F1F1"))
    return f'<span class="pill" style="background:{bg};color:{fg}">{label}</span>'


def gauge_html(pct, band_label, colour):
    arc = 251.3
    offset = arc * (1 - pct / 100.0)
    _, fg = _SOFT.get(colour, ("", colour))
    return f"""
    <div style="text-align:center">
      <svg viewBox="0 0 200 128" width="220">
        <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#34D399"/>
        <stop offset=".5" stop-color="#FBBF24"/><stop offset="1" stop-color="#F87171"/></linearGradient></defs>
        <path d="M20 118 A80 80 0 0 1 180 118" fill="none" stroke="#12303550" stroke-width="16" stroke-linecap="round"/>
        <path d="M20 118 A80 80 0 0 1 180 118" fill="none" stroke="url(#g)" stroke-width="16"
          stroke-linecap="round" stroke-dasharray="{arc}" stroke-dashoffset="{offset:.1f}"/>
      </svg>
      <div style="margin-top:-16px">
        <div class="mono" style="font-size:38px;font-weight:600;color:{fg}">{pct:.0f}<span style="font-size:18px;color:#7C9799">%</span></div>
        <div style="font-weight:600;color:{fg}">{band_label}</div>
      </div>
    </div>"""


def driver_bars_html(drivers, title=""):
    if not drivers:
        return '<div class="note">No drivers.</div>'
    mx = max(abs(v) for _, v in drivers) or 1e-4
    rows = sorted(drivers, key=lambda kv: abs(kv[1]), reverse=True)
    out = [f'<h4 style="font-size:11px;text-transform:uppercase;color:#A9C2C4;margin:0 0 8px">{title}</h4>'] if title else []
    for tk, v in rows:
        w = abs(v) / mx * 50
        if v >= 0:
            fill = f'<div class="fill" style="left:50%;width:{w:.1f}%;background:{RED}"></div>'
        else:
            fill = f'<div class="fill" style="right:50%;width:{w:.1f}%;background:{GREEN}"></div>'
        out.append(f'<div class="bar"><div class="tk">{tk}</div>'
                   f'<div class="track"><div class="mid"></div>{fill}</div>'
                   f'<div class="num">{v:+.2f}</div></div>')
    return "".join(out)


def sentence_html(text, drivers):
    w = {k.lower(): v for k, v in drivers}
    mx = max((abs(x) for x in w.values()), default=1e-4) or 1e-4
    out = []
    for tok in text.split():
        key = "".join(c for c in tok.lower() if c.isalnum())
        if key in w:
            v = w[key]
            a = 0.30 + 0.55 * min(abs(v) / mx, 1)
            rgb = "239,68,68" if v >= 0 else "16,185,129"
            out.append(f'<span class="tok" style="background:rgba({rgb},{a:.2f})">{tok}</span>')
        else:
            out.append(f'<span style="color:#C7D8D9">{tok}</span>')
    legend = ('<div class="legend"><span class="sw" style="background:#F87171"></span>toward ADE'
              '<span class="sw" style="background:#34D399"></span>away from ADE</div>')
    return f'<div class="sentence">{" ".join(out)}</div>{legend}'


def precedents_html(neighbours):
    rows = ""
    for i, n in enumerate(neighbours, 1):
        if n["label"] == 1:
            lab = '<span class="pill" style="background:rgba(248,113,113,.18);color:#FCA5A5">ADE</span>'
        else:
            lab = '<span class="pill" style="background:rgba(52,211,153,.18);color:#6EE7B7">No ADE</span>'
        rows += (f'<tr><td class="mono">{i}</td><td class="mono">{n["sim"]:.2f}</td>'
                 f'<td>{lab}</td><td class="mono" style="font-size:11px">{n["domain"]}</td>'
                 f'<td>{n["text"]}</td></tr>')
    return ('<table class="prec"><thead><tr><th>#</th><th>Sim</th><th>Label</th>'
            f'<th>Domain</th><th>Retrieved precedent</th></tr></thead><tbody>{rows}</tbody></table>')


def kpi(n, label):
    return f'<div class="kpi"><div class="n">{n}</div><div class="l">{label}</div></div>'
