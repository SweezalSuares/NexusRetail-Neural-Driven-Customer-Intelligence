"""Reusable UI pieces: CSS theme, KPI cards, badges and the Plotly template."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src import config as C

CSS = f"""
<style>
html, body, [class*="css"] {{ font-family: "Inter", "Segoe UI", system-ui, sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {C.CHARCOAL_LIGHT} 0%, {C.CHARCOAL} 100%);
  border-right: 1px solid {C.CHARCOAL_MID}; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.hero {{ background: linear-gradient(120deg, {C.TEAL_DARK} 0%, {C.TEAL} 55%, {C.MINT_DARK} 100%);
  border-radius: 18px; padding: 26px 30px; margin-bottom: 18px; box-shadow: 0 8px 28px rgba(15,139,141,.25); }}
.hero h1 {{ margin: 0; font-size: 1.9rem; color: #fff; }}
.hero p {{ margin: 6px 0 0 0; color: {C.MINT_LIGHT}; font-size: 1rem; }}
.kpi {{ background: {C.CHARCOAL_LIGHT}; border: 1px solid {C.CHARCOAL_MID}; border-left: 4px solid {C.MINT};
  border-radius: 14px; padding: 14px 16px; transition: transform .15s ease, box-shadow .15s ease; height: 100%; }}
.kpi:hover {{ transform: translateY(-2px); box-shadow: 0 6px 18px rgba(62,219,180,.15); }}
.kpi .label {{ color: {C.MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }}
.kpi .value {{ color: {C.TEXT}; font-size: 1.6rem; font-weight: 700; margin-top: 2px; }}
.kpi .sub {{ color: {C.MINT}; font-size: .8rem; margin-top: 2px; }}
.card {{ background: {C.CHARCOAL_LIGHT}; border: 1px solid {C.CHARCOAL_MID}; border-radius: 14px; padding: 16px 18px;
  margin-bottom: 12px; height: 100%; }}
.card h4 {{ margin: 0 0 4px 0; font-size: .95rem; color: {C.TEXT}; }}
.card .big {{ font-size: 1.55rem; font-weight: 700; color: {C.MINT}; margin: 2px 0; }}
.card .big.muted {{ color: {C.MUTED}; }}
.card .q {{ color: {C.MUTED}; font-size: .78rem; margin-bottom: 4px; }}
.card .note {{ color: {C.MUTED}; font-size: .78rem; margin-top: 6px; }}
.badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .72rem; font-weight: 600; }}
.b-useful {{ background: rgba(62,219,180,.16); color: {C.MINT}; border: 1px solid {C.MINT_DARK}; }}
.b-weak {{ background: rgba(242,184,75,.14); color: {C.WARN}; border: 1px solid #a8812f; }}
.b-none {{ background: rgba(157,178,171,.12); color: {C.MUTED}; border: 1px solid {C.CHARCOAL_MID}; }}
.b-high {{ background: rgba(239,111,108,.16); color: {C.DANGER}; border: 1px solid #a64643; }}
.b-med {{ background: rgba(242,184,75,.14); color: {C.WARN}; border: 1px solid #a8812f; }}
.b-low {{ background: rgba(62,219,180,.16); color: {C.MINT}; border: 1px solid {C.MINT_DARK}; }}
.callout {{ background: rgba(242,184,75,.08); border: 1px solid rgba(242,184,75,.35); border-left: 4px solid {C.WARN};
  border-radius: 12px; padding: 12px 16px; color: {C.TEXT}; font-size: .9rem; margin: 6px 0 14px 0; }}
.callout.ok {{ background: rgba(62,219,180,.07); border-color: rgba(62,219,180,.35); border-left-color: {C.MINT}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ background: {C.CHARCOAL_LIGHT}; border-radius: 10px 10px 0 0; padding: 8px 16px; }}
.stTabs [aria-selected="true"] {{ background: {C.TEAL_DARK}; color: #fff; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 10px; border: 1px solid {C.MINT_DARK};
  background: {C.TEAL_DARK}; color: #fff; transition: all .15s ease; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background: {C.MINT_DARK}; border-color: {C.MINT}; color: {C.CHARCOAL}; }}
footer {{ visibility: hidden; }}
.app-footer {{ color: {C.MUTED}; font-size: .78rem; text-align: center; margin-top: 36px; padding-top: 12px;
  border-top: 1px solid {C.CHARCOAL_MID}; }}
@media (max-width: 760px) {{ .hero h1 {{ font-size: 1.4rem; }} .kpi .value {{ font-size: 1.25rem; }} }}
</style>
"""


def setup_plotly():
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C.TEXT, family="Inter, Segoe UI, sans-serif"),
        colorway=C.PALETTE, margin=dict(l=10, r=10, t=44, b=10),
        xaxis=dict(gridcolor=C.CHARCOAL_MID, zerolinecolor=C.CHARCOAL_MID, linecolor=C.CHARCOAL_MID),
        yaxis=dict(gridcolor=C.CHARCOAL_MID, zerolinecolor=C.CHARCOAL_MID, linecolor=C.CHARCOAL_MID),
        legend=dict(bgcolor="rgba(0,0,0,0)"), title=dict(font=dict(size=15)),
        hoverlabel=dict(bgcolor=C.CHARCOAL_LIGHT, font_color=C.TEXT),
    )
    pio.templates["mint_teal"] = t
    pio.templates.default = "mint_teal"


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def badge(kind, text):
    return f'<span class="badge b-{kind}">{text}</span>'


SIGNAL_BADGE = {"Useful": ("useful", "✓ Useful signal"), "Weak": ("weak", "◐ Weak signal"),
                "None": ("none", "○ At chance level")}


def signal_badge(level):
    k, t = SIGNAL_BADGE.get(level, ("none", "n/a"))
    return badge(k, t)


def risk_badge(level):
    return badge({"High": "high", "Medium": "med", "Low": "low"}[level], f"{level} risk")


def callout(text, ok=False):
    st.markdown(f'<div class="callout {"ok" if ok else ""}">{text}</div>', unsafe_allow_html=True)


def pred_card(col, icon, title, question, value, note="", level="None"):
    muted = " muted" if level == "None" else ""
    col.markdown(
        f'<div class="card"><h4>{icon} {title}</h4><div class="q">{question}</div>'
        f'<div class="big{muted}">{value}</div>{signal_badge(level)}'
        f'<div class="note">{note}</div></div>', unsafe_allow_html=True)


def gauge(value, title, maxv=100, suffix="%", color=C.MINT):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number=dict(suffix=suffix, font=dict(color=C.TEXT)),
        title=dict(text=title, font=dict(size=14, color=C.MUTED)),
        gauge=dict(axis=dict(range=[0, maxv], tickcolor=C.MUTED), bar=dict(color=color), bgcolor=C.CHARCOAL_MID,
                   borderwidth=0, steps=[dict(range=[0, maxv * .4], color=C.CHARCOAL_LIGHT),
                                         dict(range=[maxv * .4, maxv * .7], color=C.CHARCOAL_MID)])))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=0))
    return fig


def footer():
    st.markdown('<div class="app-footer">Customer Intelligence · Multilayer Perceptron (ANN) models built with Keras · '
                'Mint · Teal · Charcoal</div>', unsafe_allow_html=True)
