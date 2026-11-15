"""Reusable UI pieces: CSS theme, KPI cards, badges and the Plotly template."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src import config as C


CSS = f"""
<style>
html, body, [class*="css"] {{
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}}

.stApp {{
    background: {C.DUSKY_BACKGROUND};
    color: {C.SOFT_IVORY};
}}

.block-container {{
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {C.CHARCOAL_LIGHT}, {C.CHARCOAL});
    border-right: 1px solid {C.CHARCOAL_MID};
}}

h1, h2, h3 {{
    color: {C.SOFT_IVORY};
    letter-spacing: -0.02em;
}}

.hero {{
    background: linear-gradient(
        120deg,
        {C.CHARCOAL_LIGHT},
        {C.TEAL_DARK},
        {C.CHARCOAL}
    );
    border: 1px solid {C.CHARCOAL_MID};
    border-radius: 18px;
    padding: 28px 30px;
    margin-bottom: 20px;
    box-shadow: 0 8px 28px rgba(0,0,0,.18);
}}

.hero h1 {{
    margin: 0;
    font-size: 1.9rem;
    color: {C.TEXT};
}}

.hero p {{
    margin-top: 8px;
    color: {C.SOFT_IVORY};
}}

.kpi {{
    background: {C.CHARCOAL_LIGHT};
    border: 1px solid {C.CHARCOAL_MID};
    border-left: 4px solid {C.TEAL};
    border-radius: 14px;
    padding: 16px;
    height: 100%;
    transition: .2s ease;
}}

.kpi:hover {{
    transform: translateY(-3px);
    border-left-color: {C.CHAMPAGNE_GOLD};
    box-shadow: 0 8px 20px rgba(0,0,0,.2);
}}

.kpi .label {{
    color: {C.MUTED};
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: .06em;
}}

.kpi .value {{
    color: {C.TEXT};
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 5px;
}}

.kpi .sub {{
    color: {C.OLIVE};
    font-size: .8rem;
    margin-top: 4px;
}}

.card {{
    background: {C.CHARCOAL_LIGHT};
    border: 1px solid {C.CHARCOAL_MID};
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
    height: 100%;
}}

.card h4 {{
    color: {C.SOFT_IVORY};
    font-size: .95rem;
}}

.card .big {{
    font-size: 1.55rem;
    font-weight: 700;
    color: {C.MINT};
}}

.card .big.muted {{
    color: {C.MUTED};
}}

.card .q, .card .note {{
    color: {C.MUTED};
    font-size: .8rem;
}}

.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: .72rem;
    font-weight: 600;
}}

.b-useful, .b-low {{
    background: rgba(79,143,131,.16);
    color: {C.MINT};
    border: 1px solid {C.TEAL};
}}

.b-weak, .b-med {{
    background: rgba(217,185,140,.12);
    color: {C.CHAMPAGNE_GOLD};
    border: 1px solid {C.CHAMPAGNE_GOLD};
}}

.b-none {{
    background: rgba(185,181,169,.10);
    color: {C.MUTED};
    border: 1px solid {C.CHARCOAL_MID};
}}

.b-high {{
    background: rgba(201,130,130,.14);
    color: {C.DANGER};
    border: 1px solid {C.DANGER};
}}

.callout {{
    background: rgba(217,185,140,.08);
    border: 1px solid rgba(217,185,140,.3);
    border-left: 4px solid {C.CHAMPAGNE_GOLD};
    border-radius: 12px;
    padding: 13px 16px;
    color: {C.SOFT_IVORY};
    margin: 8px 0 16px;
}}

.callout.ok {{
    border-left-color: {C.TEAL};
    background: rgba(79,143,131,.08);
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}

.stTabs [data-baseweb="tab"] {{
    background: {C.CHARCOAL_LIGHT};
    border-radius: 9px 9px 0 0;
    padding: 9px 16px;
    color: {C.MUTED};
}}

.stTabs [aria-selected="true"] {{
    background: {C.TEAL_DARK};
    color: white;
}}

.stButton > button,
.stDownloadButton > button {{
    background: {C.TEAL_DARK};
    color: {C.TEXT};
    border: 1px solid {C.TEAL};
    border-radius: 10px;
    transition: .2s ease;
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
    background: {C.TEAL};
    border-color: {C.CHAMPAGNE_GOLD};
    color: white;
}}

footer {{
    visibility: hidden;
}}

.app-footer {{
    color: {C.MUTED};
    font-size: .78rem;
    text-align: center;
    margin-top: 36px;
    padding-top: 14px;
    border-top: 1px solid {C.CHARCOAL_MID};
}}

@media (max-width: 760px) {{
    .hero h1 {{ font-size: 1.4rem; }}
    .kpi .value {{ font-size: 1.25rem; }}
}}
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
    pio.templates["nexus_retail"] = t
    pio.templates.default = "nexus_retail"


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
    st.markdown('<div class="app-footer">NexusRetail · Neural Driven Customer Intelligence · ANN models built with Keras · '
                'Rosewood · Champagne gold · Cocoa brown · Blush beige · Soft ivory · Muted rose</div>',
                unsafe_allow_html=True)
