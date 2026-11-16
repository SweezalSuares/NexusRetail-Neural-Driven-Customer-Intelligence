"""Reusable UI pieces: CSS theme, KPI cards, badges and the Plotly template."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src import config as C


CSS = f"""
<style>
html, body, [class*="css"] {{
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    color: {C.TEXT};
}}

.stApp {{
    background: {C.CREAMY_BACKGROUND};
    color: {C.TEXT};
}}

.block-container {{
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}}

section[data-testid="stSidebar"] {{
    background: {C.COCOA_BROWN};
    color: {C.SIDEBAR_SECONDARY};
    border-right: 1px solid {C.BORDER};
}}

section[data-testid="stSidebar"] h2 {{
    color: {C.CHAMPAGNE_GOLD};
}}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: {C.SIDEBAR_SECONDARY};
}}

section[data-testid="stSidebar"] [role="radiogroup"] label {{
    color: {C.SIDEBAR_LINK};
}}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    color: {C.SIDEBAR_SECONDARY};
}}

.sidebar-user {{
    color: {C.SIDEBAR_USER};
    font-weight: 650;
    margin-top: 12px;
}}

h1, h2, h3 {{
    color: {C.TEXT};
    letter-spacing: -0.02em;
}}

a {{
    color: {C.DEEP_ROSE};
}}

.hero {{
    background: {C.CARD_BACKGROUND};
    border: 1px solid {C.CHAMPAGNE_GOLD};
    border-radius: 18px;
    padding: 28px 30px;
    margin-bottom: 20px;
    box-shadow: 0 8px 28px rgba(59,42,36,.07);
}}

.hero h1 {{
    margin: 0;
    font-size: 1.9rem;
    color: {C.TEXT};
}}

.hero p {{
    margin-top: 8px;
    color: {C.SOFT_COCOA};
}}

.kpi {{
    background: {C.CARD_BACKGROUND};
    border: 1px solid {C.BORDER};
    border-left: 4px solid {C.CHAMPAGNE_GOLD};
    border-radius: 14px;
    padding: 16px;
    height: 100%;
    transition: .2s ease;
}}

.kpi:hover {{
    transform: translateY(-3px);
    border-left-color: {C.CHAMPAGNE_GOLD};
    box-shadow: 0 8px 20px rgba(59,42,36,.10);
}}

.kpi .label {{
    color: {C.SOFT_COCOA};
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: .06em;
}}

.kpi .value {{
    color: {C.DEEP_ROSE};
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 5px;
}}

.kpi .sub {{
    color: {C.SOFT_COCOA};
    font-size: .8rem;
    margin-top: 4px;
}}

.card {{
    background: {C.CARD_BACKGROUND};
    border: 1px solid {C.BORDER};
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
    height: 100%;
}}

.card h4 {{
    color: {C.COCOA_BROWN};
    font-size: .95rem;
}}

.card .big {{
    font-size: 1.55rem;
    font-weight: 700;
    color: {C.DEEP_ROSE};
}}

.card .big.muted {{
    color: {C.SOFT_COCOA};
}}

.card .q, .card .note {{
    color: {C.SOFT_COCOA};
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
    background: {C.SAGE_SOFT};
    color: #3F5F48;
    border: 1px solid {C.SAGE};
}}

.b-weak, .b-med {{
    background: {C.GOLD_SOFT};
    color: #7A5A14;
    border: 1px solid {C.CHAMPAGNE_GOLD};
}}

.b-none {{
    background: {C.PANEL_BACKGROUND};
    color: {C.SOFT_COCOA};
    border: 1px solid {C.BORDER};
}}

.b-high {{
    background: {C.ROSE_SOFT};
    color: #8A3A47;
    border: 1px solid {C.MUTED_ROSE};
}}

.callout {{
    background: {C.GOLD_SOFT};
    border: 1px solid {C.CHAMPAGNE_GOLD};
    border-left: 4px solid {C.CHAMPAGNE_GOLD};
    border-radius: 12px;
    padding: 13px 16px;
    color: {C.COCOA_BROWN};
    margin: 8px 0 16px;
}}

.callout.ok {{
    border-left-color: {C.SAGE};
    background: {C.SAGE_SOFT};
}}

/* LOGIN AND SIGNUP NAVIGATION TABS */

.stTabs [role="tab"] {{
    background-color: #FAF5EF !important;
    color: #79604D !important;
    border: 1px solid #D8B8AD !important;
    border-radius: 10px 10px 0 0 !important;
    box-shadow: 0 2px 6px rgba(59,42,36,.08);
    }}

.stTabs [role="tab"] *,
.stTabs [role="tab"] p {{
    color: #79604D !important;
    -webkit-text-fill-color: #79604D !important;
}}

.stTabs [role="tab"][aria-selected="true"] {{
    background-color: #79604D !important;
    color: #FAF5EF !important;
    border-color: #79604D !important;
}}

.stTabs [role="tab"][aria-selected="true"] *,
.stTabs [role="tab"][aria-selected="true"] p {{
    color: #FAF5EF !important;
    -webkit-text-fill-color: #FAF5EF !important;
}}

.stButton > button,
.stDownloadButton > button {{
    background: {C.DEEP_ROSE};
    color: {C.SOFT_IVORY};
    border: 1px solid {C.DEEP_ROSE};
    border-radius: 10px;
    transition: .2s ease;
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
    background: {C.COCOA_BROWN};
    border-color: {C.CHAMPAGNE_GOLD};
    color: {C.SOFT_IVORY};
}}

.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div,
.stTextArea textarea, .stDateInput input {{
    background: {C.PANEL_BACKGROUND};
    color: {C.TEXT};
    border-color: {C.BORDER};
}}

.auth-shell {{
    max-width: 760px;
    margin: 4vh auto 0;
    padding: 28px;
    background: {C.COCOA_BROWN};
    color: {C.HERO_TEXT};
    border: 1px solid {C.CHAMPAGNE_GOLD};
    border-radius: 20px;
    box-shadow: 0 12px 36px rgba(59,42,36,.14);
}}

.auth-brand {{
    color: {C.CHAMPAGNE_GOLD};
    font-size: 1.8rem;
    font-weight: 750;
    letter-spacing: -.03em;
}}

.auth-shell p {{
    color: {C.HERO_TEXT};
}}

footer {{
    visibility: hidden;
}}

.app-footer {{
    color: {C.SOFT_COCOA};
    font-size: .78rem;
    text-align: center;
    margin-top: 36px;
    padding-top: 14px;
    border-top: 1px solid {C.BORDER};
}}

@media (max-width: 760px) {{
    .hero h1 {{ font-size: 1.4rem; }}
    .kpi .value {{ font-size: 1.25rem; }}
}}
.stTextInput label,
.stTextInput label p,
.stTextInput label span {{
    color: {C.COCOA_BROWN} !important;
}}
</style>
"""

def setup_plotly():
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C.TEXT, family="Inter, Segoe UI, sans-serif", size=13),
        colorway=C.PALETTE, margin=dict(l=10, r=10, t=44, b=10),
        xaxis=dict(gridcolor="#EFE2D6", zerolinecolor="#EFE2D6", linecolor=C.BORDER,
                   tickfont=dict(color=C.TEXT, size=13), title_font=dict(color=C.SOFT_COCOA, size=12)),
        yaxis=dict(gridcolor="#EFE2D6", zerolinecolor="#EFE2D6", linecolor=C.BORDER,
                   tickfont=dict(color=C.TEXT, size=13), title_font=dict(color=C.SOFT_COCOA, size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C.TEXT, size=13)),
        title=dict(font=dict(size=17, family="Fraunces, Georgia, serif", color=C.TEXT),
                   x=0, xanchor="left"),
        hoverlabel=dict(bgcolor=C.CARD_BACKGROUND, bordercolor=C.BORDER, font_color=C.TEXT),
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
