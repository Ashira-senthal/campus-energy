"""
Visual theme for the dashboard: dark glass, purple glow, pill controls.

Two layers, on purpose:

1. ``.streamlit/config.toml`` sets Streamlit's own palette, so widgets, the
   sidebar and the dataframe widget are dark without any hacking.
2. This module injects CSS for the things the theme system cannot express --
   layered background glows, glass cards, gradient pills, the hero header --
   and provides small HTML builders plus a Plotly styler.

Why the CSS is written the way it is: Streamlit's internal class names change
between releases, so styling leans on three stable hooks:

* ``.st-key-<key>`` -- Streamlit adds this class to any keyed element, so
  ``div[class*="st-key-panel_"]`` styles every panel container we create.
* ``.stTabs [data-baseweb="tab-list"]`` -- BaseWeb's tab markup, stable for years.
* plain HTML we render ourselves (KPI cards, list rows), which cannot rot.

Every colour lives in ``PALETTE`` so the whole look can be retuned in one place.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

PALETTE = {
    "bg": "#0a0610",
    "panel": "rgba(255, 255, 255, 0.045)",
    "panel_hover": "rgba(255, 255, 255, 0.07)",
    "border": "rgba(255, 255, 255, 0.09)",
    "text": "#EDE9F2",
    "muted": "#a79fb5",
    "accent": "#b98cf5",
    "accent_deep": "#6d3fa8",
    "pink": "#e9b8ff",
    "positive": "#8ee6b0",
    "negative": "#ffa3b3",
    "series": ["#c9a2ff", "#8ee6b0", "#ffa3b3", "#9db7ff", "#e9b8ff", "#f5d08c"],
}

CSS = f"""
<style>
/* ---------- page shell: layered glows on near-black plum ---------- */
.stApp {{
    background:
        radial-gradient(1100px 520px at 8% -12%, rgba(124, 77, 255, 0.20), transparent 62%),
        radial-gradient(820px 460px at 102% -6%, rgba(233, 184, 255, 0.10), transparent 58%),
        radial-gradient(900px 620px at 50% 118%, rgba(124, 77, 255, 0.10), transparent 62%),
        {PALETTE["bg"]};
    color: {PALETTE["text"]};
}}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3.2rem; max-width: 1560px; }}

/* ---------- sidebar: glass column ---------- */
section[data-testid="stSidebar"] {{ background: transparent; }}
section[data-testid="stSidebar"] > div {{
    background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.015));
    border-right: 1px solid {PALETTE["border"]};
    padding-top: 1.1rem;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ font-size: .95rem; }}

/* ---------- tabs as gradient pills ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: .5rem; background: transparent; border-bottom: none; padding: 0 0 .2rem 0;
}}
.stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {{
    background: {PALETTE["panel"]};
    border: 1px solid {PALETTE["border"]};
    border-radius: 999px;
    padding: .38rem 1.05rem;
    height: auto;
    color: {PALETTE["muted"]};
    font-size: .88rem;
    font-weight: 500;
    transition: all .18s ease;
}}
.stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"]:hover {{
    background: {PALETTE["panel_hover"]}; color: {PALETTE["text"]};
}}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
    background: linear-gradient(135deg, {PALETTE["accent_deep"]}, {PALETTE["accent"]});
    border-color: transparent;
    color: #fff;
    box-shadow: 0 8px 26px rgba(150, 100, 255, .38);
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* ---------- panels: any container keyed with panel_* becomes a glass card ---------- */
div[class*="st-key-panel_"] {{
    background: linear-gradient(160deg, rgba(255,255,255,0.065), rgba(255,255,255,0.025));
    border: 1px solid {PALETTE["border"]};
    border-radius: 22px;
    padding: 1.05rem 1.15rem;
    box-shadow: 0 18px 50px rgba(0, 0, 0, .35);
}}

/* ---------- widgets ---------- */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid {PALETTE["border"]} !important;
    border-radius: 14px !important;
}}
div[data-baseweb="tag"] {{
    background: rgba(185, 140, 245, .22) !important;
    border-radius: 999px !important;
}}
.stButton > button, .stDownloadButton > button {{
    border-radius: 999px; border: 1px solid rgba(255,255,255,.16);
    background: rgba(255,255,255,0.05); color: {PALETTE["text"]};
    padding: .42rem 1.2rem; font-size: .86rem; transition: all .18s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background: linear-gradient(135deg, {PALETTE["accent_deep"]}, {PALETTE["accent"]});
    border-color: transparent; color: #fff;
    box-shadow: 0 8px 26px rgba(150,100,255,.35);
}}
/* segmented control (range pills) */
div[data-testid="stSegmentedControl"] button,
div[data-baseweb="button-group"] button {{
    border-radius: 999px !important;
    border: 1px solid {PALETTE["border"]} !important;
    background: rgba(255,255,255,0.04) !important;
    color: {PALETTE["muted"]} !important;
    font-size: .8rem !important;
}}
div[data-testid="stSegmentedControl"] button[aria-checked="true"],
div[data-baseweb="button-group"] button[aria-pressed="true"] {{
    background: linear-gradient(135deg, {PALETTE["accent_deep"]}, {PALETTE["accent"]}) !important;
    color: #fff !important; border-color: transparent !important;
}}
/* dataframe sits inside our own card, so keep it flat and dark */
div[data-testid="stDataFrame"] {{
    border-radius: 16px; overflow: hidden; border: 1px solid {PALETTE["border"]};
}}

/* ---------- typography helpers ---------- */
.hero {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem;
         flex-wrap: wrap; margin: .2rem 0 1.15rem 0; }}
.hero h1 {{ font-size: 2.05rem; font-weight: 600; letter-spacing: -.02em; margin: 0; color: #fff; }}
.hero .accent {{ background: linear-gradient(120deg, {PALETTE["accent"]}, {PALETTE["pink"]});
                 -webkit-background-clip: text; background-clip: text; color: transparent; }}
.hero p {{ color: {PALETTE["muted"]}; font-size: .9rem; margin: .3rem 0 0 0; }}
.chip-row {{ display: flex; gap: .5rem; flex-wrap: wrap; }}
.chip {{ display: inline-flex; align-items: center; gap: .4rem;
         background: {PALETTE["panel"]}; border: 1px solid {PALETTE["border"]};
         border-radius: 999px; padding: .38rem .85rem; font-size: .78rem; color: {PALETTE["muted"]}; }}
.chip b {{ color: {PALETTE["text"]}; font-weight: 600; }}
.chip .dot {{ width: 7px; height: 7px; border-radius: 50%; background: {PALETTE["positive"]};
              box-shadow: 0 0 10px {PALETTE["positive"]}; }}

/* ---------- KPI cards ---------- */
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
             gap: .85rem; margin-bottom: .35rem; }}
.kpi {{ background: linear-gradient(160deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
        border: 1px solid {PALETTE["border"]}; border-radius: 20px; padding: 1rem 1.1rem; }}
.kpi .label {{ font-size: .76rem; color: {PALETTE["muted"]}; letter-spacing: .03em;
               text-transform: uppercase; }}
.kpi .value {{ font-size: 1.65rem; font-weight: 600; color: #fff; line-height: 1.25;
               margin-top: .3rem; }}
.kpi .note {{ font-size: .76rem; margin-top: .3rem; color: {PALETTE["muted"]}; }}
.kpi .pos {{ color: {PALETTE["positive"]}; }}
.kpi .neg {{ color: {PALETTE["negative"]}; }}

/* ---------- list rows (top anomalies) ---------- */
.row {{ display: flex; align-items: center; justify-content: space-between; gap: .7rem;
        padding: .58rem 0; border-bottom: 1px solid rgba(255,255,255,.06); }}
.row:last-child {{ border-bottom: none; }}
.row .left {{ display: flex; align-items: center; gap: .65rem; min-width: 0; }}
.row .badge {{ width: 30px; height: 30px; border-radius: 10px; display: grid; place-items: center;
               font-size: .82rem; background: rgba(185,140,245,.16); color: {PALETTE["pink"]};
               border: 1px solid rgba(185,140,245,.25); flex-shrink: 0; }}
.row .title {{ font-size: .84rem; color: {PALETTE["text"]};
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.row .sub {{ font-size: .72rem; color: {PALETTE["muted"]}; }}
.row .right {{ text-align: right; font-size: .84rem; white-space: nowrap; }}
.row .up {{ color: {PALETTE["negative"]}; font-size: .72rem; }}

/* ---------- mini building cards ---------- */
.mini-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: .7rem; }}
.mini {{ background: rgba(255,255,255,0.045); border: 1px solid {PALETTE["border"]};
         border-radius: 16px; padding: .75rem .8rem; }}
.mini .name {{ font-size: .74rem; color: {PALETTE["muted"]};
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.mini .big {{ font-size: 1.02rem; font-weight: 600; color: #fff; margin-top: .15rem; }}
.mini .tag {{ font-size: .7rem; margin-top: .2rem; color: {PALETTE["positive"]}; }}

/* ---------- promo card (gradient glow) ---------- */
.promo {{ position: relative; border-radius: 22px; padding: 1.15rem 1.2rem 1.3rem 1.2rem;
          border: 1px solid rgba(233,184,255,.14); overflow: hidden;
          background:
            radial-gradient(420px 220px at 50% 118%, rgba(185,140,245,.42), transparent 70%),
            rgba(255,255,255,0.03); }}
.promo h4 {{ margin: 0; font-size: 1rem; color: #fff; font-weight: 600; }}
.promo p {{ margin: .35rem 0 .9rem 0; font-size: .78rem; color: {PALETTE["muted"]}; line-height: 1.5; }}
.promo .pill {{ display: inline-block; padding: .42rem 1.15rem; border-radius: 999px;
                font-size: .78rem; color: #fff; border: 1px solid rgba(255,255,255,.22);
                background: linear-gradient(135deg, rgba(233,184,255,.28), rgba(124,77,255,.35)); }}

.panel-title {{ font-size: .98rem; font-weight: 600; color: #fff; margin: 0 0 .15rem 0; }}
.panel-sub {{ font-size: .76rem; color: {PALETTE["muted"]}; margin: 0 0 .55rem 0; }}
.section-gap {{ height: .9rem; }}
</style>
"""


def inject_css() -> None:
    """Add the stylesheet once per rerun."""
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, accent: str, subtitle: str, chips: list[str]) -> None:
    """Page header: gradient title, one-line subtitle, status chips on the right."""
    chip_html = "".join(f'<div class="chip">{c}</div>' for c in chips)
    st.markdown(
        f"""
        <div class="hero">
          <div>
            <h1>{title} <span class="accent">{accent}</span></h1>
            <p>{subtitle}</p>
          </div>
          <div class="chip-row">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_cards(cards: list[tuple[str, str, str, str]]) -> None:
    """Render KPI cards. Each card is (label, value, note, note_class) where
    ``note_class`` is ``pos``, ``neg`` or ``""``."""
    html = ['<div class="kpi-grid">']
    for label, value, note, note_class in cards:
        note_html = f'<div class="note {note_class}">{note}</div>' if note else ""
        html.append(
            f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>{note_html}</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def panel_title(title: str, subtitle: str | None = None) -> None:
    """Heading inside a panel container."""
    sub = f'<p class="panel-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f'<p class="panel-title">{title}</p>{sub}', unsafe_allow_html=True)


def anomaly_rows(rows: list[dict]) -> None:
    """Compact list used by the "top flagged hours" panel."""
    if not rows:
        st.markdown('<p class="panel-sub">Nothing flagged in this window.</p>', unsafe_allow_html=True)
        return
    html = []
    for row in rows:
        html.append(
            f'<div class="row">'
            f'  <div class="left">'
            f'    <div class="badge">{row["glyph"]}</div>'
            f'    <div><div class="title">{row["title"]}</div>'
            f'    <div class="sub">{row["sub"]}</div></div>'
            f'  </div>'
            f'  <div class="right">{row["value"]}'
            f'    <div class="up">{row["delta"]}</div></div>'
            f'</div>'
        )
    st.markdown("".join(html), unsafe_allow_html=True)


def building_cards(rows: list[dict]) -> None:
    """2x2 grid of building summaries, mirroring the reference's portfolio grid."""
    html = ['<div class="mini-grid">']
    for row in rows:
        html.append(
            f'<div class="mini"><div class="name">{row["name"]}</div>'
            f'<div class="big">{row["value"]}</div>'
            f'<div class="tag">{row["tag"]}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def style_figure(fig: go.Figure, height: int | None = None, legend: bool = True) -> go.Figure:
    """Apply the dark-glass Plotly look: transparent canvases, purple series,
    faint grid, pill-free legend above the plot."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family='-apple-system, "Segoe UI", Roboto, sans-serif',
            color=PALETTE["muted"],
            size=12,
        ),
        colorway=PALETTE["series"],
        margin=dict(l=8, r=8, t=14, b=8),
        hoverlabel=dict(
            bgcolor="#1b1324",
            bordercolor="rgba(255,255,255,.18)",
            font=dict(color=PALETTE["text"], size=12),
        ),
        legend=dict(
            orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)",
            font=dict(color=PALETTE["muted"], size=11),
        ) if legend else dict(),
        showlegend=legend,
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        linecolor="rgba(255,255,255,.12)", tickcolor="rgba(255,255,255,.25)",
        tickfont=dict(size=11, color=PALETTE["muted"]),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,.07)", zeroline=False,
        tickfont=dict(size=11, color=PALETTE["muted"]),
    )
    if height:
        fig.update_layout(height=height)
    return fig
