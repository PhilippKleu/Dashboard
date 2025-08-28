# tech_decision_tool.py

# === Konfiguration und Imports ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import ceil
from collections import defaultdict
import matplotlib.lines as mlines
import matplotlib as mpl
from scipy.stats import gaussian_kde
import re
from sklearn.cluster import KMeans
import seaborn as sns
from io import BytesIO
from zipfile import ZipFile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.graph_objs.layout import XAxis, YAxis
import requests
import base64, requests

st.set_page_config(layout="wide")

@st.cache_data(show_spinner=False)
def load_font_b64(raw_url: str) -> str:
    r = requests.get(raw_url, timeout=10)
    r.raise_for_status()
    return base64.b64encode(r.content).decode()

FONT_URL = "https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/Montserrat/static/Montserrat-Regular.ttf"
font_b64 = load_font_b64(FONT_URL)

st.markdown(f"""
<style>
/* 1) Montserrat einbetten – kein globales Anwenden */
@font-face {{
  font-family: 'Montserrat';
  src: url(data:font/ttf;base64,{font_b64}) format('truetype');
  font-weight: 400;
  font-style: normal;
  font-display: swap;
}}

/* 2) Icon-Schriften schützen */
.material-icons,
.material-icons-outlined,
.material-icons-round {{
  font-family: 'Material Icons' !important;
  font-feature-settings: 'liga' 1 !important;
  font-variant-ligatures: normal !important;
}}
.material-symbols-outlined,
.material-symbols-rounded {{
  font-family: 'Material Symbols Outlined' !important;
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
  font-feature-settings: 'liga' 1 !important;
  font-variant-ligatures: normal !important;
}}

/* 3) Hauptinhalt: Überschriften + Markdown-Text */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *:not(i):not(.material-icons):not(.material-symbols-outlined) {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}}

/* 4) Sidebar-Text */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}}

/* 5) Header-Text außer Toggle */
[data-testid="stHeader"] *:not([data-testid="collapsedControl"] *):not(.material-icons):not(.material-symbols-outlined) {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}}

/* 6) Widgets: Labels, Inputs, Buttons */
label,
.stCheckbox, .stRadio, .stSelectbox, .stMultiSelect, .stSlider, .stDateInput, .stTimeInput {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}}
input[type="text"], input[type="number"], input[type="search"], input[type="email"], input[type="password"],
textarea, select,
.stTextInput input, .stNumberInput input, .stTextArea textarea {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}}
.stButton > button, .stDownloadButton > button {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
  background-color: #C0C6D2;
  color: black;
  border: none;
  padding: 0.5em 1.2em;
  border-radius: 8px;
  font-weight: 500;
}}

/* 6a) Dropdown-Feld (sichtbares Input) */
[data-testid="stSelectbox"] *:not(.material-icons):not(.material-symbols-outlined),
[data-testid="stMultiSelect"] *:not(.material-icons):not(.material-symbols-outlined) {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}}
/* MultiSelect ausgewählte Tags */
[data-baseweb="tag"] {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}}

/* 6b) Portal-Menü der Dropdowns */
[data-baseweb="layer"] *:not(.material-icons):not(.material-symbols-outlined),
[data-baseweb="popover"] *:not(.material-icons):not(.material-symbols-outlined),
[data-baseweb="menu"] *:not(.material-icons):not(.material-symbols-outlined),
[role="listbox"] *:not(.material-icons):not(.material-symbols-outlined),
[role="option"]:not(.material-icons):not(.material-symbols-outlined),
[role="option"] *:not(.material-icons):not(.material-symbols-outlined) {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
  font-variant-ligatures: normal !important;
}}

/* 6c) Dropdown-/Multiselect-Optionsliste */
[data-baseweb="layer"] ul[role="listbox"],
[data-baseweb="layer"] ul[role="menu"] {{
  max-height: 200px !important;
  height: auto !important;
  overflow-y: auto !important;
  overscroll-behavior: contain;
  -webkit-overflow-scrolling: touch;
}}
[data-baseweb="layer"] [data-baseweb="menu"] {{
  max-height: 200px !important;
  height: auto !important;
  overflow-y: auto !important;
}}
[data-baseweb="layer"] [data-baseweb="popover"] {{
  max-height: 200px !important;
  height: auto !important;
  overflow: visible !important;
}}

/* Compact MultiSelect input */
[data-testid="stMultiSelect"] > div {{
  min-height: 20px !important;
}}
[data-testid="stMultiSelect"] div[data-baseweb="value-container"] {{
  max-height: 48px !important;
  overflow-y: auto !important;
}}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {{
  transform: scale(1);
  margin: 1px 2px !important;
}}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span {{
  font-size: 12px !important;
}}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {{
  padding-top: 2px !important;
  padding-bottom: 2px !important;
}}

/* 7) Tabs/Expander/Metrics */
[data-testid="stSubheader"] *,
[data-testid="stCaption"] *,
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"],
[data-testid="stMetricValue"],
[data-testid="stExpander"] summary,
[data-testid="stTabs"] button p {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}}

/* Tabellen & DataFrames */
[data-testid="stTable"] table,
[data-testid="stTable"] th,
[data-testid="stTable"] td {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
}}
[data-testid="stDataFrame"] *:not(.material-icons):not(.material-symbols-outlined),
[data-testid="stDataFrame"] div[role="gridcell"],
[data-testid="stDataFrame"] div[role="columnheader"],
[data-testid="stDataFrame"] div[role="row"] {{
  font-family: 'Montserrat', system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
  font-variant-ligatures: normal !important;
}}

/* 9) Hintergrund & Hover
   — Light = Default (#f4f4f4)
   — Dark: überschreibe HINTERGRUND auf allen relevanten Containern */
body, .stApp {{ background-color: #f4f4f4; }}

@media (prefers-color-scheme: dark){{
  html, body,
  .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stSidebar"],
  [data-testid="stHeader"],
  .main,
  .block-container {{
    background: #0f1116 !important;
    background-color: #0f1116 !important;
    color-scheme: dark;
  }}
  /* evtl. Overlay/Gradient entfernen */
  [data-testid="stAppViewContainer"]::before {{
    content: none !important;
    background: none !important;
    background-image: none !important;
  }}
}}
/* greift zusätzlich, falls Streamlit explizit ein Dark-Theme setzt */
html[data-theme="dark"] body,
html[data-theme="dark"] .stApp,
html[data-theme="dark"] [data-testid="stAppViewContainer"],
html[data-theme="dark"] [data-testid="stSidebar"],
html[data-theme="dark"] [data-testid="stHeader"],
html[data-theme="dark"] .main,
html[data-theme="dark"] .block-container {{
  background: #0f1116 !important;
  background-color: #0f1116 !important;
  color-scheme: dark;
}}

div[data-testid="stFileUploader"]:hover,
div[data-testid^="stSelectbox"]:hover,
div[data-testid^="stMultiSelect"]:hover,
div[data-testid^="stNumberInput"]:hover,
div[data-testid^="stRadio"]:hover,
div[data-testid^="stSlider"]:hover {{
  transform: scale(1.01);
  transition: transform 0.2s ease;
}}

/* === HEADER-TOGGLE: Inhalt ausblenden & eigene Pfeile === */
[data-testid="collapsedControl"],
[data-testid="stHeader"] [data-testid="collapsedControl"],
[data-testid="stHeader"] button[aria-label*="sidebar" i],
[data-testid="stHeader"] button[title*="sidebar" i] {{
  position: relative !important;
}}
[data-testid="collapsedControl"] *,
[data-testid="stHeader"] [data-testid="collapsedControl"] *,
[data-testid="stHeader"] button[aria-label*="sidebar" i] *,
[data-testid="stHeader"] button[title*="sidebar" i] * {{
  opacity: 0 !important;
}}
[data-testid="collapsedControl"]::after,
[data-testid="stHeader"] [data-testid="collapsedControl"]::after,
[data-testid="stHeader"] button[aria-label*="sidebar" i]::after,
[data-testid="stHeader"] button[title*="sidebar" i]::after {{
  content: "❮❮";
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  font-family: "Segoe UI", system-ui, sans-serif;
  font-size: 22px;
  line-height: 1;
}}
[data-testid="stSidebar"] [data-testid="collapsedControl"]::after {{
  content: "❯❯";
}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root{
  --ms-box-h: 100px;   /* Höhe der MultiSelect-Box */
}
[data-testid="stMultiSelect"] [data-baseweb="select"],
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[aria-haspopup="listbox"]{
  min-height: var(--ms-box-h) !important;
  height: var(--ms-box-h) !important;
}
[data-testid="stMultiSelect"] div[data-baseweb="value-container"]{
  max-height: calc(var(--ms-box-h) - 6px) !important;
  overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ========= Dark Mode: nur Farben überschreiben ========= */
@media (prefers-color-scheme: dark){

  /* Hintergrund – auf allen Hauptcontainern */
  html, body, .stApp,
  .st-emotion-cache-18ni7ap,  /* App-Haupt-Wrapper */
  .block-container { 
    background-color: #0f1116 !important;
    color-scheme: dark;
  }

  /* Textfarben (Hauptbereich) */
  .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
  [data-testid="stMarkdownContainer"],
  [data-testid="stMarkdownContainer"] *:not(i):not(.material-icons):not(.material-symbols-outlined){
    color: #e6e6e6 !important;
  }

  /* Sidebar-Texte */
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] h4,
  [data-testid="stSidebar"] h5,
  [data-testid="stSidebar"] h6,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] small{
    color: #e6e6e6 !important;
  }

  /* Buttons */
  .stButton > button, .stDownloadButton > button{
    background-color: #3a3f4b !important;
    color: #ffffff !important;
  }

  /* Inputs */
  input[type="text"], input[type="number"], input[type="search"],
  input[type="email"], input[type="password"],
  textarea, select,
  .stTextInput input, .stNumberInput input, .stTextArea textarea{
    background-color: #1b202b !important;
    color: #e6e6e6 !important;
    border-color: #2e3441 !important;
  }

  /* Dropdown/Popover */
  [data-baseweb="layer"] [data-baseweb="menu"],
  [data-baseweb="layer"] [data-baseweb="popover"]{
    background-color: #161a22 !important;
    border: 1px solid #2a2f3a !important;
  }
  [data-baseweb="layer"] *:not(.material-icons):not(.material-symbols-outlined){
    color: #e6e6e6 !important;
  }

  /* Tabellen & DataFrames */
  [data-testid="stTable"] table,
  [data-testid="stTable"] th,
  [data-testid="stTable"] td{
    background-color: #161a22 !important;
    color: #e6e6e6 !important;
    border-color: #2a2f3a !important;
  }
  [data-testid="stDataFrame"] div[role="gridcell"],
  [data-testid="stDataFrame"] div[role="columnheader"]{
    color: #e6e6e6 !important;
  }

  /* Header-Toggle-Pfeile */
  [data-testid="collapsedControl"]::after,
  [data-testid="stHeader"] [data-testid="collapsedControl"]::after,
  [data-testid="stHeader"] button[aria-label*="sidebar" i]::after,
  [data-testid="stHeader"] button[title*="sidebar" i]::after{
    color: #e6e6e6 !important;
  }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ========= Dark Mode: nur Farbe/Background überschreiben ========= */
/* Greift, wenn System-Dark-Mode aktiv ist ODER Streamlit-Theme auf "Dark" steht */
@media (prefers-color-scheme: dark){
  html, body,
  .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stSidebar"],
  [data-testid="stHeader"],
  .main,
  .block-container{
    background: #0f1116 !important;
    background-color: #0f1116 !important;
    color-scheme: dark;
  }

  /* Falls Streamlit einen Gradient als Overlay nutzt */
  [data-testid="stAppViewContainer"]::before{
    content: none !important;
    background: none !important;
    background-image: none !important;
  }

  /* Sicherheitshalber alle möglichen Background-Images kappen */
  html, body, .stApp, [data-testid="stAppViewContainer"], .main, .block-container{
    background-image: none !important;
  }

  /* Texte */
  .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
  [data-testid="stMarkdownContainer"],
  [data-testid="stMarkdownContainer"] *:not(i):not(.material-icons):not(.material-symbols-outlined){
    color: #e6e6e6 !important;
  }
  [data-testid="stSidebar"] *{
    color: #e6e6e6 !important;
  }

  /* Buttons */
  .stButton > button, .stDownloadButton > button{
    background-color: #3a3f4b !important;
    color: #ffffff !important;
    border-color: transparent !important;
  }

  /* Inputs */
  input[type="text"], input[type="number"], input[type="search"],
  input[type="email"], input[type="password"],
  textarea, select,
  .stTextInput input, .stNumberInput input, .stTextArea textarea{
    background-color: #1b202b !important;
    color: #e6e6e6 !important;
    border-color: #2e3441 !important;
  }

  /* Dropdown/Popover */
  [data-baseweb="layer"] [data-baseweb="menu"],
  [data-baseweb="layer"] [data-baseweb="popover"]{
    background-color: #161a22 !important;
    border: 1px solid #2a2f3a !important;
  }
  [data-baseweb="layer"] *:not(.material-icons):not(.material-symbols-outlined){
    color: #e6e6e6 !important;
  }

  /* Tabellen & DataFrames */
  [data-testid="stTable"] table,
  [data-testid="stTable"] th,
  [data-testid="stTable"] td{
    background-color: #161a22 !important;
    color: #e6e6e6 !important;
    border-color: #2a2f3a !important;
  }
  [data-testid="stDataFrame"] div[role="gridcell"],
  [data-testid="stDataFrame"] div[role="columnheader"]{
    color: #e6e6e6 !important;
  }

  /* Header-Toggle-Pfeile */
  [data-testid="collapsedControl"]::after,
  [data-testid="stHeader"] [data-testid="collapsedControl"]::after,
  [data-testid="stHeader"] button[aria-label*="sidebar" i]::after,
  [data-testid="stHeader"] button[title*="sidebar" i]::after{
    color: #e6e6e6 !important;
  }
}

/* Zusätzlich: greife, wenn Streamlit explizit das Theme auf "dark" setzt */
html[data-theme="dark"] body,
html[data-theme="dark"] .stApp,
html[data-theme="dark"] [data-testid="stAppViewContainer"],
html[data-theme="dark"] [data-testid="stSidebar"],
html[data-theme="dark"] [data-testid="stHeader"],
html[data-theme="dark"] .main,
html[data-theme="dark"] .block-container{
  background: #0f1116 !important;
  background-color: #0f1116 !important;
  color-scheme: dark;
}
html[data-theme="dark"] [data-testid="stAppViewContainer"]::before{
  content: none !important;
  background: none !important;
  background-image: none !important;
}
html[data-theme="dark"] .stApp h1,
html[data-theme="dark"] .stApp h2,
html[data-theme="dark"] .stApp h3,
html[data-theme="dark"] .stApp h4,
html[data-theme="dark"] .stApp h5,
html[data-theme="dark"] .stApp h6,
html[data-theme="dark"] [data-testid="stMarkdownContainer"],
html[data-theme="dark"] [data-testid="stMarkdownContainer"] *:not(i):not(.material-icons):not(.material-symbols-outlined){
  color: #e6e6e6 !important;
}
html[data-theme="dark"] [data-testid="stSidebar"] *{
  color: #e6e6e6 !important;
}
</style>
""", unsafe_allow_html=True)

# ==== Row-based sizing (Plotly & Matplotlib) ====
DEFAULT_ROW_HEIGHT_PX = 320   # feste Pixelhöhe je Plot-Zeile in Plotly
DEFAULT_HSPACE_FRAC   = 0.1  # 0..1
DEFAULT_VSPACE_FRAC   = 0.1  # 0..1

DEFAULT_ROW_HEIGHT_IN = 3.2   # feste Zollhöhe je Plot-Zeile in Matplotlib
DEFAULT_COL_WIDTH_IN  = 5.5   # feste Zollbreite je Plot-Spalte in Matplotlib

DEFAULT_TOP_MARGIN    = 80
DEFAULT_BOTTOM_MARGIN = 70

def compute_plotly_grid(
    n_plots:int,
    n_cols:int,
    row_height_px:int=DEFAULT_ROW_HEIGHT_PX,
    hspace_frac:float=DEFAULT_HSPACE_FRAC,
    vspace_px:int=12,                      # << NEU: vertikal in Pixel
    top_margin_px:int=DEFAULT_TOP_MARGIN,  # konsistent zur Layout-Margin
    bottom_margin_px:int=DEFAULT_BOTTOM_MARGIN
):
    import math
    n_cols = max(1, int(n_cols))
    n_rows = int(math.ceil(n_plots / n_cols))

    # Gesamt-Höhe in Pixel: Zeilen * Höhe + (Zeilen-1) * Lücke + äußere Margins
    total_height = top_margin_px + bottom_margin_px + n_rows * row_height_px + max(0, n_rows-1) * vspace_px

    # Plotly: spacing ist Anteil der **Grid-Höhe** (ohne äußere Margins!)
    grid_height = max(1, total_height - top_margin_px - bottom_margin_px)
    vertical_spacing = 0.0 if n_rows <= 1 else (vspace_px / grid_height)

    # Plotly fordert: spacing < 1/n_rows (und bei 1 Zeile -> 0)
    if n_rows > 1:
        vertical_spacing = min(vertical_spacing, (1.0 / n_rows) - 1e-6)

    # Horizontal als Anteil (Breite ist responsive, Pixel kennt Streamlit hier nicht)
    horizontal_spacing = 0.0 if n_cols <= 1 else min(max(0.0, float(hspace_frac)), (1.0 / n_cols) - 1e-6)

    return n_rows, n_cols, horizontal_spacing, vertical_spacing, int(total_height), int(top_margin_px), int(bottom_margin_px)

def compute_mpl_figsize(n_plots:int, n_cols:int,
                        col_w_in:float=DEFAULT_COL_WIDTH_IN,
                        row_h_in:float=DEFAULT_ROW_HEIGHT_IN):
    import math
    n_cols = max(1, int(n_cols))
    n_rows = int(math.ceil(n_plots / n_cols))
    width_in  = n_cols * col_w_in
    height_in = n_rows * row_h_in
    return (width_in, height_in), n_rows, n_cols
def add_subplot_borders(fig, color="#C0C6D2", width=1.5, dash=None, pad=0.0, above=True, only_used=False):
    """
    Zeichnet für jeden Subplot (jede Achskombination) einen Rahmen.
    - color: Linienfarbe
    - width: Linienstärke
    - dash: 'dash', 'dot', 'dashdot' oder None
    - pad: in Domain-Einheiten (0..0.5); z.B. 0.01 = kleiner Innenabstand
    - above: True => Rahmen über Traces; False => darunter
    - only_used: nur Subplots mit Daten (Traces) umranden
    """
    # vorhandene Shapes übernehmen
    shapes = list(fig.layout.shapes) if fig.layout.shapes else []

    # vorhandene Achs-IDs finden (xaxis, xaxis2, ...)
    layout_dict = fig.to_dict().get("layout", {})
    x_ids = set()
    y_ids = set()
    for k in layout_dict.keys():
        if k.startswith("xaxis"):
            suffix = k[5:]
            x_ids.add(1 if suffix == "" else int(suffix))
        if k.startswith("yaxis"):
            suffix = k[5:]
            y_ids.add(1 if suffix == "" else int(suffix))

    # optional: nur Achsen verwenden, die auch Traces haben
    used_ids = set()
    if only_used:
        for tr in fig.data:
            # 'xaxis'/'yaxis' sind z.B. 'x', 'x2', ...
            xa = getattr(tr, "xaxis", None) or "x"
            ya = getattr(tr, "yaxis", None) or "y"
            xi = 1 if xa == "x" else int(xa[1:])
            yi = 1 if ya == "y" else int(ya[1:])
            used_ids.add((xi, yi))

    # Rahmen je Achspaar erzeugen
    for xi in sorted(x_ids):
        for yi in sorted(y_ids):
            # wenn only_used aktiv ist: nur tatsächliche genutzte Paare umranden
            if only_used and (xi, yi) not in used_ids:
                continue

            xref = "x domain" if xi == 1 else f"x{xi} domain"
            yref = "y domain" if yi == 1 else f"y{yi} domain"

            shapes.append(dict(
                type="rect",
                xref=xref, yref=yref,
                x0=0.0 + pad, x1=1.0 - pad,
                y0=0.0 + pad, y1=1.0 - pad,
                line=dict(color=color, width=width, dash=dash) if dash else dict(color=color, width=width),
                fillcolor="rgba(0,0,0,0)",
                layer="above" if above else "below"
            ))

    fig.update_layout(shapes=shapes)
    return fig

# Export-Flag + Bin
st.session_state.setdefault("capture_exports", False)
st.session_state.setdefault("export_bin", {})  # path -> (bytes, mime)

def _export_put(path: str, data: bytes, mime: str):
    st.session_state["export_bin"][path] = (data, mime)

def emit_plotly(fig, tag: str):
    """Rendern ODER Capturen – abhängig von st.session_state['capture_exports']."""
    if fig is None:
        return
    if st.session_state.get("capture_exports"):
        try:
            data = fig.to_image(format="png", scale=2)  # Kaleido benötigt
            _export_put(f"plots/{tag}.png", data, "image/png")
        except Exception:
            html = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
            _export_put(f"plots/{tag}.html", html, "text/html")
        # Kein Rendern, nur speichern
    else:
        st.plotly_chart(fig, use_container_width=True)

def emit_mpl(fig, tag: str, dpi=200):
    if fig is None:
        return
    if st.session_state.get("capture_exports"):
        b = BytesIO()
        fig.savefig(b, format="png", dpi=dpi, bbox_inches="tight")
        _export_put(f"plots/{tag}.png", b.getvalue(), "image/png")
        plt.close(fig); del fig; gc.collect()
    else:
        st.pyplot(fig, use_container_width=True)

def make_current_excel_bytes():
    excel_buf = BytesIO()
    base_idx = current_indices if 'current_indices' in locals() else tech_data.index
    with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
        frames = [tech_data.loc[base_idx]]
        if MAA_PREFIX == "VALUE_":
            inst_cols = [c for c in vertex_df.columns if c.startswith(INSTALLED_CAPACITY_PREFIX)]
            if inst_cols:
                frames.append(vertex_df.loc[base_idx, inst_cols])
        if additional_cols:
            frames.append(vertex_df.loc[base_idx, additional_cols])
        pd.concat(frames, axis=1).to_excel(writer, index=False, sheet_name="Original_Vertices")
        try:
            if (st.session_state.get('show_convex')
                and 'filtered_convex_data' in locals()
                and not filtered_convex_data.empty):
                conv_frames = [filtered_convex_data.reset_index(drop=True)]
                if additional_cols and 'filtered_convex_additional' in locals() and not filtered_convex_additional.empty:
                    conv_frames.append(filtered_convex_additional[additional_cols].reset_index(drop=True))
                convex_all = pd.concat(conv_frames, axis=1)
                convex_all = convex_all.loc[:, ~convex_all.columns.duplicated()]
                convex_all.to_excel(writer, index=False, sheet_name="Convex_Combinations")
        except Exception:
            pass
    excel_buf.seek(0)
    return excel_buf.getvalue()

# === Montserrat für Matplotlib registrieren ===
from matplotlib import font_manager as fm
import tempfile, base64, os, matplotlib as mpl

# Font-Datei aus dem vorhandenen base64-String schreiben
_tmp_dir = tempfile.gettempdir()
MONTSERRAT_TTF = os.path.join(_tmp_dir, "Montserrat-Regular.ttf")
try:
    if not os.path.exists(MONTSERRAT_TTF):
        with open(MONTSERRAT_TTF, "wb") as _f:
            _f.write(base64.b64decode(font_b64))

    # Bei Matplotlib registrieren und als Standard setzen
    fm.fontManager.addfont(MONTSERRAT_TTF)
    mpl.rcParams["font.family"] = "Montserrat"
    mpl.rcParams["font.sans-serif"] = ["Montserrat"]
    mpl.rcParams["axes.unicode_minus"] = False  # minus korrekt mit Montserrat
except Exception as e:
    st.warning(f"⚠️ Konnte Montserrat nicht für Matplotlib setzen: {e}")

# === Montserrat als Standard für Plotly setzen ===
import plotly.io as pio

# Bestehendes Template kopieren und Überschrift/Text-Font global setzen
_mont = pio.templates["plotly_white"].layout.to_plotly_json()
_mont["font"] = {"family": "Montserrat"}  # globaler Font für Titel, Achsen, Legend
pio.templates["montserrat_white"] = pio.templates["plotly_white"]
pio.templates["montserrat_white"].layout.font.family = "Montserrat"

# global aktivieren (wirkt für alle nachfolgenden Plotly-Figuren)
pio.templates.default = "montserrat_white"


# === Erzwinge Montserrat je Achse (nur für Additional Metrics) ===
def force_montserrat(ax):
    # Titel & Achsentitel
    if ax.get_title(): ax.title.set_fontfamily("Montserrat")
    if ax.get_xlabel(): ax.xaxis.label.set_fontfamily("Montserrat")
    if ax.get_ylabel(): ax.yaxis.label.set_fontfamily("Montserrat")
    # Ticks
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("Montserrat")
    # Legende
    leg = ax.get_legend()
    if leg:
        for text in leg.get_texts():
            text.set_fontfamily("Montserrat")
def _parse_selected_vertex(sel):
    """
    Gibt ('orig'| 'convex' | None, int|None) zurück.
    Erlaubt: int/np.int (Original) oder 'convex:<id>' (Konvex).
    """
    import numpy as np
    if sel is None:
        return (None, None)
    if isinstance(sel, str) and sel.startswith("convex:"):
        try:
            return ("convex", int(sel.split(":", 1)[1]))
        except Exception:
            return ("convex", None)
    # Original (int oder str-zahl)
    try:
        val = int(sel)
        return ("orig", val)
    except Exception:
        return (None, None)

def clean_plot_indices(raw_indices):
    """
    Wandelt eine Liste wie ['np.int64(22)', 'np.int64(738)'] in [22, 738] um.
    Funktioniert auch bei echten ints und np.int64-Objekten.
    """
    cleaned = []
    for idx in raw_indices:
        try:
            if isinstance(idx, int):
                cleaned.append(idx)
            elif isinstance(idx, str):
                match = re.search(r"\d+", idx)
                if match:
                    cleaned.append(int(match.group()))
            else:
                cleaned.append(int(idx))  # z. B. np.int64
        except Exception as e:
            print(f"⚠️ Error converting index {idx}: {e}")
    return cleaned
def select_representative_vertices_by_kmeans(
    df: pd.DataFrame,
    cols: list,
    n_vertices: int,
    index_subset=None,
    random_state: int = 42
) -> list:
    """
    Wählt repräsentative Vertices per k-Means-Clustering basierend auf den angegebenen Spalten.

    Args:
        df (pd.DataFrame): Vollständiges DataFrame mit Vertex-Daten.
        cols (list): Spalten, die für das Clustering verwendet werden (z. B. Zeitreihendaten).
        n_vertices (int): Gewünschte Anzahl an Vertices.
        index_subset (optional): Teilmenge von Indizes zur Auswahl (z. B. current_indices).
        random_state (int): Reproduzierbarkeit des Clusters.

    Returns:
        list: Liste mit ausgewählten Vertex-Indizes.
    """
    if index_subset is not None:
        data = df.loc[index_subset, cols]
    else:
        data = df[cols]

    # Fehlende Werte mit 0 ersetzen (oder ggf. eine andere Strategie verwenden)
    data = data.fillna(0)

    # Begrenze Anzahl Cluster auf max. Anzahl verfügbarer Zeilen
    n_clusters = min(len(data), n_vertices)

    if n_clusters == 0:
        return []

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(data)

    selected_indices = []
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        if not cluster_points.empty:
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points.values - center, axis=1)
            closest_idx = cluster_points.index[np.argmin(distances)]
            selected_indices.append(closest_idx)

    return selected_indices
    
def select_and_show_vertex_info(
    plot_indices,
    current_indices,
    vertex_df,
    tech_data,
    additional_cols,
    convex_plot_indices=None,        # NEU
    convex_additional=None           # NEU (gefilterte konvexe Zusatzmetriken)
):
    vertex_placeholder = "— Please select —"
    orig_opts = [str(i) for i in plot_indices]
    conv_opts = [f"convex:{int(i)}" for i in (convex_plot_indices or [])]
    options_with_placeholder = [vertex_placeholder] + orig_opts + conv_opts

    # Hübschere Labels
    def _fmt(label):
        if label == vertex_placeholder:
            return label
        if isinstance(label, str) and label.startswith("convex:"):
            return f"Convex #{label.split(':', 1)[1]}"
        return f"Vertex {label}"

    selected_label = st.selectbox(
        "Choose a displayed Vertex to highlight.",
        options=options_with_placeholder,
        format_func=_fmt
    )
    selected_vertex = None if selected_label == vertex_placeholder else selected_label

    # ==== Zusatzinfos anzeigen (bis zu 5 Metriken) ====
    if selected_vertex is not None and additional_cols:
        sel_type, sel_id = _parse_selected_vertex(selected_vertex)

        if sel_type == "orig" and sel_id in current_indices:
            st.markdown("### ℹ️ Zusatzinformationen (Original)")
            filtered_additional = vertex_df.loc[current_indices, additional_cols[:5]]
            full_additional = vertex_df.loc[tech_data.index, additional_cols[:5]]
            extra_data = filtered_additional.loc[sel_id]
            col1_inner, col2_inner = st.columns(2)
            extra_items = list(extra_data.items())
            for i in range(0, len(extra_items), 2):
                for col, (key, val) in zip([col1_inner, col2_inner], extra_items[i:i+2]):
                    val_display = f"{val:.2f}" if pd.notna(val) else "n/a"
                    col.markdown(f"**{key}**: {val_display}")
                    if pd.notna(val):
                        col_vals = full_additional[key].dropna()
                        if not col_vals.empty:
                            q1 = col_vals.quantile(0.25)
                            q2 = col_vals.quantile(0.5)
                            q3 = col_vals.quantile(0.75)
                            min_val = col_vals.min()
                            max_val = col_vals.max()
                            fig, ax = plt.subplots(figsize=(3.5, 0.3))
                            ax.hlines(0, min_val, max_val, color="lightgray", linewidth=6)
                            for q in [q1, q2, q3]:
                                ax.vlines(q, -0.1, 0.1, color="gray", linewidth=1)
                            ax.plot(val, 0, 'o')
                            ax.set_xlim(min_val, max_val)
                            ax.set_yticks([]); ax.set_xticks([])
                            for spine in ax.spines.values():
                                spine.set_visible(False)
                            col.pyplot(fig)

        elif sel_type == "convex" and convex_additional is not None and not convex_additional.empty:
            if 0 <= sel_id < len(convex_additional):
                st.markdown("### ℹ️ Zusatzinformationen (Convex)")
                # Nur innerhalb der konvexen Daten betrachten
                filtered_additional = convex_additional[additional_cols[:5]].copy() if set(additional_cols[:5]).issubset(convex_additional.columns) else pd.DataFrame()
                if not filtered_additional.empty:
                    extra_data = filtered_additional.iloc[sel_id]
                    col1_inner, col2_inner = st.columns(2)
                    extra_items = list(extra_data.items())
                    for i in range(0, len(extra_items), 2):
                        for col, (key, val) in zip([col1_inner, col2_inner], extra_items[i:i+2]):
                            val_display = f"{val:.2f}" if pd.notna(val) else "n/a"
                            col.markdown(f"**{key}**: {val_display}")

    return selected_vertex

def plot_density_contours(
    tech_time_map,
    vertex_df,
    current_indices,
    num_interpolated_points=3,
    grid_density=50,
    color_levels=10,
    max_vertices_for_density=250,
    tag="density",   # << NEU
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    # Anzahl Plots = Anzahl Techs mit mindestens einer Spalte
    techs_for_plot = []
    for tech, year_cols in sorted(tech_time_map.items()):
        if any(c in vertex_df.columns for _, c in year_cols):
            techs_for_plot.append(tech)

    if not techs_for_plot:
        if not st.session_state.get("capture_exports"):
            st.info("ℹ️ No data to plot.")
        return None

    (fig_w_in, fig_h_in), n_rows, n_cols = compute_mpl_figsize(
        n_plots=len(techs_for_plot),
        n_cols=st.session_state.get("n_cols_plots", 3),
        col_w_in=st.session_state.get("col_w_in", DEFAULT_COL_WIDTH_IN),
        row_h_in=st.session_state.get("row_h_in", DEFAULT_ROW_HEIGHT_IN),
    )
    fig_dichte, axs = plt.subplots(n_rows, n_cols, figsize=(fig_w_in, fig_h_in))
    axs = axs.flatten()
    fig_dichte.patch.set_facecolor('#f4f4f4')

    plot_indices = st.session_state.get("plot_indices", current_indices)
    df_base = vertex_df.loc[plot_indices]
    if len(df_base) > max_vertices_for_density:
        df_base = df_base.sample(n=max_vertices_for_density, random_state=42)

    i = 0
    drew_anything = False
    for tech in techs_for_plot:
        year_cols = tech_time_map.get(tech, [])
        if not year_cols:
            axs[i].set_visible(False)
            i += 1
            continue

        year_cols = sorted(year_cols, key=lambda x: x[0])
        years = [y for y, _ in year_cols]
        cols  = [c for _, c in year_cols if c in df_base.columns]

        if not cols:
            axs[i].set_visible(False)
            i += 1
            continue

        df = df_base[cols]
        if df.dropna(how='all').empty:
            axs[i].set_visible(False)
            i += 1
            continue

        axs[i].set_facecolor('#f0f0f0')

        x_vals = np.array(years[:len(cols)])  # robust falls columns < years
        all_points = []
        for row in df.itertuples(index=False):
            y_vals = np.array(row)
            if np.isnan(y_vals).any():
                continue
            # lin. Interpolation zwischen aufeinanderfolgenden Jahren
            for j in range(len(x_vals) - 1):
                x_interp = np.linspace(x_vals[j], x_vals[j + 1], num_interpolated_points)
                y_interp = np.linspace(y_vals[j], y_vals[j + 1], num_interpolated_points)
                all_points.extend(zip(x_interp, y_interp))

        if not all_points:
            axs[i].set_visible(False)
            i += 1
            continue

        X, Y = np.meshgrid(
            np.linspace(min(x_vals), max(x_vals), grid_density),
            np.linspace(-0.5, df.max().max() + 0.5, grid_density)
        )

        Z = np.reshape(
            gaussian_kde(np.array(all_points).T, bw_method=0.1)(np.vstack([X.ravel(), Y.ravel()])),
            X.shape
        )

        Z_masked = np.where((Y >= -0.5) & (Y <= df.max().max() + 0.5), Z, np.nan)

        contour = axs[i].contourf(X, Y, Z_masked, levels=color_levels, cmap="inferno", extend="both")
        axs[i].set_facecolor("white")
        axs[i].set_title(tech.replace("_", " ").title())
        axs[i].set_xlabel("Year")
        axs[i].set_ylabel("Installed Capacity")
        axs[i].set_ylim(-1, df.max().max() * 1.05)

        min_vals = df.min()
        max_vals = df.max()
        axs[i].fill_between(x_vals, max_vals + 0.5, max_vals.max() + 0.5, facecolor='#f4f4f4', alpha=1)
        axs[i].fill_between(x_vals, min_vals - 0.5, -0.5, facecolor='#f4f4f4', alpha=1)

        cbar = plt.colorbar(contour, ax=axs[i], label="Density")
        cbar.set_ticks(np.linspace(np.nanmin(Z_masked), np.nanmax(Z_masked), 4))
        cbar.set_ticklabels([f"{val:.3f}" for val in np.linspace(np.nanmin(Z_masked), np.nanmax(Z_masked), 4)])

        drew_anything = True
        i += 1

    for j in range(i, len(axs)):
        fig_dichte.delaxes(axs[j])

    if not drew_anything:
        plt.close(fig_dichte)
        if not st.session_state.get("capture_exports"):
            st.info("ℹ️ No data to plot.")
        return None

    fig_dichte.subplots_adjust(top=0.95, bottom=0.07, hspace=0.44, wspace=0.3)

    # << NEU: zentral entscheiden
    emit_mpl(fig_dichte, tag)
    return None
    

def plot_violin_values(
    vertex_df,
    valid_techs_value,
    value_time_map,
    plot_indices_val,
    current_indices,
    filtered_convex_data,
    MAA_PREFIX="MAA_",
    tag="operational_violin",   # << NEU
):
    """
    Violinplots mit separatem Subplot für 'Cumulated'.
    - Bei 'capture_exports' werden die Figuren nicht gerendert, sondern per emit_mpl() exportiert.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from math import ceil

    def _nice(t: str) -> str:
        return t.replace("_", " ").title()

    def _xlab(y: int) -> str:
        return "Static" if y == -1 else str(y)

    # ---------- 1) Plot-Spezifikationen ----------
    plot_specs = []
    for tech in valid_techs_value:
        pairs = sorted(value_time_map.get(tech, []), key=lambda x: x[0])
        if not pairs:
            continue

        yearly = [(y, c) for (y, c) in pairs if y != 0 and c in vertex_df.columns]
        if yearly:
            years_main = [y for y, _ in yearly]
            cols_main  = [c for _, c in yearly]
            plot_specs.append({"tech": tech, "kind": "main", "years": years_main, "cols": cols_main})

        cum = [(y, c) for (y, c) in pairs if y == 0 and c in vertex_df.columns]
        if cum:
            cols_cum = [c for _, c in cum]
            plot_specs.append({"tech": tech, "kind": "cum", "cols": cols_cum})

    if not plot_specs:
        if not st.session_state.get("capture_exports"):
            st.info("ℹ️ No data to plot.")
        return None

    # ---------- 2) Grid ----------
    (fig_w_in, fig_h_in), n_rows, n_cols = compute_mpl_figsize(
        n_plots=len(plot_specs),
        n_cols=st.session_state.get("n_cols_plots", 3),
        col_w_in=st.session_state.get("col_w_in", DEFAULT_COL_WIDTH_IN),
        row_h_in=st.session_state.get("row_h_in", DEFAULT_ROW_HEIGHT_IN),
    )
    fig_val, axes_val = plt.subplots(n_rows, n_cols, figsize=(fig_w_in, fig_h_in), squeeze=False)
    fig_val.patch.set_facecolor('#f4f4f4')
    axes_list = axes_val.flatten().tolist()

    # ---------- 3) Zeichnen ----------
    for i, spec in enumerate(plot_specs):
        ax = axes_list[i]
        ax.set_facecolor('#f0f0f0')
        tech = spec["tech"]

        if spec["kind"] == "main":
            years = spec["years"]
            cols  = spec["cols"]
            values_matrix = vertex_df.loc[current_indices, cols].copy()

            if (
                st.session_state.get('show_convex', False)
                and filtered_convex_data is not None
                and not filtered_convex_data.empty
                and all(c in filtered_convex_data.columns for c in cols)
            ):
                values_matrix = pd.concat([values_matrix, filtered_convex_data[cols]], axis=0)

            x_positions, x_labels, data_series = [], [], []
            for j, (y, c) in enumerate(sorted(zip(years, cols), key=lambda t: t[0]), start=1):
                vals = values_matrix[c].dropna().values
                if len(vals) > 0:
                    x_positions.append(j)
                    x_labels.append(_xlab(y))
                    data_series.append(vals)

            if not data_series:
                ax.set_visible(False)
                continue

            vp = ax.violinplot(
                data_series,
                positions=x_positions,
                showmeans=False,
                showmedians=True,
                showextrema=True,
                widths=0.9
            )
            for pc in vp['bodies']:
                pc.set_alpha(0.7)
            if 'cmedians' in vp:
                vp['cmedians'].set_color('black')

            sel = st.session_state.get("selected_vertex")
            if sel is not None and sel in vertex_df.index:
                try:
                    sel_vals = vertex_df.loc[sel, cols]
                    sorted_cols = [c for _, c in sorted(zip(years, cols), key=lambda t: t[0])]
                    for x_pos, c in zip(x_positions, sorted_cols):
                        v = sel_vals.get(c, np.nan)
                        if pd.notnull(v):
                            ax.scatter(x_pos, float(v), color="black", s=60, zorder=3,
                                       label="Selected Vertex")
                except Exception as e:
                    st.warning(f"⚠️ Error highlighting selected vertex for {tech}: {e}")

            if st.session_state.get('show_original_ranges', False):
                try:
                    orig = vertex_df.loc[current_indices, cols]
                    sorted_cols = [c for _, c in sorted(zip(years, cols), key=lambda t: t[0])]
                    for x_pos, c in zip(x_positions, sorted_cols):
                        col_vals = orig[c].dropna()
                        if not col_vals.empty:
                            omin, omax = col_vals.min(), col_vals.max()
                            ax.fill_between([x_pos - 0.25, x_pos + 0.25], omin, omax,
                                            color=(1.0, 0.0, 0.0, 0.08), zorder=1)
                except Exception:
                    pass

            ax.set_title(_nice(tech))
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels)
            if (i % n_cols) == 0:
                ax.set_ylabel("VALUE_")
            ax.grid(True, linestyle="--", alpha=0.4)

        else:  # cumulated
            cols_cum = spec["cols"]
            values_cum = vertex_df.loc[current_indices, cols_cum].copy()

            if (
                st.session_state.get('show_convex', False)
                and filtered_convex_data is not None
                and not filtered_convex_data.empty
                and all(c in filtered_convex_data.columns for c in cols_cum)
            ):
                values_cum = pd.concat([values_cum, filtered_convex_data[cols_cum]], axis=0)

            if values_cum.shape[1] > 1:
                cum_vals = values_cum.mean(axis=1).dropna().values
            elif values_cum.shape[1] == 1:
                cum_vals = values_cum.iloc[:, 0].dropna().values
            else:
                cum_vals = np.array([])

            if cum_vals.size == 0:
                ax.set_visible(False)
                continue

            vp = ax.violinplot(
                [cum_vals],
                positions=[1],
                showmeans=False,
                showmedians=True,
                showextrema=True,
                widths=0.9
            )
            for pc in vp['bodies']:
                pc.set_alpha(0.7)
            if 'cmedians' in vp:
                vp['cmedians'].set_color('black')

            sel = st.session_state.get("selected_vertex")
            if sel is not None and sel in vertex_df.index:
                try:
                    sel_vals = vertex_df.loc[sel, cols_cum]
                    v = float(np.nanmean(sel_vals.values)) if sel_vals.size > 0 else np.nan
                    if np.isfinite(v):
                        ax.scatter(1, v, color="black", s=60, zorder=3)
                except Exception as e:
                    st.warning(f"⚠️ Error highlighting selected vertex (cumulated) for {tech}: {e}")

            if st.session_state.get('show_original_ranges', False):
                try:
                    orig = vertex_df.loc[current_indices, cols_cum]
                    omin, omax = float(np.nanmin(orig.values)), float(np.nanmax(orig.values))
                    if np.isfinite(omin) and np.isfinite(omax):
                        ax.fill_between([1 - 0.25, 1 + 0.25], omin, omax,
                                        color=(1.0, 0.0, 0.0, 0.08), zorder=1)
                except Exception:
                    pass

            ax.set_title(f"{_nice(tech)}_Cumulated")
            ax.set_xticks([1]); ax.set_xticklabels(["Cumulated"])
            if (i % n_cols) == 0:
                ax.set_ylabel("VALUE_")
            ax.grid(True, linestyle="--", alpha=0.4)

    # ---------- 4) Überzählige Subplots ausblenden ----------
    for j in range(len(plot_specs), len(axes_list)):
        axes_list[j].set_visible(False)

    # Optional: Legende für Selected Vertex
    if any(hasattr(ax, 'has_data') and ax.has_data() for ax in axes_list) and st.session_state.get("selected_vertex") is not None:
        vertex_dot = mlines.Line2D([], [], color="black", marker='o', linestyle='None', markersize=8,
                                   label="Selected Vertex")
        fig_val.legend(
            [vertex_dot], ["Selected Vertex"],
            loc='upper center', bbox_to_anchor=(0.5, 1.02),
            ncol=1, frameon=True, fancybox=True, fontsize=12
        )

    fig_val.subplots_adjust(top=0.90, bottom=0.08, hspace=0.35, wspace=0.25)

    # << NEU: zentral entscheiden
    emit_mpl(fig_val, tag)
    return None

def plot_operational_variables_over_time(
    vertex_df,
    current_indices,
    plot_indices_val,
    time_column_map,
    selected_vertex,
    n_cols_val=3,
    show_convex=False,
    st_convex=None,
    filtered_convex_data=None,
    show_original_ranges=False,
    max_plot_vertices=20,
    maa_prefix="MAA_",
    apply_prefix=True,
    plot_title="Operational Variables Over Time",
    h_gap=0.09,          # ungenutzt (Backward-compat)
    v_gap=0.08,          # ungenutzt (Backward-compat)
    convex_cluster_indices=None,
    tag="operational_line",   # << NEU: Export-Tag/Dateiname
):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def _x_label(y: int) -> str:
        return "Static" if y == -1 else str(y)

    sel_type, sel_id = _parse_selected_vertex(selected_vertex)

    # 1) Plot-Spezifikationen aufbauen
    plot_specs = []
    valid_techs = sorted([tech for tech, pairs in time_column_map.items() if pairs])
    for tech in valid_techs:
        year_cols = sorted(time_column_map.get(tech, []), key=lambda x: x[0])
        if not year_cols:
            continue

        # "main" = alle Jahre != 0
        filt_main = [(y, c) for y, c in year_cols if y != 0 and (not apply_prefix or str(c).startswith(maa_prefix + tech))]
        if filt_main:
            years_main = [y for y, _ in filt_main]
            cols_main  = [c for _, c in filt_main]
            plot_specs.append({"tech": tech, "kind": "main", "years": years_main, "cols": cols_main})

        # "cum" = Pseudo-Jahr 0
        filt_cum = [(y, c) for y, c in year_cols if y == 0 and (not apply_prefix or str(c).startswith(maa_prefix + tech))]
        if filt_cum:
            cols_cum = [c for _, c in filt_cum]
            plot_specs.append({"tech": tech, "kind": "cum", "cols": cols_cum})

    n_plots = len(plot_specs)
    if n_plots == 0:
        if not st.session_state.get("capture_exports"):
            st.info("ℹ️ No data to plot.")
        return None

    # 2) Grid/Höhe anhand Session-Settings
    row_h_px = int(st.session_state.get("row_h_px", DEFAULT_ROW_HEIGHT_PX))
    hgap     = float(st.session_state.get("hspace_frac", DEFAULT_HSPACE_FRAC))
    vgap_px  = int(st.session_state.get("row_gap_px", 12))

    n_rows, n_cols, horizontal_spacing, vertical_spacing, height, top_m, bottom_m = compute_plotly_grid(
        n_plots=n_plots,
        n_cols=n_cols_val,
        row_height_px=row_h_px,
        hspace_frac=hgap,
        vspace_px=vgap_px,
        top_margin_px=DEFAULT_TOP_MARGIN,
        bottom_margin_px=DEFAULT_BOTTOM_MARGIN
    )

    # Subplot-Titel
    subplot_titles = []
    for spec in plot_specs:
        nice = spec["tech"].replace("_", " ").title()
        subplot_titles.append(nice if spec["kind"] == "main" else f"{nice}_Cumulated")
    subplot_titles += [""] * max(0, n_rows * n_cols - n_plots)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
        specs=[[{"type": "xy"} for _ in range(n_cols)] for _ in range(n_rows)],
    )

    # 3) Zeichnen
    for i, spec in enumerate(plot_specs):
        row = i // n_cols + 1
        col = i % n_cols + 1
        tech = spec["tech"]

        if spec["kind"] == "main":
            years_main = spec["years"]; cols_main = spec["cols"]
            if not cols_main:
                continue

            years_labels = [_x_label(y) for y in years_main]
            all_numeric_years = all(isinstance(y, int) and y >= 1000 for y in years_main)

            full_values_main = vertex_df.loc[current_indices, cols_main]
            values_main      = vertex_df.loc[plot_indices_val, cols_main]

            # Original-Vertices (Linien oder Marker)
            if not values_main.dropna(how="all").empty:
                plot_mode = "lines" if (len(years_labels) > 1 and all_numeric_years) else "markers"
                for idx_v in values_main.index:
                    vals = values_main.loc[idx_v].values
                    is_sel = (sel_type == "orig" and idx_v == sel_id)
                    fig.add_trace(
                        go.Scatter(
                            x=years_labels, y=vals,
                            mode=plot_mode,
                            line=dict(
                                color="rgba(26,102,204,0.9)" if is_sel else "rgba(26,102,204,0.35)",
                                width=4 if is_sel else 1,
                            ) if plot_mode == "lines" else None,
                            marker=dict(
                                size=10,
                                color="rgba(26,102,204,0.9)" if is_sel else "rgba(26,102,204,0.35)",
                            ) if plot_mode == "markers" else None,
                            hovertemplate=f"Vertex {idx_v}<br>%{{x}}: %{{y:.2f}}<extra></extra>",
                            showlegend=False,
                        ),
                        row=row, col=col,
                    )

            # Gültiger Bereich über alle gültigen Vertices (blau)
            if all_numeric_years and not full_values_main.empty:
                try:
                    vmin = full_values_main.min(); vmax = full_values_main.max()
                    x_vals = list(map(str, years_main)) + list(map(str, years_main[::-1]))
                    y_vals = vmin.tolist() + vmax.tolist()[::-1]
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals, y=y_vals,
                            fill="toself", fillcolor="rgba(26,102,204,0.15)",
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip", showlegend=False,
                        ),
                        row=row, col=col,
                    )
                except Exception as e:
                    st.warning(f"⚠️ Error adding min/max fill for {tech}: {e}")

            # Original-Bereich (rot), wenn gewünscht
            if show_original_ranges and all_numeric_years:
                try:
                    orig = vertex_df.loc[current_indices, cols_main]
                    omin = orig.min(); omax = orig.max()
                    x_vals = list(map(str, years_main)) + list(map(str, years_main[::-1]))
                    y_vals = omin.tolist() + omax.tolist()[::-1]
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals, y=y_vals,
                            fill="toself", fillcolor="rgba(255,0,0,0.08)",
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip", showlegend=False,
                        ),
                        row=row, col=col,
                    )
                except Exception as e:
                    st.write(f"❌ Error displaying original range for {tech}: {e}")

            # Konvex-Overlays (repräsentative Linien)
            if show_convex and filtered_convex_data is not None and not filtered_convex_data.empty:
                try:
                    numeric_pairs = [
                        (y, c) for y, c in sorted(zip(years_main, cols_main), key=lambda t: t[0])
                        if (isinstance(y, int) and y >= 1000) and (c in filtered_convex_data.columns)
                    ]
                    if numeric_pairs and all_numeric_years:
                        x_conv = [str(y) for y, _ in numeric_pairs]
                        conv_cols = [c for _, c in numeric_pairs]
                        reps = (convex_cluster_indices or
                                select_representative_vertices_by_kmeans(filtered_convex_data, conv_cols, 5))
                        reps = [r for r in reps if r in filtered_convex_data.index]
                        for idx_conv in reps:
                            vals = filtered_convex_data.loc[idx_conv, conv_cols].values.astype(float)
                            if np.isnan(vals).all():
                                continue
                            is_sel = (sel_type == "convex" and idx_conv == sel_id)
                            fig.add_trace(
                                go.Scatter(
                                    x=x_conv, y=vals, mode="lines",
                                    line=dict(
                                        color="rgba(200,0,0,0.9)" if is_sel else "rgba(255,0,0,0.35)",
                                        width=4 if is_sel else 1
                                    ),
                                    hovertemplate=f"Convex {idx_conv}<br>%{{x}}: %{{y:.2f}}<extra></extra>",
                                    showlegend=False,
                                ),
                                row=row, col=col,
                            )
                except Exception as e:
                    st.warning(f"⚠️ Error adding convex overlays for {tech}: {e}")

            fig.update_xaxes(type="category", row=row, col=col, title_text=None)

        else:
            cols_cum = spec["cols"]
            if not cols_cum:
                continue

            values_cum      = vertex_df.loc[plot_indices_val, cols_cum]
            full_values_cum = vertex_df.loc[current_indices, cols_cum]

            # Original-Punkte (Mittelwert je Zeile)
            if not values_cum.dropna(how="all").empty:
                for idx_v in values_cum.index:
                    vals = values_cum.loc[idx_v].values
                    v = np.nanmean(vals) if len(vals) > 1 else (vals[0] if len(vals) == 1 else np.nan)
                    if np.isnan(v):
                        continue
                    is_sel = (sel_type == "orig" and idx_v == sel_id)
                    fig.add_trace(
                        go.Scatter(
                            x=["Cumulated"], y=[v], mode="markers",
                            marker=dict(
                                size=10,
                                color="rgba(26,102,204,0.9)" if is_sel else "rgba(26,102,204,0.35)",
                            ),
                            hovertemplate=f"Vertex {idx_v}<br>Cumulated: {v:.2f}<extra></extra>",
                            showlegend=False,
                        ),
                        row=row, col=col,
                    )

            # Gültiger Bereich (blau)
            if not full_values_cum.dropna(how="all").empty:
                try:
                    vmin = float(np.nanmin(full_values_cum.values))
                    vmax = float(np.nanmax(full_values_cum.values))
                    if np.isfinite(vmin) and np.isfinite(vmax):
                        fig.add_trace(
                            go.Scatter(
                                x=["Cumulated", "Cumulated"], y=[vmin, vmax],
                                mode="lines", line=dict(width=0),
                                fill="toself", fillcolor="rgba(26,102,204,0.15)",
                                hoverinfo="skip", showlegend=False,
                            ),
                            row=row, col=col,
                        )
                except Exception:
                    pass

            # Originalbereich (rot)
            if show_original_ranges and not full_values_cum.dropna(how="all").empty:
                try:
                    vmin = float(np.nanmin(full_values_cum.values))
                    vmax = float(np.nanmax(full_values_cum.values))
                    if np.isfinite(vmin) and np.isfinite(vmax):
                        fig.add_trace(
                            go.Scatter(
                                x=["Cumulated", "Cumulated"], y=[vmin, vmax],
                                mode="lines", line=dict(width=0),
                                fill="toself", fillcolor="rgba(255,0,0,0.08)",
                                hoverinfo="skip", showlegend=False,
                            ),
                            row=row, col=col,
                        )
                except Exception:
                    pass

            # Konvex-Punkte (x-Marker)
            if show_convex and filtered_convex_data is not None and not filtered_convex_data.empty:
                try:
                    needed = [c for c in cols_cum if c in filtered_convex_data.columns]
                    if needed:
                        reps = (convex_cluster_indices or
                                select_representative_vertices_by_kmeans(filtered_convex_data, needed, 5))
                        reps = [r for r in reps if r in filtered_convex_data.index]
                        for idx_conv in reps:
                            vals = filtered_convex_data.loc[idx_conv, needed].values
                            v = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
                            if np.isfinite(v):
                                is_sel = (sel_type == "convex" and idx_conv == sel_id)
                                fig.add_trace(
                                    go.Scatter(
                                        x=["Cumulated"], y=[v],
                                        mode="markers",
                                        marker=dict(symbol="x", size=12 if is_sel else 8),
                                        marker_color="rgba(200,0,0,0.9)" if is_sel else "rgba(255,0,0,0.6)",
                                        hovertemplate=f"Convex {idx_conv}<br>Cumulated: {v:.2f}<extra></extra>",
                                        showlegend=False,
                                    ),
                                    row=row, col=col,
                                )
                except Exception as e:
                    st.warning(f"⚠️ Error adding convex overlays (cumulated) for {tech}: {e}")

            fig.update_xaxes(type="category", row=row, col=col, tickvals=["Cumulated"], title_text=None)

    # 4) Layout & Rahmen
    fig.update_layout(
        title=dict(text=plot_title, x=0, xanchor="left"),
        font=dict(size=12, family="Montserrat", color="#333"),
        paper_bgcolor="#f4f4f4", plot_bgcolor="#f4f4f4",
        hovermode="closest",
        margin=dict(l=10, r=10, t=DEFAULT_TOP_MARGIN, b=DEFAULT_BOTTOM_MARGIN),
        showlegend=False,
        height=height,
    )
    fig.update_xaxes(constrain="domain", automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_annotations(font=dict(size=12, color="#222", family="Montserrat"))

    fig = add_subplot_borders(
        fig,
        color="#C0C6D2",
        width=1.2,
        dash=None,
        pad=0.01,
        above=True,
        only_used=False
    )

    # << NEU: zentral entscheiden, ob rendern oder capturen
    emit_plotly(fig, tag)
    return None


def build_cols_from_time_map(time_map, techs, MAA_PREFIX,mode=None):
    """
    Erzeugt die Spaltenliste (cols) aus dem time_map für die gegebenen Technologien.
    - Sortiert pro Technologie nach Jahr (Index 0 der Tupel).
    - Für VALUE_: nimmt nur Spaltennamen, die mit f"{MAA_PREFIX}{tech}" beginnen.
    - Für MAA_: entpackt (year, col)-Paare und sammelt die Spalten.
    - Entfernt Duplikate bei gleichzeitiger Stabilisierung der Reihenfolge.
    """
    cols = []
    
    for tech in techs:
        year_cols = time_map.get(tech, [])
        years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
        
        
        if MAA_PREFIX == "VALUE_":
            if mode == "operational":
                cols.extend([c for _, c in years_cols_sorted
                 if isinstance(c, str) and c.startswith(MAA_PREFIX + tech)])
            else:
                cols.extend([c for _, c in years_cols_sorted
                      if isinstance(c, str) and c.startswith("INSTALLED_CAPACITY_" + tech)])
        else:  # MAA_
            try:
                _, cols_tech = zip(*years_cols_sorted) if years_cols_sorted else ([], [])
                cols.extend(list(cols_tech))
            except ValueError:
                # Falls year_cols leer/unkonsistent sind: überspringen
                continue

    # Duplikate entfernen, Reihenfolge beibehalten
    cols = list(dict.fromkeys(cols))
    return cols

def prepare_vertex_selection(
    MAA_PREFIX,
    vertex_df,
    tech_time_map,
    extract_time_series_map,
    select_representative_vertices_by_kmeans,
    current_indices,
    max_plot_vertices,
    st,
    mode=None,
):
    """
    Bereitet die Vertex-Auswahl basierend auf dem MAA_PREFIX vor.
    Gibt: time_map, valid_techs, plot_indices, selected_tech, cols zurück
    """
    if MAA_PREFIX == "VALUE_":
        source = "value_time_map"
        
        time_map = extract_time_series_map(vertex_df, MAA_PREFIX, mode="operational")
        valid_techs = sorted([tech for tech, v in time_map.items() if len(v) >= 1])
    elif MAA_PREFIX == "MAA_":
        source = "tech_time_map"
        time_map = tech_time_map
        valid_techs = sorted([tech for tech, v in time_map.items() if len(v) >= 1])
    else:
        st.error(f"❌ Unknown MAA_PREFIX: '{MAA_PREFIX}'. Expected 'VALUE_' or 'MAA_'.")
        return {}, [], [], None, []

    if not valid_techs:
        st.warning(f"⚠️ No valid technologies found in `{source}`.")
        return time_map, valid_techs, [], None, []

    tech = valid_techs[0]

    try:
        # ⬇️ Neu: Cols-Erstellung ausgelagert; Verhalten bleibt: nur erste Tech
        cols = build_cols_from_time_map(time_map, valid_techs, MAA_PREFIX,mode)

        if not cols:
            st.error(f"❌ No matching columns found for technology '{tech}'.")
            return time_map, valid_techs, [], tech, []

        if len(current_indices) > max_plot_vertices:
            
            plot_indices = select_representative_vertices_by_kmeans(
                df=vertex_df,
                cols=cols,
                n_vertices=max_plot_vertices,
                index_subset=current_indices
            )
        else:
            plot_indices = current_indices

    except Exception as e:
        st.error(f"❌ Error processing technology '{tech}': {e}")
        return time_map, valid_techs, [], tech, []

    return time_map, valid_techs, plot_indices, tech, cols
    
# === Initialisiere Session State ===
def initialize_session_state():
    defaults = {
        'convex_combinations': pd.DataFrame(),
        'convex_additional': pd.DataFrame(),
        'show_convex': True,
        'show_original_ranges': False,          # <--- ergänzt
        'plot_type_selector': 'Violinplot',     # <--- ergänzt
        'stored_figures': [],                   # <--- ergänzt
        'excel_loaded': False,
        'excel_path': '',
        'excel_error': None,
        'n_cols_plots': 3,                      # Standardanzahl Subplot-Spalten
        'max_plot_vertices': 5,
        "column_ratio" : 0.3,
        'layout_mode': "Two-column layout",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session_state()

# === Lade Excel Datei ===
@st.cache_data
def load_excel_data(path):
    return pd.read_excel(path)

# === Extrahiere Technologien aus DataFrame ===
def extract_technologies(df):
    maa_cols = [col for col in df.columns if col.startswith(MAA_PREFIX)]
    techs = [col.replace(MAA_PREFIX, '') for col in maa_cols]
    return techs, df[maa_cols].drop_duplicates(), maa_cols

# === Extrahiere Zeitreihe pro Technologie ===
def extract_time_series_map(df, maa_prefix, mode="installed"):
    """
    Baut ein Mapping: Technologie -> Liste[(Jahr, Spaltenname)].

    Für VALUE_ (mode="operational"):
      - Jahreswerte: VALUE_Tech[(2025,)]
      - Kumulierte Werte ohne Jahr: VALUE_Tech_Cumulated  -> Pseudo-Jahr 0 ("Cumulated")
      - Nicht-jahresspezifisch: VALUE_Tech               -> Pseudo-Jahr -1 ("Static")

    Für MAA_ (installed):
      - MAA_INSTALLED_CAPACITY_Tech_2025
    """
    tech_time_map = defaultdict(list)
    seen_keys = set()

    for col in df.columns:
        # === MAA_-Modus (Installed Capacities) ===
        if maa_prefix == "MAA_":
            match = re.match(r"^MAA_INSTALLED_CAPACITY_(.+)_(\d{4})$", col)
            if match:
                tech, year = match.groups()
                key = (tech, int(year))
                if key not in seen_keys:
                    tech_time_map[tech].append((int(year), col))
                    seen_keys.add(key)

        # === VALUE_-Modus ===
        elif maa_prefix == "VALUE_":
            if mode == "operational":
                # 1) Jahreswerte: VALUE_Tech[(2025,)]
                m_year = re.match(r"^(VALUE_)([^[]+)\[\((\d{4}),?\)\]$", col)
                if m_year:
                    _, tech, year = m_year.groups()
                    key = (tech, int(year))
                    if key not in seen_keys:
                        tech_time_map[tech].append((int(year), col))
                        seen_keys.add(key)
                    continue

                # 2) Kumuliert (ohne Jahr): VALUE_Tech_Cumulated -> Pseudo-Jahr 0
                m_cum = re.match(r"^VALUE_(.+)_Cumulated$", col)
                if m_cum:
                    tech = m_cum.group(1)
                    key = (tech, 0)  # 0 == "Cumulated"
                    if key not in seen_keys:
                        tech_time_map[tech].append((0, col))
                        seen_keys.add(key)
                    continue

                # 3) Nicht-jahresspezifisch (ohne Suffix): VALUE_Tech -> Pseudo-Jahr -1
                m_static = re.match(r"^VALUE_([A-Za-z0-9_]+)$", col)
                if m_static:
                    tech = m_static.group(1)
                    # Schutz, damit keine install/new capacity hier reingerät
                    if not tech.startswith("INSTALLED_CAPACITY_") and not tech.startswith("NEW_CAPACITY_"):
                        key = (tech, -1)  # -1 == "Static"
                        if key not in seen_keys:
                            tech_time_map[tech].append((-1, col))
                            seen_keys.add(key)
                    continue

            elif mode == "installed":
                # Für Installed Capacities im VALUE_-Modus: INSTALLED_CAPACITY_Tech_2025
                m_inst = re.match(r"^(INSTALLED_CAPACITY_)(.+)_(\d{4})$", col)
                if m_inst:
                    _, tech, year = m_inst.groups()
                    key = (tech, int(year))
                    if key not in seen_keys:
                        tech_time_map[tech].append((int(year), col))
                        seen_keys.add(key)

    # Sortiere Jahre innerhalb jeder Technologie
    return {
        tech: sorted(entries, key=lambda x: x[0])
        for tech, entries in tech_time_map.items()
    }

# === Bestimme zusätzliche Metrikspalten ===
def get_additional_columns(df):
    new_cap_indices = [i for i, col in enumerate(df.columns) if col.startswith(NEW_CAPACITY_PREFIX)]
    if not new_cap_indices:
        return []
    last_new_cap_idx = max(new_cap_indices)
    return [
        col for col in df.columns[last_new_cap_idx + 1:]
        if pd.api.types.is_numeric_dtype(df[col])
    ]

# === Einheitliche Filterlogik für Vertex- oder Konvexdaten ===
def apply_tech_filters(data, data_additional, session_state, ordered_techs, prefix):
    import streamlit as st

    if data.empty:
        
        return pd.DataFrame(), pd.DataFrame()

    # Sicherstellen, dass beide DataFrames die gleiche Indizierung haben
    data = data.copy()
    data_additional = data_additional.copy()
    data.index = data.index.astype(str)
    data_additional.index = data_additional.index.astype(str)
    common_indices = data.index.intersection(data_additional.index)
    data = data.loc[common_indices]
    data_additional = data_additional.loc[common_indices]

    filtered_indices = data.index

    for tech in ordered_techs:
        key = f"slider_{tech}"
        col_data = f"{prefix}{tech}"
        col_additional = tech

        if key in session_state:
            min_val, max_val = session_state[key]
            
            if col_data in data.columns:
                col_values = data.loc[filtered_indices, [col_data]]
                

                mask = (col_values[col_data] >= min_val) & (col_values[col_data] <= max_val)
                filtered_indices = filtered_indices[mask]

              

            elif col_additional in data_additional.columns:
                col_values = data_additional.loc[filtered_indices, [col_additional]]
               

                mask = (col_values[col_additional] >= min_val) & (col_values[col_additional] <= max_val)
                filtered_indices = filtered_indices[mask]

               

    return data.loc[filtered_indices].reset_index(drop=True), data_additional.loc[filtered_indices].reset_index(drop=True)

# === Titel & Initialisierung ===
st.title(" Technology Decision Tool")
# === Excel-Datei Ladebereich via Upload ===
DEFAULT_EXCEL_URL = "https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/vertex_results_default.xlsx"

@st.cache_data(show_spinner="📥 Load Default-Excel ...")
def read_default_excel_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("❌ Could not load the default file.")
    return pd.read_excel(BytesIO(response.content))



if not st.session_state.get("excel_loaded", False):

    st.subheader("📂 Select Excel File")
    col1, spacer, col2 = st.columns([2, 0.3, 1])

    with col1:
        upload_file = st.file_uploader("📤 Upload your own Excel file (.xlsx)", type=["xlsx"])
        # Upload in Session speichern, falls vorhanden
        if upload_file is not None:
            st.session_state["uploaded_file"] = upload_file
            st.session_state["use_default_excel"] = False

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📁 Use default Excel file"):
            st.session_state["use_default_excel"] = True
            # Entferne eventuell vorhandene Upload-Datei, damit keine Verwirrung entsteht
            if "uploaded_file" in st.session_state:
                del st.session_state["uploaded_file"]

    uploaded_file = st.session_state.get("uploaded_file", None)
    use_default_excel = st.session_state.get("use_default_excel", False)

    # Nur fortfahren, wenn eine der beiden Optionen gewählt wurde
    if use_default_excel or uploaded_file is not None:
        st.subheader("Excel Read-In method")

        option = st.selectbox(
            "Choose Option:",
            ["", "📥 Read-in all vertices", "📊 Apply clustering to retain representative vertices"]
        )

        if option == "📥 Read-in all vertices":
            if st.button("Read-in Excel File"):
                try:
                    df = read_default_excel_from_url(DEFAULT_EXCEL_URL) if use_default_excel else load_excel_data(uploaded_file)
                    st.session_state["uploaded_excel"] = df.copy()
                    st.session_state["excel_loaded"] = True
                    st.session_state["excel_error"] = None
                    st.rerun()
                except Exception as e:
                    st.session_state["excel_error"] = f"❌ Error while reading the file: {e}"

        elif option == "📊 Apply clustering to retain representative vertices":
            k_value = st.number_input(
                "Number of representative vertices to retain (KMeans)",
                min_value=50,
                max_value=5000,
                value=1000,
                step=50,
                key="clustering_k"
            )

            if st.button("Apply Clustering and Read-in"):
                try:
                    df = read_default_excel_from_url(DEFAULT_EXCEL_URL) if use_default_excel else load_excel_data(uploaded_file)
                    amount_vertices_requested = int(k_value)

                    coeff_columns = [col for col in df.columns if col.startswith("COEFF_")]
                    if not coeff_columns:
                        raise ValueError("❌ No COEFF_ columns found.")

                    last_coeff_col = coeff_columns[-1]
                    last_index_with_minus1 = df[df[last_coeff_col] == -1].index.max()
                    df_first_part = df.loc[:last_index_with_minus1].copy()
                    df_remaining = df.loc[last_index_with_minus1 + 1:].copy()

                    if amount_vertices_requested >= len(df):
                        st.session_state["uploaded_excel"] = df.copy()
                        st.session_state["excel_loaded"] = True
                        st.session_state["excel_error"] = None
                        st.rerun()

                    elif amount_vertices_requested <= len(df_first_part):
                        st.session_state["uploaded_excel"] = df_first_part.copy()
                        st.session_state["excel_loaded"] = True
                        st.session_state["excel_error"] = None
                        st.rerun()

                    else:
                        cluster_columns = [col for col in df.columns if col.startswith("VALUE_") or col.startswith("MAA_")]
                        df_remaining_unique = df_remaining.drop_duplicates(subset=cluster_columns)
                        remaining_target = amount_vertices_requested - len(df_first_part)

                        if len(df_remaining_unique) > remaining_target:
                            X = df_remaining_unique[cluster_columns].fillna(0).to_numpy()
                            kmeans = KMeans(n_clusters=remaining_target, random_state=42, n_init="auto")
                            df_remaining_unique["cluster"] = kmeans.fit_predict(X)
                            representative_indices = df_remaining_unique.groupby("cluster").head(1).index
                            df_clustered = df_remaining.loc[representative_indices].copy()
                        else:
                            df_clustered = df_remaining_unique.copy()

                        df_final = pd.concat([df_first_part, df_clustered], ignore_index=True)
                        st.session_state["uploaded_excel"] = df_final.copy()
                        st.session_state["excel_loaded"] = True
                        st.session_state["excel_error"] = None
                        st.rerun()

                except Exception as e:
                    st.session_state["excel_error"] = f"❌ Error during clustering: {e}"

    if st.session_state.get("excel_error"):
        st.error(st.session_state["excel_error"])

    st.stop()

# === Daten laden & vorbereiten ===
if "uploaded_excel" not in st.session_state:
    st.warning("⚠️ Please upload and (optionally) cluster an Excel file first.")
    st.stop()

vertex_df = st.session_state["uploaded_excel"]

# Präfixe automatisch bestimmen
if any(col.startswith("VALUE_") for col in vertex_df.columns):
    MAA_PREFIX = "VALUE_"
    INSTALLED_CAPACITY_PREFIX = "INSTALLED_CAPACITY_"
    NEW_CAPACITY_PREFIX = "NEW_CAPACITY_"
elif any(col.startswith("MAA_") for col in vertex_df.columns):
    MAA_PREFIX = "MAA_"
    INSTALLED_CAPACITY_PREFIX = "MAA_INSTALLED_CAPACITY_"
    NEW_CAPACITY_PREFIX = "NEW_CAPACITY_"
else:
    st.error("❌ Could not detect expected prefixes ('VALUE_' or 'MAA_') in the Excel columns.")
    st.stop()
technologies, tech_data, maa_cols = extract_technologies(vertex_df)
tech_time_map = extract_time_series_map(vertex_df,MAA_PREFIX)
additional_cols = get_additional_columns(vertex_df)

# === Sidebar: Einstellungen ===
st.sidebar.markdown("## ⚙️ Settings")


# === Sidebar: Strukturierte Einstellungen ===

with st.sidebar.expander("Layout Options", expanded=True):
    
    total_vertices_available = len(tech_data)
    st.radio(
        "Select layout mode",
        ["Two-column layout", "Full-width layout"],
        index=0,
        key="layout_mode"
    )
    if st.session_state.layout_mode == "Two-column layout":
        st.slider(
            "Column ratio (Right vs Left)",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.01,
            key="column_ratio"
        )
    st.number_input(
        "Number of columns in plot layout",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        key="n_cols_plots"
    )
    # --- NEU: Row-based sizing (ganz unten) ---
    st.markdown("---")
    st.caption("Row-based sizing")
    st.number_input(
        "Row height per subplot row (Plotly, px)",
        min_value=50, max_value=1200,
        value=st.session_state.get("row_h_px", DEFAULT_ROW_HEIGHT_PX),
        step=20, key="row_h_px"
    )
    st.number_input(
        "Row gap between subplot rows (Plotly, px)",
        min_value=0, max_value=200,
        value=70, step=10, key="row_gap_px"
    )
    
    st.number_input(
        "h-gap (Plotly, 0…1)",
        min_value=0.0, max_value=0.2,
        value=st.session_state.get("hspace_frac", DEFAULT_HSPACE_FRAC),
        step=0.01, key="hspace_frac"
    )

    st.number_input(
        "Row height per subplot row (Matplotlib, inches)",
        min_value=2.0, max_value=10.0,
        value=st.session_state.get("row_h_in", DEFAULT_ROW_HEIGHT_IN),
        step=1.0, key="row_h_in"
    )
    
    

with st.sidebar.expander("Plot Options"):
    st.radio(
        "Choose plot type",
        options=["Line Plot", "Violin Plot"],
        index=0,
        key="plot_type_selector2"
    )
    if st.session_state.plot_type_selector2 == "Line Plot":
        st.number_input(
            "Max vertices to display in plots",
            min_value=1,
            max_value=total_vertices_available,
            value=min(5, total_vertices_available),
            step=1,
            key="max_plot_vertices"
        )
    
    st.checkbox("Show convex combinations in all plots", value=True, key="show_convex")
    st.checkbox("Show original flexibility ranges (red shaded)", value=False, key="show_original_ranges")
    st.checkbox("Enable density plots", key="show_density")  

with st.sidebar.expander("Additional Metrics"):
    st.radio(
        "Plot type for additional metrics",
        ["Violin Plot", "Scatter Plot"],
        index=0,
        key="plot_type_selector"
    )

# === Different Tabs ===
tab1, tab2,tab3 = st.tabs(["📊 Decision Tool", "❓ Explanation","📥 Download Results"])

# === Auswahl & Filter-UI ===
with tab1: 
    if st.session_state["layout_mode"] == "Two-column layout":
        ratio = st.session_state.get("column_ratio")
        col1, spacer, col2 = st.columns([0.5+ratio, 0.1, 1.5-ratio])
        with col1:
            st.markdown("### Select and Filter Technologies")
            st.caption(f"⚡️ **Note:** Proceed sequential.")
            col_select, col_reset = st.columns([4, 1])
            with col_select:
                all_filter_options = technologies + additional_cols
                selected_techs_raw = st.multiselect("Select variables to be constrained", all_filter_options)
                ordered_techs = selected_techs_raw.copy()
        
            with col_reset:
                st.markdown("""
                    <style>
                        div[data-testid="stButton"] > button {
                            padding: 0.3rem 0.6rem;
                            font-size: 0.75rem;
                        }
                    </style>
                """, unsafe_allow_html=True)
                if st.button("🔄 Reset"):
                    for key in list(st.session_state.keys()):
                        if key.startswith("slider_"):
                            del st.session_state[key]
                    st.rerun()
        
            filtered_data = pd.DataFrame()
            convex_data = pd.DataFrame()
        
            selected_techs = [t for t in ordered_techs if t in technologies]
            selected_metrics = [m for m in ordered_techs if m in additional_cols]
        
            selected_data = pd.DataFrame(index=tech_data.index)
            
            if selected_techs:
                
                selected_data = pd.concat([selected_data, tech_data[[f"{MAA_PREFIX}{t}" for t in selected_techs]]], axis=1)
            if selected_metrics:
                selected_data = pd.concat([selected_data, vertex_df.loc[tech_data.index, selected_metrics]], axis=1)
        
            current_indices = selected_data.index
            for i, tech in enumerate(ordered_techs):
                key = f"slider_{tech}"
                col = f"{MAA_PREFIX}{tech}" if tech in technologies else tech
        
                partial_indices = selected_data.index
                for j in range(i):
                    prev = ordered_techs[j]
                    prev_col = f"{MAA_PREFIX}{prev}" if prev in technologies else prev
                    prev_range = st.session_state.get(f"slider_{prev}", (selected_data[prev_col].min(), selected_data[prev_col].max()))
                    partial_indices = partial_indices[
                        (selected_data.loc[partial_indices, prev_col] >= prev_range[0]) &
                        (selected_data.loc[partial_indices, prev_col] <= prev_range[1])
                    ]
        
                valid_values = selected_data.loc[partial_indices, col].dropna()
                overall_min = selected_data[col].min()
                overall_max = selected_data[col].max()
        
                missing_slider = any(
                    f"slider_{ordered_techs[j]}" not in st.session_state
                    or st.session_state[f"slider_{ordered_techs[j]}"] is None
                    for j in range(i)
                )
                if missing_slider:
                    st.info(f"➡️ Please configure previous sliders to activate **{tech}**.")
                    st.slider(f"{tech}", float(overall_min), float(overall_max),
                              (float(overall_min), float(overall_max)), key=key, disabled=True)
                    continue
        
                if valid_values.empty:
                    st.warning(f"⚠️ No valid vertices remaining for {tech}.")
                    st.slider(f"{tech}", float(overall_min), float(overall_max),
                              (float(overall_min), float(overall_max)), key=key, disabled=True)
                    continue
        
                min_val = valid_values.min()
                max_val = valid_values.max()
                if min_val == max_val:
                    st.info(f"**{tech}**: No decision flexibility (constant value: {min_val:.2f})")
                    st.session_state[key] = (min_val, max_val)
                    current_indices = current_indices[
                        (selected_data.loc[current_indices, col] >= min_val) &
                        (selected_data.loc[current_indices, col] <= max_val)
                    ]
                    continue
        
                default_val = (float(min_val), float(max_val))
                value = st.session_state.get(key, default_val)
                value = tuple(map(float, value))
        
                slider_value = st.slider(
                    f"{tech}",
                    float(overall_min),
                    float(overall_max),
                    value=value,
                    step=0.01
                )
                clipped_range = (max(min_val, slider_value[0]), min(max_val, slider_value[1]))
        
                if slider_value != clipped_range:
                    st.warning(f"⚠️ Selection for {tech} exceeds valid range ({min_val:.1f}–{max_val:.1f}). Resetting.")
                    if key in st.session_state:
                        del st.session_state[key]
                    st.rerun()
        
                st.session_state[key] = clipped_range
                current_indices = current_indices[
                    (selected_data.loc[current_indices, col] >= clipped_range[0]) &
                    (selected_data.loc[current_indices, col] <= clipped_range[1])
                ]
        
            filtered_data = selected_data.loc[current_indices] if ordered_techs else pd.DataFrame(index=current_indices)
        
            with st.sidebar.expander("Convex Combination Settings"):
                st.number_input("Total number of convex combinations", min_value=10, max_value=10000, value=100, step=10, key="n_samples")
                max_vertices = len(current_indices) if not current_indices.empty else 0
                st.number_input("Vertices used per combination", min_value=2, max_value=max_vertices if max_vertices > 1 else 2,
                                value=max_vertices if max_vertices > 1 else 2, step=1, key="n_vertices_convex")
                st.selectbox("Dirichlet α (weight spread)", [0.01, 0.1, 1.0],
                             index=[0.01, 0.1, 1.0].index(st.session_state.get('alpha_value', 0.1)), key="alpha_value")
                st.number_input("Combinations per batch", min_value=1, max_value=st.session_state["n_samples"],
                                value=min(10, st.session_state["n_samples"]), step=1, key="n_batch_size")
        
                col_gen, col_reset = st.columns(2)
                with col_gen:
                    generate_convex = st.button("Generate", key="generate_convex_button_sidebar")
                with col_reset:
                    reset_convex = st.button("Reset", key="reset_convex_button_sidebar")
        
                if reset_convex:
                    st.session_state['convex_combinations'] = pd.DataFrame()
                    st.session_state['convex_additional'] = pd.DataFrame()
        
                if generate_convex and not current_indices.empty:
                    base_data_full = tech_data.loc[current_indices]
                    base_additional_data = vertex_df.loc[current_indices, additional_cols] if additional_cols else pd.DataFrame(index=current_indices)
                    include_installed_capacity = MAA_PREFIX == "VALUE_"
                    installed_data = pd.DataFrame(index=current_indices)
        
                    if include_installed_capacity:
                        installed_cols = [col for col in vertex_df.columns if col.startswith(INSTALLED_CAPACITY_PREFIX)]
                        installed_data = vertex_df.loc[current_indices, installed_cols]
        
                    n_total = st.session_state["n_samples"]
                    batch_size = st.session_state["n_batch_size"]
                    n_vertices = st.session_state["n_vertices_convex"]
                    alpha = st.session_state["alpha_value"]
        
                    all_samples = []
                    all_additional_samples = []
                    all_installed_samples = [] if include_installed_capacity else None
                    n_batches = int(np.ceil(n_total / batch_size))
        
                    for _ in range(n_batches):
                        base_sample = base_data_full.sample(
                            n=n_vertices if len(base_data_full) > n_vertices else len(base_data_full),
                            random_state=np.random.randint(0, 999999)
                        )
                        base_additional_sample = base_additional_data.loc[base_sample.index] if not base_additional_data.empty else pd.DataFrame(index=base_sample.index)
                        effective_batch_size = min(batch_size, n_total - len(all_samples) * batch_size)
                        weights = np.random.dirichlet([alpha] * len(base_sample), size=effective_batch_size)
                        batch_samples = weights @ base_sample.values
        
                        if include_installed_capacity and not installed_data.empty:
                            installed_sample = installed_data.loc[base_sample.index]
                            batch_installed = weights @ installed_sample.values
                            all_installed_samples.append(pd.DataFrame(batch_installed, columns=installed_sample.columns))
        
                        all_samples.append(pd.DataFrame(batch_samples, columns=base_sample.columns))
        
                        if not base_additional_sample.empty:
                            batch_additional = weights @ base_additional_sample.values
                            all_additional_samples.append(pd.DataFrame(batch_additional, columns=base_additional_sample.columns))
        
                        if sum(len(df) for df in all_samples) >= n_total:
                            break
        
                    convex_df = pd.concat(all_samples, ignore_index=True)
                    st.session_state['convex_combinations'] = pd.concat([st.session_state['convex_combinations'], convex_df], ignore_index=True)
        
                    if include_installed_capacity and all_installed_samples:
                        installed_comb_df = pd.concat(all_installed_samples, ignore_index=True)
                        st.session_state['convex_combinations'][installed_data.columns] = installed_comb_df
        
                    if all_additional_samples:
                        additional_comb_df = pd.concat(all_additional_samples, ignore_index=True)
                        st.session_state['convex_additional'] = pd.concat(
                            [st.session_state.get('convex_additional', pd.DataFrame()), additional_comb_df], ignore_index=True
                        )
        
                    n_convex = len(st.session_state['convex_combinations'])
                    st.sidebar.info(f"**Currently {n_convex} convex combination(s)** generated.")
        
            filtered_convex_data, filtered_convex_additional = apply_tech_filters(
                st.session_state['convex_combinations'],
                st.session_state['convex_additional'],
                st.session_state,
                ordered_techs,
                prefix=MAA_PREFIX
            )
           
            time_map, valid_techs, plot_indices, tech,cols = prepare_vertex_selection(
                MAA_PREFIX=MAA_PREFIX,
                vertex_df=vertex_df,
                tech_time_map=tech_time_map,
                extract_time_series_map=extract_time_series_map,
                select_representative_vertices_by_kmeans=select_representative_vertices_by_kmeans,
                current_indices=current_indices,
                max_plot_vertices=st.session_state["max_plot_vertices"],
                st=st,
                mode="operational"
            )

            # === Repräsentative konvexe Vertices (für VALUE_ operational) vorab clustern ===
            st.session_state["convex_cluster_indices"] = []
            if st.session_state.get("show_convex") and filtered_convex_data is not None and not filtered_convex_data.empty:
                try:
                    # 'cols' kommt aus prepare_vertex_selection(..., mode="operational")
                    conv_cols_for_kmeans = [c for c in list(cols) if c in filtered_convex_data.columns]
                    if conv_cols_for_kmeans:
                        st.session_state["convex_cluster_indices"] = select_representative_vertices_by_kmeans(
                            df=filtered_convex_data,
                            cols=conv_cols_for_kmeans,
                            n_vertices=5
                        )
                except Exception as e:
                    st.warning(f"⚠️ Could not cluster convex combinations: {e}")
            
            if valid_techs:
                st.markdown("---")
                st.markdown("### Highlight Vertex & View Details")
                selected_vertex = select_and_show_vertex_info(
                    plot_indices=plot_indices,
                    current_indices=current_indices,
                    vertex_df=vertex_df,
                    tech_data=tech_data,
                    additional_cols=additional_cols,
                    convex_plot_indices=st.session_state.get("convex_cluster_indices", []),    # NEU
                    convex_additional=filtered_convex_additional                               # NEU
                )
                st.session_state["selected_vertex"] = (
                    selected_vertex if selected_vertex != "— Please select —" else None
                )
            else:
                selected_vertex = None
            
        
        with col2:        
            if MAA_PREFIX == "VALUE_":
                
                st.markdown("### Operational Variables Over Time")
                if len(current_indices) > st.session_state["max_plot_vertices"]:
                    
                    try:
                        plot_indices_val = select_representative_vertices_by_kmeans(
                            df=vertex_df,
                            cols=list(cols),  # <- sicherstellen, dass die Spalten existieren
                            n_vertices=st.session_state["max_plot_vertices"],
                            index_subset=current_indices
                        )
                        st.caption(
                            f"⚡️ **Note:** Displaying a clustered sample of "
                            f"{st.session_state['max_plot_vertices']} out of {len(current_indices)} valid vertices."
                        )
                    except Exception as e:
                        st.error(f"❌ Fehler in KMeans: {e}")
                        st.stop()  # stoppt die Ausführung, damit nichts Falsches weiterläuft
                else:
                    plot_indices_val = current_indices
                    st.caption(f"⚡️ **Note:** {len(current_indices)} valid vertices remaining.")
                value_time_map = extract_time_series_map(vertex_df, MAA_PREFIX, mode="operational")
                
                valid_techs_value = sorted([tech for tech, v in value_time_map.items() if len(v) >= 1])
                if st.session_state.get("plot_type_selector2") == "Line Plot":
                                        
                    plot_operational_variables_over_time(
                        vertex_df=vertex_df,
                        current_indices=current_indices,
                        plot_indices_val=plot_indices_val,
                        time_column_map=value_time_map,
                        selected_vertex=st.session_state.get("selected_vertex"),
                        n_cols_val=st.session_state.get("n_cols_plots", 3),
                        show_convex=st.session_state["show_convex"],
                        st_convex=st.session_state["convex_combinations"],
                        filtered_convex_data=filtered_convex_data,
                        show_original_ranges=st.session_state["show_original_ranges"],
                        max_plot_vertices=st.session_state["max_plot_vertices"],
                        maa_prefix=MAA_PREFIX,
                        apply_prefix=True,
                        plot_title="Operational Variables Over Time",
                        convex_cluster_indices=st.session_state.get("convex_cluster_indices", [])   # << NEU
                    )


                else:
                    plot_violin_values(
                        vertex_df,
                        valid_techs_value,
                        value_time_map,
                        plot_indices_val,
                        current_indices,
                        filtered_convex_data,
                        MAA_PREFIX="MAA_"  # optional
                    )

                # === Installed Capacities Plot ===
            st.markdown("### Installed Capacities Over Time")

            # === Auswahl Mapping je nach Prefix
            source_map = extract_time_series_map(vertex_df, MAA_PREFIX, mode="installed")
                        
            # === Gültige Technologien + Auswahl
            valid_techs_all = sorted([tech for tech, v in source_map.items() if len(v) >= 1])
                        
            selected_techs = st.multiselect(
                "Select technologies to display:",
                options=valid_techs_all,
                default=valid_techs_all,
                help="Select one or more technologies to display."
            )
            
            valid_techs = selected_techs
            
            # << neu: nur selektierte Techs in eigener Map
            source_map_selected = {tech: source_map.get(tech, []) for tech in valid_techs}
            
            # === Initiale Spaltenauswahl für KMeans
            if MAA_PREFIX == "VALUE_":
                cols = build_cols_from_time_map(extract_time_series_map(vertex_df, MAA_PREFIX, mode="installed"), valid_techs_all, MAA_PREFIX)
            else:
                
                cols = build_cols_from_time_map(extract_time_series_map(vertex_df,MAA_PREFIX), valid_techs_all, MAA_PREFIX)
            
            if not cols:
                st.error("❌ No matching columns found.")
                st.stop()
            
            # === Data Check vor KMeans
            subset_df = vertex_df.loc[current_indices, cols]
            
            
            # === Auswahl von Vertices mittels KMeans mit Fehlerbehandlung
            if MAA_PREFIX == "VALUE_":
                plot_indices=plot_indices_val
                if len(current_indices) > st.session_state["max_plot_vertices"] and st.session_state.get("plot_type_selector2") == "Line Plot":
                    st.caption(f"⚡️ **Note:** Displaying a clustered sample of {st.session_state['max_plot_vertices']} out of {len(current_indices)} valid vertices.")
                else:
                    st.caption(f"⚡️ **Note:** {len(current_indices)} valid vertices remaining.")
            
            else:
                try:
                    raw_plot_indices = select_representative_vertices_by_kmeans(
                        df=vertex_df,
                        cols=cols,
                        n_vertices=st.session_state["max_plot_vertices"],
                        index_subset=current_indices
                    )
                    plot_indices = clean_plot_indices(raw_plot_indices)
                    if len(current_indices) > st.session_state["max_plot_vertices"] and st.session_state.get("plot_type_selector2") == "Line Plot":
                        st.caption(f"⚡️ **Note:** Displaying a clustered sample of {st.session_state['max_plot_vertices']} out of {len(current_indices)} valid vertices.")
                    else:
                        st.caption(f"⚡️ **Note:** {len(current_indices)} valid vertices remaining.")
                except Exception as e:
                    st.exception(f"❌ Error selecting vertices using KMeans: {e}")
                    st.stop()
                            
            # === Plot-Erstellung
            if st.session_state.get("plot_type_selector2") == "Line Plot":
                
            
                plot_operational_variables_over_time(
                    vertex_df=vertex_df,
                    current_indices=current_indices,
                    plot_indices_val=plot_indices,
                    time_column_map=source_map_selected,  # oder source_map
                    selected_vertex=st.session_state.get("selected_vertex"),
                    n_cols_val=st.session_state.get("n_cols_plots", 3),
                    show_convex=st.session_state["show_convex"],
                    st_convex=st.session_state["convex_combinations"],
                    filtered_convex_data=filtered_convex_data,
                    show_original_ranges=st.session_state.get("show_original_ranges", False),
                    apply_prefix=False,
                    plot_title="Installed Capacities Over Time",
                    convex_cluster_indices=st.session_state.get("convex_cluster_indices", [])  # << NEU
                )
            
            else:
                plot_violin_values(
                    vertex_df,
                    valid_techs,            # << selektierte Techs
                    source_map_selected,    # << nur selektierte Techs
                    plot_indices,
                    current_indices,
                    filtered_convex_data,
                    MAA_PREFIX="MAA_"  # optional
                )
                
            
            # === Dichteplots: Kernel Density Estimation über Zeitverläufe ===
            
            
            if st.session_state.get("show_density"):
                st.divider()
                plot_density_contours(
                    tech_time_map=tech_time_map,
                    vertex_df=vertex_df,
                    current_indices=current_indices
                )
                
        
    else:
       
        st.markdown("### Select and Filter Technologies")
        
        col_select, col_reset = st.columns([4, 1])
        with col_select:
            all_filter_options = technologies + additional_cols
            selected_techs_raw = st.multiselect("Select variables to be constrained", all_filter_options)
            ordered_techs = selected_techs_raw.copy()
    
        with col_reset:
            st.markdown("""
                <style>
                    div[data-testid="stButton"] > button {
                        padding: 0.3rem 0.6rem;
                        font-size: 0.75rem;
                    }
                </style>
            """, unsafe_allow_html=True)
            if st.button("🔄 Reset"):
                for key in list(st.session_state.keys()):
                    if key.startswith("slider_"):
                        del st.session_state[key]
                st.rerun()
    
        filtered_data = pd.DataFrame()
        convex_data = pd.DataFrame()
    
        # Separiere Technologien und Metriken
        selected_techs = [t for t in ordered_techs if t in technologies]
        selected_metrics = [m for m in ordered_techs if m in additional_cols]
        
        # Basisdaten vorbereiten
        selected_data = pd.DataFrame(index=tech_data.index)
        if selected_techs:
            selected_data = pd.concat([selected_data, tech_data[[f"{MAA_PREFIX}{t}" for t in selected_techs]]], axis=1)
        if selected_metrics:
            selected_data = pd.concat([selected_data, vertex_df.loc[tech_data.index, selected_metrics]], axis=1)
        current_indices = selected_data.index
        for i, tech in enumerate(ordered_techs):
            key = f"slider_{tech}"
            col = f"{MAA_PREFIX}{tech}" if tech in technologies else tech
        
            # Vorherige Einschränkungen anwenden
            partial_indices = selected_data.index
            for j in range(i):
                prev = ordered_techs[j]
                prev_col = f"{MAA_PREFIX}{prev}" if prev in technologies else prev
                prev_range = st.session_state.get(f"slider_{prev}", (selected_data[prev_col].min(), selected_data[prev_col].max()))
                partial_indices = partial_indices[
                    (selected_data.loc[partial_indices, prev_col] >= prev_range[0]) &
                    (selected_data.loc[partial_indices, prev_col] <= prev_range[1])
                ]
        
            valid_values = selected_data.loc[partial_indices, col].dropna()
            overall_min = selected_data[col].min()
            overall_max = selected_data[col].max()
    
            missing_slider = any(
                f"slider_{ordered_techs[j]}" not in st.session_state
                or st.session_state[f"slider_{ordered_techs[j]}"] is None
                for j in range(i)
            )
            if missing_slider:
                st.info(f"➡️ Please configure previous sliders to activate **{tech}**.")
                st.slider(f"{tech}", float(overall_min), float(overall_max),
                          (float(overall_min), float(overall_max)), key=key, disabled=True)
                continue
    
            if valid_values.empty:
                st.warning(f"⚠️ No valid vertices remaining for {tech}.")
                st.slider(f"{tech}", float(overall_min), float(overall_max),
                          (float(overall_min), float(overall_max)), key=key, disabled=True)
                continue
    
            min_val = valid_values.min()
            max_val = valid_values.max()
            if min_val == max_val:
                st.info(f"**{tech}**: No decision flexibility (constant value: {min_val:.2f})")
                st.session_state[key] = (min_val, max_val)
                current_indices = current_indices[
                    (selected_data.loc[current_indices, col] >= min_val) &
                    (selected_data.loc[current_indices, col] <= max_val)
                ]
                continue
    
            default_val = (float(min_val), float(max_val))
            value = st.session_state.get(key, default_val)
            value = tuple(map(float, value))
    
            slider_value = st.slider(
                f"{tech}",
                float(overall_min),
                float(overall_max),
                value=value,
                step=0.01
            )
            clipped_range = (max(min_val, slider_value[0]), min(max_val, slider_value[1]))
    
            if slider_value != clipped_range:
                st.warning(f"⚠️ Selection for {tech} exceeds valid range ({min_val:.1f}–{max_val:.1f}). Resetting.")
                if key in st.session_state:
                    del st.session_state[key]
                st.rerun()
    
            st.session_state[key] = clipped_range
            current_indices = current_indices[
                (selected_data.loc[current_indices, col] >= clipped_range[0]) &
                (selected_data.loc[current_indices, col] <= clipped_range[1])
            ]
    
        filtered_data = selected_data.loc[current_indices] if ordered_techs else pd.DataFrame(index=current_indices)
        # === Konvexe Kombinationen ===
        with st.sidebar.expander("Convex Combination Settings"):
            st.number_input(
                "Total number of convex combinations",
                min_value=10,
                max_value=10000,
                value=100,
                step=10,
                key="n_samples"
            )
        
            max_vertices = len(current_indices) if not current_indices.empty else 0
            st.number_input(
                "Vertices used per combination",
                min_value=2,
                max_value=max_vertices if max_vertices > 1 else 2,
                value=max_vertices if max_vertices > 1 else 2,
                step=1,
                key="n_vertices_convex"
            )
        
            st.selectbox(
                "Dirichlet α (weight spread)",
                [0.01, 0.1, 1.0],
                index=[0.01, 0.1, 1.0].index(st.session_state.get('alpha_value', 0.1)),
                key="alpha_value"
            )
        
            st.number_input(
                "Combinations per batch",
                min_value=1,
                max_value=st.session_state["n_samples"],
                value=min(10, st.session_state["n_samples"]),
                step=1,
                key="n_batch_size"
            )
        
            col_gen, col_reset = st.columns(2)
            with col_gen:
                generate_convex=st.button("Generate", key="generate_convex_button_sidebar")
            with col_reset:
                reset_convex=st.button("Reset", key="reset_convex_button_sidebar")
            
            if reset_convex:
                st.session_state['convex_combinations'] = pd.DataFrame()
                st.session_state['convex_additional'] = pd.DataFrame()
        
            if generate_convex and not current_indices.empty:
                base_data_full = tech_data.loc[current_indices]
                base_additional_data = vertex_df.loc[current_indices, additional_cols] if additional_cols else pd.DataFrame(index=current_indices)
        
                # Nur wenn MAA_PREFIX == "VALUE_", auch INSTALLED_CAPACITY einbeziehen
                include_installed_capacity = MAA_PREFIX == "VALUE_"
                installed_data = pd.DataFrame(index=current_indices)
        
                if include_installed_capacity:
                    installed_cols = [col for col in vertex_df.columns if col.startswith(INSTALLED_CAPACITY_PREFIX)]
                    installed_data = vertex_df.loc[current_indices, installed_cols]
                n_total = st.session_state["n_samples"]
                batch_size = st.session_state["n_batch_size"]
                n_vertices = st.session_state["n_vertices_convex"]
                alpha = st.session_state["alpha_value"]
        
                all_samples = []
                all_additional_samples = []
                all_installed_samples = [] if include_installed_capacity else None
                n_batches = int(np.ceil(n_total / batch_size))
        
                for _ in range(n_batches):
                    base_sample = base_data_full.sample(
                        n=n_vertices if len(base_data_full) > n_vertices else len(base_data_full),
                        random_state=np.random.randint(0, 999999)
                    )
        
                    base_additional_sample = base_additional_data.loc[base_sample.index] if not base_additional_data.empty else pd.DataFrame(index=base_sample.index)
        
                    effective_batch_size = min(batch_size, n_total - len(all_samples) * batch_size)
                    weights = np.random.dirichlet([alpha] * len(base_sample), size=effective_batch_size)
        
                    batch_samples = weights @ base_sample.values
                    if include_installed_capacity and not installed_data.empty and MAA_PREFIX == "VALUE_":
                        installed_sample = installed_data.loc[base_sample.index]
                        batch_installed = weights @ installed_sample.values
                        all_installed_samples.append(pd.DataFrame(batch_installed, columns=installed_sample.columns))
                    elif include_installed_capacity and not installed_data.empty:
                        batch_installed = weights @ installed_data.values
                        all_installed_samples.append(pd.DataFrame(batch_installed, columns=installed_data.columns))
                    all_samples.append(pd.DataFrame(batch_samples, columns=base_sample.columns))
        
                    if not base_additional_sample.empty:
                        batch_additional = weights @ base_additional_sample.values
                        all_additional_samples.append(pd.DataFrame(batch_additional, columns=base_additional_sample.columns))
        
                    if sum(len(df) for df in all_samples) >= n_total:
                        break
        
                convex_df = pd.concat(all_samples, ignore_index=True)
                st.session_state['convex_combinations'] = pd.concat(
                    [st.session_state['convex_combinations'], convex_df],
                    ignore_index=True
                )
                if include_installed_capacity and all_installed_samples:
                    installed_comb_df = pd.concat(all_installed_samples, ignore_index=True)
                    st.session_state['convex_combinations'][installed_data.columns] = installed_comb_df
        
                if all_additional_samples:
                    additional_comb_df = pd.concat(all_additional_samples, ignore_index=True)
                    st.session_state['convex_additional'] = pd.concat(
                        [st.session_state.get('convex_additional', pd.DataFrame()), additional_comb_df],
                        ignore_index=True
                    )
        
                n_convex = len(st.session_state['convex_combinations'])
                st.sidebar.info(f"**Currently {n_convex} convex combination(s)** generated.")
    
        # === Konvexe Kombinationen filtern ===
        filtered_convex_data,filtered_convex_additional = apply_tech_filters(
            st.session_state['convex_combinations'],
            st.session_state['convex_additional'],
            st.session_state,
            ordered_techs,
            prefix=MAA_PREFIX
        )
    
        
       
        # === Highlight ===
        time_map, valid_techs, plot_indices, tech, cols = prepare_vertex_selection(
            MAA_PREFIX=MAA_PREFIX,
            vertex_df=vertex_df,
            tech_time_map=tech_time_map,
            extract_time_series_map=extract_time_series_map,
            select_representative_vertices_by_kmeans=select_representative_vertices_by_kmeans,
            current_indices=current_indices,
            max_plot_vertices=st.session_state["max_plot_vertices"],
            st=st,
            mode="operational"  # << hinzufügen
        )

        # === Repräsentative konvexe Vertices (für VALUE_ operational) vorab clustern ===
        st.session_state["convex_cluster_indices"] = []
        if st.session_state.get("show_convex") and filtered_convex_data is not None and not filtered_convex_data.empty:
            try:
                conv_cols_for_kmeans = [c for c in list(cols) if c in filtered_convex_data.columns]
                if conv_cols_for_kmeans:
                    st.session_state["convex_cluster_indices"] = select_representative_vertices_by_kmeans(
                        df=filtered_convex_data,
                        cols=conv_cols_for_kmeans,
                        n_vertices=5
                    )
            except Exception as e:
                st.warning(f"⚠️ Could not cluster convex combinations: {e}")

        if valid_techs:
            st.markdown("---")
            st.markdown("### Highlight Vertex & View Details")
            selected_vertex = select_and_show_vertex_info(
                plot_indices=plot_indices,
                current_indices=current_indices,
                vertex_df=vertex_df,
                tech_data=tech_data,
                additional_cols=additional_cols,
                convex_plot_indices=st.session_state.get("convex_cluster_indices", []),    # NEU
                convex_additional=filtered_convex_additional                               # NEU
            )
            st.session_state["selected_vertex"] = (
                selected_vertex if selected_vertex != "— Please select —" else None
            )
        else:
            selected_vertex = None
        
        st.divider()
        
        
        if MAA_PREFIX == "VALUE_":
            
            st.markdown("### Operational Variables Over Time")

            value_time_map = extract_time_series_map(vertex_df, MAA_PREFIX, mode="operational")
            valid_techs_value = sorted([tech for tech, v in value_time_map.items() if len(v) >= 1])
            
            if len(current_indices) > st.session_state["max_plot_vertices"]:
                    plot_indices_val = select_representative_vertices_by_kmeans(
                        df=vertex_df,
                        cols=list(cols),  # <- Stelle sicher, dass `cols` korrekt definiert ist
                        n_vertices=st.session_state["max_plot_vertices"],
                        index_subset=current_indices
                    )
                    st.caption(f"⚡️ **Note:** Displaying a clustered sample of {st.session_state['max_plot_vertices']} out of {len(current_indices)} valid vertices.")
            else:
                plot_indices_val = current_indices
                st.caption(f"⚡️ **Note:** {len(current_indices)} valid vertices remaining.")
                
            if st.session_state.get("plot_type_selector2") == "Line Plot":
                
                plot_operational_variables_over_time(
                    vertex_df=vertex_df,
                    current_indices=current_indices,
                    plot_indices_val=plot_indices_val,
                    time_column_map=value_time_map,
                    selected_vertex=st.session_state.get("selected_vertex"),
                    n_cols_val=st.session_state.get("n_cols_plots", 3),
                    show_convex=st.session_state["show_convex"],
                    st_convex=st.session_state["convex_combinations"],
                    filtered_convex_data=filtered_convex_data,
                    show_original_ranges=st.session_state["show_original_ranges"],
                    max_plot_vertices=st.session_state["max_plot_vertices"],
                    maa_prefix=MAA_PREFIX,
                    apply_prefix=True,
                    plot_title="Operational Variables Over Time",
                    convex_cluster_indices=st.session_state.get("convex_cluster_indices", [])   # << NEU
                )


            else:
                plot_violin_values(
                    vertex_df,
                    valid_techs_value,
                    value_time_map,
                    plot_indices_val,
                    current_indices,
                    filtered_convex_data,
                    MAA_PREFIX="MAA_"  # optional
                )
                
        st.markdown("### Installed Capacities Over Time")
        source_map = extract_time_series_map(vertex_df, MAA_PREFIX, mode="installed")
                        
        # === Gültige Technologien + Auswahl
        valid_techs_all = sorted([tech for tech, v in source_map.items() if len(v) >= 1])
                    
        selected_techs = st.multiselect(
            "Select technologies to display:",
            options=valid_techs_all,
            default=valid_techs_all,
            help="Select one or more technologies to display."
        )
        
        valid_techs = selected_techs          
        
        # === Initiale Spaltenauswahl für KMeans
        if MAA_PREFIX == "VALUE_":
            cols = []
            for tech in valid_techs:
                year_cols = source_map.get(tech, [])
                cols.extend([col for _, col in year_cols])
            cols = list(set(cols))  # Doppelte entfernen
        else:
            # Fallback: Nur erste Technologie wie gehabt
            year_cols = source_map[valid_techs_all[0]]
            years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
            try:
                _, cols = zip(*years_cols_sorted)
                cols = list(cols)
               
            except ValueError:
                st.error("❌ Could not extract columns from years_cols_sorted.")
                st.stop()
        
        if not cols:
            st.error("❌ No matching columns found.")
            st.stop()
        
        # === Data Check vor KMeans
        subset_df = vertex_df.loc[current_indices, cols]
        
        
        # === Auswahl von Vertices mittels KMeans mit Fehlerbehandlung
        if MAA_PREFIX == "VALUE_":
            plot_indices=plot_indices_val
            if len(current_indices) > st.session_state["max_plot_vertices"] and st.session_state.get("plot_type_selector2") == "Line Plot":
                st.caption(f"⚡️ **Note:** Displaying a clustered sample of {st.session_state['max_plot_vertices']} out of {len(current_indices)} valid vertices.")
            else:
                st.caption(f"⚡️ **Note:** {len(current_indices)} valid vertices remaining.")

        
        else:
            try:
                raw_plot_indices = select_representative_vertices_by_kmeans(
                    df=vertex_df,
                    cols=cols,
                    n_vertices=st.session_state["max_plot_vertices"],
                    index_subset=current_indices
                )
                plot_indices = clean_plot_indices(raw_plot_indices)
                if len(current_indices) > st.session_state["max_plot_vertices"] and st.session_state.get("plot_type_selector2") == "Line Plot":
                    st.caption(f"⚡️ **Note:** Displaying a clustered sample of {st.session_state['max_plot_vertices']} out of {len(current_indices)} valid vertices.")
                else:
                    st.caption(f"⚡️ **Note:** {len(current_indices)} valid vertices remaining.")
            except Exception as e:
                st.exception(f"❌ Error selecting vertices using KMeans: {e}")
                st.stop()
                
        # === Plot-Erstellung
        if st.session_state.get("plot_type_selector2") == "Line Plot":
            st.markdown("### Installed Capacities Over Time")
        
            plot_operational_variables_over_time(
                vertex_df=vertex_df,
                current_indices=current_indices,
                plot_indices_val=plot_indices,
                time_column_map=source_map_selected,  # oder source_map
                selected_vertex=st.session_state.get("selected_vertex"),
                n_cols_val=st.session_state.get("n_cols_plots", 3),
                show_convex=st.session_state["show_convex"],
                st_convex=st.session_state["convex_combinations"],
                filtered_convex_data=filtered_convex_data,
                show_original_ranges=st.session_state.get("show_original_ranges", False),
                apply_prefix=False,
                plot_title="Installed Capacities Over Time",
                convex_cluster_indices=st.session_state.get("convex_cluster_indices", [])  # << NEU
            )

        else:
            plot_violin_values(
                vertex_df,
                valid_techs_all,
                source_map,
                plot_indices,
                current_indices,
                filtered_convex_data,
                MAA_PREFIX="MAA_"  # optional
            )
        # === Dichteplots: Kernel Density Estimation über Zeitverläufe ===
        
        
        if st.session_state.get("show_density"):
            st.divider()
            plot_density_contours(
                tech_time_map=tech_time_map,
                vertex_df=vertex_df,
                current_indices=current_indices
            )
           
        
    st.divider()
    # === Weitere Metriken anzeigen ===
    # Beispiel-Daten
    st.markdown("### Additional Metrics")

    if additional_cols:
        # === Trennung in Jahresmetriken und Einzelmetriken ===
        yearly_metrics = [col for col in additional_cols if re.search(r"\b\d{4}\b", col)]
        single_metrics = [col for col in additional_cols if not re.search(r"\b\d{4}\b", col)]
    
        # === Gruppierung der Jahresmetriken nach Basismetrik ===
        base_metric_dict = defaultdict(list)
        for col in yearly_metrics:
            match = re.search(r"\b(\d{4})\b", col)
            if match:
                year = int(match.group(1))
                base_name = re.sub(r"\b\d{4}\b", "", col).strip(" _()-")
                base_metric_dict[base_name].append((year, col))
    
        # === Dropdown für beide Typen
        options_dropdown = (
            [f"🔹 {m}" for m in single_metrics] +
            [f"📈 {base}" for base in sorted(base_metric_dict.keys())]
        )
        default_selection = options_dropdown[:3] if len(options_dropdown) >= 3 else options_dropdown

        selected_combined = st.multiselect(
            "📊 Select metrics to plot",
            options_dropdown,
            default=default_selection
        )
    
        # === Auswahl trennen
        selected_single = [item.replace("🔹 ", "") for item in selected_combined if item.startswith("🔹")]
        selected_base_metrics = [item.replace("📈 ", "") for item in selected_combined if item.startswith("📈")]
    
        total_plots = len(selected_single) + len(selected_base_metrics)
    
        if total_plots > 0:
            max_cols = 3
            n_cols = min(max_cols, total_plots)
            n_rows = -(-total_plots // max_cols)
    
            fig_combined, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
            fig_combined.patch.set_facecolor('#f4f4f4')
            axes = axes.flatten() if total_plots > 1 else [axes]
            ax_idx = 0
    
            # === Einzelmetriken ===
            if selected_single:
                additional_data = vertex_df.loc[tech_data.index, selected_single]
                filtered_additional = additional_data.loc[current_indices]
    
                if st.session_state.get("show_convex") and not filtered_convex_additional.empty:
                    filtered_combined = pd.concat(
                        [filtered_additional, filtered_convex_additional[selected_single]],
                        axis=0
                    )
                else:
                    filtered_combined = filtered_additional
    
                for col in selected_single:
                    ax = axes[ax_idx]
                    ax.set_facecolor('#f0f0f0')
                    values = filtered_combined[col].dropna().values
                    if len(values) == 0:
                        ax.set_visible(False)
                        ax_idx += 1
                        continue
    
                    if st.session_state.get("show_original_ranges", False):
                        try:
                            original_values = vertex_df[col].dropna()
                            omin, omax = original_values.min(), original_values.max()
                            ax.fill_between([1 - 0.4, 1 + 0.4], omin, omax, color=(1.0, 0.0, 0.0, 0.08), zorder=1)
                        except:
                            pass
    
                    if st.session_state.get("plot_type_selector") == "Violin Plot":
                        vp = ax.violinplot([values], positions=[1], showmeans=False, showmedians=True, showextrema=True, widths=0.8)
                        for pc in vp['bodies']:
                            pc.set_facecolor((0.1, 0.4, 0.8, 0.7))
                            pc.set_edgecolor('black')
                            pc.set_alpha(0.7)
                        if 'cmedians' in vp:
                            vp['cmedians'].set_color('black')
                        ax.set_xlim(0.5, 1.5)
                        ax.set_xticks([])
    
                        # Konvexwerte (falls aktiv)
                        if st.session_state.get("show_convex") and not filtered_convex_additional.empty:
                            if col in filtered_convex_additional.columns:
                                cvals = filtered_convex_additional[col].dropna().values
                                ax.scatter([1] * len(cvals), cvals, color='crimson', alpha=0.5, marker='x', label='Convex')
    
                    else:
                        x_vals = [0] * len(values)
                        ax.scatter(x_vals, values, alpha=0.7, color="#444444")
                        if st.session_state.get("show_convex") and not filtered_convex_additional.empty:
                            if col in filtered_convex_additional.columns:
                                cx_vals = [0] * len(filtered_convex_additional[col].dropna())
                                cy_vals = filtered_convex_additional[col].dropna().values
                                ax.scatter(cx_vals, cy_vals, alpha=0.5, color='crimson', marker='x', label='Convex')
                        ax.set_xlim(-0.5, 0.5)
                        ax.set_xticks([])
    
                    ax.set_title(col, fontsize=12)
                    ax.set_ylabel("Value")
                    ax.grid(True, linestyle="--", alpha=0.4)
                    force_montserrat(ax)
                    ax_idx += 1
    
            # === Jahresmetriken ===
            # === Jahresmetriken ===
            for base in selected_base_metrics:
                ax = axes[ax_idx]
                ax.set_facecolor('#f0f0f0')
            
                entries = sorted(base_metric_dict[base])
                years = [year for year, _ in entries]
                columns = [col for _, col in entries]
            
                           
                data = vertex_df.loc[tech_data.index, columns]
                filtered_data = data.loc[current_indices]
            
                # === Konvexe Daten: KEIN current_indices!
                available_cols = [col for col in columns if col in filtered_convex_additional.columns]
                convex_data_available = (
                    st.session_state.get("show_convex") and
                    not filtered_convex_additional.empty and
                    len(available_cols) > 0
                )
                if convex_data_available:
                    filtered_convex_data = filtered_convex_additional[available_cols]  # Keine .loc[current_indices]
            
                if st.session_state.get("plot_type_selector") == "Violin Plot":
                    year_values = [filtered_data[col].dropna().values for col in columns]
            
                    if convex_data_available:
                        convex_year_values = [filtered_convex_data[col].dropna().values for col in available_cols]
            
                    if st.session_state.get("show_original_ranges", False):
                        for i, col in enumerate(columns):
                            try:
                                ovals = vertex_df[col].dropna()
                                omin, omax = ovals.min(), ovals.max()
                                ax.fill_between([i + 0.6, i + 1.4], omin, omax, color=(1.0, 0.0, 0.0, 0.08), zorder=1)
                            except:
                                pass
            
                    vp = ax.violinplot(year_values, positions=range(1, len(years) + 1), showmeans=False, showmedians=True, showextrema=True)
                    for pc in vp['bodies']:
                        pc.set_facecolor((0.1, 0.4, 0.8, 0.7))
                        pc.set_edgecolor('black')
                        pc.set_alpha(0.7)
                    if 'cmedians' in vp:
                        vp['cmedians'].set_color('black')
            
                    if convex_data_available:
                        for i, cvals in enumerate(convex_year_values):
                            ax.scatter([i + 1] * len(cvals), cvals, color='crimson', alpha=0.5, marker='x', label='Convex' if i == 0 else "")
            
                    ax.set_xticks(range(1, len(years) + 1))
                    ax.set_xticklabels([str(y) for y in years])
            
                else:  # Scatter für Jahresmetriken
                    for i, col in enumerate(columns):
                        values = filtered_data[col].dropna().values
                        x_vals = [i + 1] * len(values)
                        ax.scatter(x_vals, values, alpha=0.7, color="#444444")
            
                        if convex_data_available and col in filtered_convex_data.columns:
                            cvals = filtered_convex_data[col].dropna().values
                            cx_vals = [i + 1] * len(cvals)
                            ax.scatter(cx_vals, cvals, alpha=0.5, color='crimson', marker='x')
            
                        if st.session_state.get("show_original_ranges", False):
                            try:
                                ovals = vertex_df[col].dropna()
                                omin, omax = ovals.min(), ovals.max()
                                ax.fill_between([i + 0.8, i + 1.2], omin, omax, color=(1.0, 0.0, 0.0, 0.08))
                            except:
                                pass
            
                    ax.set_xticks(range(1, len(years) + 1))
                    ax.set_xticklabels([str(y) for y in years])
            
                ax.set_title(base, fontsize=12)
                ax.set_xlabel("Year")
                ax.set_ylabel("Value")
                ax.grid(True, linestyle="--", alpha=0.4)
                force_montserrat(ax)
                ax_idx += 1
    
            for i in range(ax_idx, len(axes)):
                axes[i].set_visible(False)
    
            plt.tight_layout()
            st.pyplot(fig_combined)
            st.session_state["stored_figures"].append(("Combined_Metric_Subplots", fig_combined))
    
    else:
        st.info("No numeric columns found after the last 'NEW_CAPACITY' column.")
        
        
    # === Daten als Tabelle anzeigen ===
    st.divider()
    st.markdown("### Show remaining vertices as table")
    
    # === Original-Vertices ===
    st.markdown("#### Original Vertices")
    
    # Falls keine Daten vorhanden sind
    filtered_full_data = tech_data.loc[current_indices] if not current_indices.empty else pd.DataFrame()
    
    if filtered_data.empty:
        st.dataframe(filtered_full_data, use_container_width=True)
    else:
        frames_to_concat = [filtered_full_data]  # Index bleibt erhalten
    
        # Füge Installed Capacity-Spalten hinzu, falls VALUE_-Modus
        if MAA_PREFIX == "VALUE_":
            installed_cols = [col for col in vertex_df.columns if col.startswith(INSTALLED_CAPACITY_PREFIX)]
            installed_part = vertex_df.loc[filtered_full_data.index, installed_cols]
            frames_to_concat.append(installed_part)  # KEIN reset_index
    
        # Füge zusätzliche Metriken hinzu
        if additional_cols:
            additional_metrics_part = vertex_df.loc[filtered_full_data.index, additional_cols]
            frames_to_concat.append(additional_metrics_part)  # KEIN reset_index
    
        # Alles korrekt entlang der Indizes zusammenführen
        full_with_all = pd.concat(frames_to_concat, axis=1)
    
        # Tabelle anzeigen
        st.dataframe(full_with_all, use_container_width=True)

    # === Konvexe Kombinationen ===
    if (
        st.session_state['show_convex'] 
        and not st.session_state['convex_combinations'].empty
    ):
        st.markdown("---")
        st.markdown("#### Convex Combinations")
    
        if filtered_convex_data.empty:
            st.dataframe(filtered_convex_data, use_container_width=True)
        else:
            frames_to_concat = [filtered_convex_data.reset_index(drop=True)]
                
    
            if additional_cols and not filtered_convex_additional.empty:
                additional_convex_part = filtered_convex_additional[additional_cols].reset_index(drop=True)
                    
                frames_to_concat.append(additional_convex_part)
    
            # Check auf doppelte Spaltennamen vor dem concat
            all_columns = pd.concat(frames_to_concat, axis=1).columns
            duplicated_cols = all_columns[all_columns.duplicated()].tolist()
    
            if duplicated_cols:
                st.error(f"❌ Duplicate column names found: {duplicated_cols}")
                # Optional: automatisch duplikate entfernen oder umbenennen
                convex_with_all = pd.concat(frames_to_concat, axis=1)
                convex_with_all = convex_with_all.loc[:, ~convex_with_all.columns.duplicated()]
                st.info("ℹ️ Duplicate columns were automatically removed.")
            else:
                convex_with_all = pd.concat(frames_to_concat, axis=1)
    
            st.markdown("✅ Final combined DataFrame")
            st.dataframe(convex_with_all, use_container_width=True)

with tab2:
    st.markdown("## Overview")

    # =====================
    # SECTION: Overview
    # =====================
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("""
        This tool consists of a structured interface with **three main tabs**:
        
        - **Decision Tool**: Core interactive analysis and filtering of read in excel file  
        - **Explanation**: This overview section explains all components  
        - **Download Results**: Export diagrams and tables as zip file
        
        On the **left side**, a collapsible **sidebar** provides configuration options for layout, plot controls, additional metrics, and convex combination settings.
        """)
    with col2:
        st.markdown("""
        <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 1000px; margin: auto;">
            <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/overview.png" width="100%">
            <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    # =====================
    # SECTION: Decision Tool
    # =====================
    st.markdown("## Decision Tool – Core Functionality")
    
    with st.expander("🎚️ Slider Explanation – Adjust parameters dynamically"):
        col1, col2 = st.columns([1, 1.5])  # Breitenverhältnis: 2/3 Text, 1/3 Bild
        with col1:
            
            st.markdown("""
The filtering system allows users to narrow down the dataset by selecting variables and adjusting value ranges using sliders. To start, a user chooses a technology or metric. For the selected variable, a slider becomes available, letting the user limit its value range.

Filtering is applied sequentially, meaning that each newly added constraint is only applied to the already filtered subset. This ensures consistent and logically ordered restriction of the dataset. For example, limiting one technology first affects all subsequent selections.

Only those vertices that fulfill all active constraints remain valid. These filtered vertices are automatically reflected in the time series plots, additional metric charts, and data tables. This provides a direct visual and tabular update based on the current selection.

At any point, the user can reset all applied filters using the Reset-button. This clears all sliders and restores the original unfiltered dataset, allowing for a fresh start.
    """)
            
        with col2:
            st.markdown("""
                <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 650px; margin: auto;">
                    <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/Slider.png" width="100%">
                    <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
                </div>
                """, unsafe_allow_html=True)
    
    with st.expander("📈 Diagram Explanation – Visual output of changes"):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
        The time series plots display how the currently valid data (vertices) evolve over time for each selected variable. Blue lines represent individual valid vertices, reflecting the outcome of all active filters.
        
        The shaded blue area indicates the full remaining range (min to max) of all valid data points at each point in time. This helps users understand the spread and variability of the current solution space.
        
        Additional plot settings can be customized via the sidebar:
        - **Plot layout** (number of columns)
        - **Maximum number of vertices displayed**: limits the number of individual lines in the plot for readability
        - **Show original range** (in red): displays the full data range before filtering
        - **Show convex combinations**: overlays the convex results in red, if generated
        
        These options help to tailor the visual analysis based on user needs and enable comparison between filtered and unfiltered data distributions.
            """)
        with col2:
            st.markdown("""
                <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 650px; margin: auto;">
                    <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/Installed_cap.png" width="100%">
                    <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
                </div>
                """, unsafe_allow_html=True)
    
    with st.expander("📊 Additional Metrics – Extended insights"):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
                This section visualizes additional numerical metrics that are not part of the main time series data. Users can select which of these metrics to display from a list of available options.
                
                In the sidebar, the following display settings can be adjusted:
                - **Plot type**: choose between a **violin plot** (distribution view) or a **scatter plot** (individual points)
                - **Show convex combinations**: toggle whether convex results are included in the visualizations
                
                    """)
        with col2:
            st.markdown("""
                <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 650px; margin: auto;">
                    <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/add_metrics.png" width="100%">
                    <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
                </div>
                """, unsafe_allow_html=True)
    
    with st.expander("🧾 Tables – Detailed data views"):
        st.markdown("""
            In the tables section, the currently valid data is presented in structured form. The output is divided into two parts:
            
            - **Original vertices**: These are the filtered data points based on all active slider constraints.
            - **Convex combinations**: If generated, these show weighted combinations of the original vertices.
            
            This separation ensures transparency between original solutions and aggregated results. All table data reflects the current filtering state and can be exported for further analysis.
                """)
    st.divider()
    # ---------------------
    # Sidebar as sub-section
    # ---------------------
    st.markdown("### Sidebar Options")
    
    with st.expander(" 🧱 Layout Options – Control layout structure"):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            The layout can be switched between **full-width** and **two-column mode**.  
            If two-column mode is selected, the **relative width of both columns** can be adjusted using a slider.  
            Additionally, users can define how many plots are displayed **side by side** using a column count setting.
            """)
        with col2:
            st.markdown("""
            <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 200px; margin: auto;">
                <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/Layout.png" width="100%">
                <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander(" 📊 Plot Options – Adjust chart types"):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            In this section, several visualization settings can be configured:
    
            - **Number of vertices to display** in plots can be set.  
              _For better performance, it is recommended to show fewer vertices, especially with large datasets._
            - Toggle the **display of convex combinations** in the plots.
            - Enable or disable the **original (unfiltered) data range**, shown as a red shaded area for reference.
            - Toggle the **density plots**, which show the distribution of values over time using contour shading.
            """)
        with col2:
            st.markdown("""
            <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 200px; margin: auto;">
                <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/Plot.png" width="100%">
                <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander(" 📌 Additional Metrics – Toggle metrics visibility"):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            This section visualizes additional numerical metrics that are not part of the main time series data.  
            Users can select which of these metrics to display from a list of available options.
    
            In the sidebar, the following display settings can be adjusted:
            - **Plot type**: choose between a **violin plot** (distribution view) or a **scatter plot** (individual points)
            - **Show convex combinations**: toggle whether convex results are included in the visualizations
            """)
        with col2:
            st.markdown("""
            <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 200px; margin: auto;">
                <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/add_metrics2.png" width="100%">
                <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander(" 🧮 Convex Combination Settings – Weighted model blending"):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            Here you can configure the generation of convex combinations:
    
            - Define the **total number of combinations** to be generated.
            - Set how many **vertices should be combined** per combination.
            - Adjust the **α parameter** of the Dirichlet distribution, which controls the spread of weights.
            - Specify how many combinations should be created **per batch** before a new set of vertices is sampled,  
              until the total number is reached.
            """)
        with col2:
            st.markdown("""
            <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 200px; margin: auto;">
                <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/Konvex.png" width="100%">
                <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
            </div>
            """, unsafe_allow_html=True)
    
    
with tab3:
    st.subheader("📦 Export per Rerun/Capture")

    if st.button("🚀 Export vorbereiten (Rerun & Capturen)"):
        st.session_state["export_bin"] = {}   # leeren
        st.session_state["capture_exports"] = True
        st.rerun()

    # Nach dem Rerun: Flag sofort wieder abschalten (ab jetzt wird wieder normal gerendert)
    if st.session_state.get("capture_exports"):
        st.session_state["capture_exports"] = False

    # Downloads, wenn etwas gesammelt wurde
    if st.session_state["export_bin"]:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        zip_buf = BytesIO()
        with ZipFile(zip_buf, "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
            z.writestr(f"tables/decision_tool_tables_{ts}.xlsx", make_current_excel_bytes())
            for path, (data, _mime) in st.session_state["export_bin"].items():
                z.writestr(path, data)
        zip_buf.seek(0)

        st.download_button(
            "⬇️ Alles als ZIP herunterladen",
            data=zip_buf.getvalue(),
            file_name=f"decision_tool_export_{ts}.zip",
            mime="application/zip"
        )

        with st.expander("Einzel-Downloads"):
            for path, (data, mime) in st.session_state["export_bin"].items():
                st.download_button(
                    f"⬇️ {path}",
                    data=data,
                    file_name=path.split("/")[-1],
                    mime=mime,
                    key=f"dl_{path}"
                )

        if st.button("🧹 Export-Cache leeren"):
            st.session_state["export_bin"] = {}
