"""Funções utilitárias compartilhadas — IBM HR Attrition App."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data" / "processed"
FIG_DIR    = ROOT / "reports" / "figures"
REPORTS    = ROOT / "reports"

TARGET    = "Attrition"
THRESHOLD = 0.35

PALETTE = ['#58a6ff', '#ff7b72', '#3fb950', '#d2a8ff',
           '#ffa657', '#79c0ff', '#ffa198']

GITHUB_DARK_CSS = """
<style>
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"]          { background-color: #161b22;
                                      border-right: 1px solid #30363d; }
[data-testid="stHeader"]           { background-color: #0d1117; }
h1, h2, h3, h4                     { color: #e6edf3 !important; }
.stMarkdown p                      { color: #8b949e !important; }
[data-testid="metric-container"]   { background: #161b22;
                                     border: 1px solid #30363d;
                                     border-radius: 8px; padding: 16px; }
[data-testid="metric-container"] [data-testid="stMetricValue"]
                                   { color: #58a6ff !important; }
.stButton > button                 { background: #21262d;
                                     border: 1px solid #30363d;
                                     color: #e6edf3; border-radius: 6px; }
.stButton > button:hover           { border-color: #58a6ff;
                                     color: #58a6ff; }
hr                                 { border-color: #30363d; }
.stDataFrame                       { border: 1px solid #30363d;
                                     border-radius: 6px; }
</style>
"""

@st.cache_resource
def carregar_pipeline():
    """Pipeline completo — ImbPipeline(preprocessor + RUS + LR)."""
    return joblib.load(MODELS_DIR / "pipeline_final.joblib")

@st.cache_data
def carregar_features():
    """Dataset processado — target já convertido para 0/1."""
    return pd.read_parquet(DATA_DIR / "features.parquet")

@st.cache_data
def carregar_comparativo():
    return pd.read_csv(REPORTS / "comparativo_modelos.csv")
