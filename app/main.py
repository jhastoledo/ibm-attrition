import streamlit as st

st.set_page_config(
    page_title="IBM HR Attrition",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS GitHub Dark ───────────────────────────────────────────
st.markdown("""
<style>
/* Fundo principal */
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"]          { background-color: #161b22; border-right: 1px solid #30363d; }
[data-testid="stHeader"]           { background-color: #0d1117; }

/* Textos */
h1, h2, h3, h4, p, label, div    { color: #e6edf3 !important; }
.stMarkdown p                      { color: #8b949e !important; }

/* Métricas */
[data-testid="metric-container"]   { background: #161b22; border: 1px solid #30363d;
                                     border-radius: 8px; padding: 16px; }
/* Botões */
.stButton > button                 { background: #21262d; border: 1px solid #30363d;
                                     color: #e6edf3; border-radius: 6px; }
.stButton > button:hover           { border-color: #58a6ff; color: #58a6ff; }

/* Sidebar nav */
[data-testid="stSidebarNav"] a     { color: #8b949e !important; }
[data-testid="stSidebarNav"] a:hover { color: #e6edf3 !important; }

/* Divisor */
hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <h1 style="font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; color: #e6edf3 !important;">
        👥 IBM HR <span style="color: #58a6ff;">Attrition</span>
    </h1>
    <p style="font-size: 1.1rem; color: #8b949e !important; margin-top: -0.5rem;">
        Predição de turnover com Log. Reg + SHAP — portfólio de Data Science
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

# ── Métricas do modelo ────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Dataset",    "1.470 funcionários")
col2.metric("Features",   "44 features")
col3.metric("ROC-AUC",    "0.784")
col4.metric("Recall",     "0.51")
col5.metric("Threshold",  "0.35")

st.markdown("<br>", unsafe_allow_html=True)

# ── Descrição ─────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
### Sobre o Projeto
Este projeto aplica um pipeline completo de Data Science para prever quais
funcionários têm maior probabilidade de deixar a empresa (*attrition*),
permitindo que o RH tome ações preventivas de retenção.

**Custo estimado de substituição:** 50–200% do salário anual.
Com 237 saídas em 1.470 funcionários, o impacto financeiro é significativo.
""")

with col_b:
    st.markdown("""
### Pipeline
| Etapa | Notebook |
|---|---|
| 📊 Análise Exploratória | NB01 — EDA |
| ⚗️ Feature Engineering | NB02 — 44 features |
| 🤖 Modelagem Baseline | NB03 — 5 modelos |
| 🎯 Tuning Optuna | NB04 — 80 trials |
| 🔍 Interpretabilidade | NB05 — SHAP |
""")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; color: #3d444d !important; font-size: 0.85rem;">
    Desenvolvido por <a href="https://github.com/jhastoledo" style="color: #58a6ff;">Jhonnes Toledo</a>
    &nbsp;|&nbsp;
    <a href="https://linkedin.com/in/jhostoledo" style="color: #58a6ff;">LinkedIn</a>
    &nbsp;|&nbsp; IBM HR Attrition Dataset (Kaggle)
</p>
""", unsafe_allow_html=True)
