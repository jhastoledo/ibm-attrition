import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="IBM HR Attrition",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navegação customizada — renomeia as páginas na sidebar ────
pg = st.navigation({
    "IBM HR Attrition": [
        st.Page("pages/home.py",
                title=" Visão Geral",
                icon="👥",
                default=True),
    ],
    "Análise": [
        st.Page("pages/1_EDA.py",
                title=" EDA",
                icon="📊"),
        st.Page("pages/2_Modelagem.py",
                title=" Modelagem",
                icon="🤖"),
        st.Page("pages/3_SHAP.py",
                title=" Interpretabilidade",
                icon="🔍"),
        st.Page("pages/4_Inferencia.py",
                title=" Predição de Risco",
                icon="🎯"),
    ],
})
pg.run()
