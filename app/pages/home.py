import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import GITHUB_DARK_CSS, carregar_features, carregar_comparativo

st.markdown(GITHUB_DARK_CSS, unsafe_allow_html=True)

st.markdown("""
<div style="padding:2rem 0 1rem 0;">
    <h1 style="font-size:2.4rem;font-weight:800;color:#e6edf3!important;">
        👥 IBM HR Attrition
        <span style="color:#58a6ff;">Predictor</span>
    </h1>
    <p style="color:#8b949e!important;margin-top:-0.5rem;font-size:1.1rem;">
        Modelo preditivo de turnover voluntário · ImbPipeline + RUS ·
        Logistic Regression · SHAP
    </p>
</div>
<hr style="border-color:#30363d;">
""", unsafe_allow_html=True)

df   = carregar_features()
n_at = int(df['Attrition'].sum())
n_ok = int((df['Attrition'] == 0).sum())

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Funcionários",    f"{len(df):,}")
col2.metric("Attrition (1)",   f"{n_at}")
col3.metric("Sem Attrition",   f"{n_ok}")
col4.metric("Taxa Attrition",  f"{df['Attrition'].mean()*100:.1f}%")
col5.metric("PR-AUC",          "0.532")
col6.metric("Recall",          "78.7%")

st.markdown("<br>", unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
### Sobre o Projeto
Previsão de turnover voluntário usando dados de RH da IBM.
O modelo identifica funcionários com maior risco de saída —
permitindo ações preventivas de retenção.

**Desbalanceamento:** 5.2:1 (16.1% de attrition)
**Solução:** RandomUnderSampler dentro do ImbPipeline —
sem data leakage no cross-validation.
""")

with col_b:
    st.markdown("""
### Pipeline
| Notebook | Etapa |
|---|---|
| 📊 NB01 | EDA · Testes estatísticos |
| ⚗️ NB02 | 2 ColumnTransformers · 5 features derivadas |
| 🤖 NB03 | ImbPipeline · 7 modelos · StratifiedKFold |
| 🎯 NB04 | Optuna 80 trials · ElasticNet |
| 🔍 NB05 | SHAP LinearExplainer |
| 🚀 NB06 | Relatório HTML + Deploy |
""")

st.markdown("### Resultado do Modelo")
col1, col2, col3, col4 = st.columns(4)

for col, titulo, val, cor, desc in [
    (col1, "Campanha aleatória", "Recall=0%", "#ff7b72",
     "Sem modelo — ninguém identificado"),
    (col2, "Threshold=0.5 (padrão)", "Recall=63%", "#ffa657",
     "Default sklearn"),
    (col3, "Threshold=0.35 (ótimo)", "Recall=79%", "#58a6ff",
     "← Maximiza detecção"),
    (col4, "Modelo perfeito", "Recall=100%", "#3fb950",
     "Teto máximo"),
]:
    with col:
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid {cor}44;
                    border-top:3px solid {cor};border-radius:10px;
                    padding:16px;text-align:center;height:140px;">
            <div style="color:#8b949e;font-size:0.8rem;">{titulo}</div>
            <div style="color:{cor};font-weight:800;
                        font-size:1.4rem;margin:8px 0;">{val}</div>
            <div style="color:#8b949e;font-size:0.75rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;
            border-radius:10px;padding:20px;">
    <div style="display:flex;gap:40px;flex-wrap:wrap;">
        <div>
            <div style="color:#58a6ff;font-size:0.8rem;font-weight:700;">
                PREDITOR #1 (SHAP)
            </div>
            <div style="color:#e6edf3;font-size:1rem;margin-top:4px;">
                OverTime — SHAP 0.588 — hora extra aumenta risco
            </div>
        </div>
        <div>
            <div style="color:#58a6ff;font-size:0.8rem;font-weight:700;">
                SURPRESA DO PROJETO
            </div>
            <div style="color:#e6edf3;font-size:1rem;margin-top:4px;">
                Logistic Reg. superou XGBoost e LGBM com Pipeline correto
            </div>
        </div>
        <div>
            <div style="color:#58a6ff;font-size:0.8rem;font-weight:700;">
                THRESHOLD ÓTIMO
            </div>
            <div style="color:#e6edf3;font-size:1rem;margin-top:4px;">
                0.35 — prioriza Recall (FN mais caro que FP em RH)
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><hr style='border-color:#30363d;'>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center;color:#30363d!important;font-size:0.85rem;">
    Desenvolvido por
    <a href="https://github.com/jhastoledo" style="color:#58a6ff;">
        Jhonnes Toledo</a>
    &nbsp;|&nbsp;
    <a href="https://linkedin.com/in/jhonnestoledo" style="color:#58a6ff;">
        LinkedIn</a>
    &nbsp;|&nbsp; IBM HR Analytics Dataset
</p>
""", unsafe_allow_html=True)
