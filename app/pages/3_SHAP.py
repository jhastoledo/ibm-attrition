import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import GITHUB_DARK_CSS, FIG_DIR

st.markdown(GITHUB_DARK_CSS, unsafe_allow_html=True)

st.markdown("## 🔍 Interpretabilidade — SHAP")
st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)

st.markdown("""
<p style='color:#8b949e;'>
SHAP (SHapley Additive exPlanations) quantifica a contribuição de cada feature
para cada predição individual. Modelo: <b style='color:#e6edf3;'>
LinearExplainer</b> — adequado para Logistic Regression.
</p>
""", unsafe_allow_html=True)

# ── Summary ───────────────────────────────────────────────────
st.markdown("### SHAP Summary Plot")
st.markdown("""
<p style='color:#8b949e;'>
OverTime (0.588) é o preditor #1 — consistente com a análise descritiva do NB01.
BusinessTravel_Frequently (0.486) em segundo — viagens frequentes aumentam risco.
YearsPerCompany (0.387) é uma <b style='color:#e6edf3;'>feature derivada criada no NB02</b>.
</p>
""", unsafe_allow_html=True)
fig_path = FIG_DIR / 'nb05_shap_summary.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

# ── Ranking ───────────────────────────────────────────────────
st.markdown("### Top 10 Features — |SHAP| Médio")
shap_data = pd.DataFrame({
    'Feature': [
        'OverTime', 'BusinessTravel_Frequently',
        'YearsPerCompany', 'JobSatisfaction',
        'EnvironmentSatisfaction', 'Dept_R&D',
        'MaritalStatus_Single', 'DistanceFromHome',
        'BusinessTravel_Rarely', 'TotalWorkingYears',
    ],
    'SHAP_mean': [
        0.588, 0.486, 0.387, 0.380,
        0.378, 0.377, 0.356, 0.317,
        0.308, 0.288,
    ],
    'Tipo': [
        'Ordinal', 'OHE', 'Derivada', 'Passthrough',
        'Passthrough', 'OHE', 'OHE', 'Standard',
        'OHE', 'Standard',
    ],
    'Ação de RH': [
        'Limitar horas extras',
        'Reduzir viagens frequentes',
        'Monitorar histórico de instabilidade',
        'Pesquisas de clima individuais',
        'Melhorar condições do ambiente físico',
        'Atenção ao departamento R&D',
        'Programas de integração social',
        'Home office ou vale-transporte',
        'Política de viagens moderadas',
        'Mentoria para veteranos',
    ],
})

for _, row in shap_data.iterrows():
    cor = {'Ordinal': '#58a6ff', 'OHE': '#3fb950',
           'Derivada': '#d2a8ff', 'Passthrough': '#ffa657',
           'Standard': '#79c0ff'}.get(row['Tipo'], '#8b949e')
    pct = row['SHAP_mean'] / 0.588 * 100
    with st.expander(
        f"**{row['Feature']}** — SHAP: {row['SHAP_mean']:.3f} | {row['Tipo']}"
    ):
        st.markdown(f"<p style='color:#8b949e;'>💡 {row['Ação de RH']}</p>",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0d1117;border-radius:6px;
                    height:8px;overflow:hidden;">
            <div style="background:{cor};width:{pct:.0f}%;
                        height:100%;border-radius:6px;"></div>
        </div>
        """, unsafe_allow_html=True)

# ── Waterfall ─────────────────────────────────────────────────
st.markdown("### SHAP Waterfall — Verdadeiro Positivo vs Falso Positivo")
st.markdown("""
<p style='color:#8b949e;'>
<b style='color:#3fb950;'>TP:</b> P=0.984 — funcionário que realmente saiu, corretamente identificado.<br>
<b style='color:#ff7b72;'>FP:</b> P=0.954 — alarme falso — conversa de retenção desnecessária.
</p>
""", unsafe_allow_html=True)
fig_path = FIG_DIR / 'nb05_shap_waterfall.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

# ── Dependence ────────────────────────────────────────────────
st.markdown("### SHAP Dependence Plots")
fig_path = FIG_DIR / 'nb05_shap_dependence.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)
