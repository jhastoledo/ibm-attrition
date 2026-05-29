import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import GITHUB_DARK_CSS, THRESHOLD, carregar_pipeline

st.set_page_config(page_title="Inferência — IBM Attrition",
                   page_icon="🎯", layout="wide")
st.markdown(GITHUB_DARK_CSS, unsafe_allow_html=True)

st.markdown("## 🎯 Predição de Attrition")
st.markdown("""
<p style='color:#8b949e;'>
Formulário com as <b>10 features mais importantes</b> segundo o SHAP.
As demais recebem valores neutros (mediana do conjunto de treino).
O pipeline calcula encoding e scaling automaticamente.
</p>
""", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)

pipeline_final = carregar_pipeline()

# ── Medianas do treino (features não expostas no formulário) ──
MEDIANAS = {
    'Age':                     36.0,
    'DailyRate':              802.0,
    'Education':                3.0,
    'HourlyRate':              66.0,
    'JobInvolvement':           3.0,
    'JobSatisfaction':          3.0,
    'MonthlyRate':          14235.0,
    'PercentSalaryHike':       14.0,
    'RelationshipSatisfaction': 3.0,
    'StockOptionLevel':         1.0,
    'TrainingTimesLastYear':    3.0,
    'WorkLifeBalance':          3.0,
    'YearsAtCompany':           5.0,
    'YearsSinceLastPromotion':  1.0,
    'Gender':               'Male',
    'Department':    'Research & Development',
    'EducationField':  'Life Sciences',
    'JobRole':       'Research Scientist',
    'MaritalStatus':      'Married',
}

# ── Formulário — Top 10 SHAP ──────────────────────────────────
st.markdown("### Top 10 Features (SHAP)")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔴 Fatores de Risco Principais**")
    overtime = st.selectbox(
        "1. OverTime — Faz hora extra?",
        ["No", "Yes"],
        help="SHAP=0.588 — maior preditor de attrition"
    )
    total_working = st.slider(
        "2. TotalWorkingYears — Anos de experiência total",
        0, 40, 10,
        help="Usado para calcular YearsPerCompany (SHAP=0.387)"
    )
    num_companies = st.slider(
        "   NumCompaniesWorked — Nº de empresas anteriores",
        0, 9, 2,
        help="Usado para calcular YearsPerCompany"
    )
    business_travel = st.selectbox(
        "3. BusinessTravel — Frequência de Viagens",
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        index=1,
        help="SHAP=0.486 (Frequently) — viagens aumentam risco"
    )
    env_satisfaction = st.select_slider(
        "4. EnvironmentSatisfaction — Satisfação com Ambiente",
        options=[1, 2, 3, 4], value=3,
        format_func=lambda x: {1:"Baixa",2:"Média",
                                3:"Alta",4:"Muito Alta"}[x],
        help="SHAP=0.378"
    )

with col2:
    st.markdown("**🔵 Fatores Moderadores**")
    job_satisfaction = st.select_slider(
        "5. JobSatisfaction — Satisfação no Trabalho",
        options=[1, 2, 3, 4], value=3,
        format_func=lambda x: {1:"Baixa",2:"Média",
                                3:"Alta",4:"Muito Alta"}[x],
        help="SHAP=0.380"
    )
    monthly_income = st.number_input(
        "6. MonthlyIncome — Salário Mensal",
        min_value=1009, max_value=19999, value=4919, step=500,
        help="SHAP=0.267 — salário baixo aumenta risco"
    )
    distance = st.slider(
        "7. DistanceFromHome — Distância de Casa (km)",
        1, 29, 7,
        help="SHAP=0.317"
    )
    marital_status = st.selectbox(
        "8. MaritalStatus — Estado Civil",
        ["Single", "Married", "Divorced"],
        index=1,
        help="SHAP=0.356 — solteiros têm maior risco"
    )
    age = st.slider(
        "9. Age — Idade",
        18, 60, 36,
        help="SHAP=0.248 — funcionários mais jovens = maior risco"
    )
    rel_satisfaction = st.select_slider(
        "10. RelationshipSatisfaction — Satisfação nos Relacionamentos",
        options=[1, 2, 3, 4], value=3,
        format_func=lambda x: {1:"Baixa",2:"Média",
                                3:"Alta",4:"Muito Alta"}[x],
        help="SHAP=0.248"
    )

# ── Botão ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Calcular Score de Risco",
                         use_container_width=True)

if predict_btn:
    # ── Features derivadas calculadas automaticamente ─────────
    years_per_company    = total_working / max(num_companies + 1, 1)
    income_per_year      = monthly_income / max(total_working + 1, 1)
    promocao_atrasada    = float(
        MEDIANAS['YearsSinceLastPromotion'] > 3)
    jovem_sem_experiencia= float(age < 30 and total_working < 3)

    # ── Input no formato raw — pipeline faz o resto ───────────
    dados = pd.DataFrame([{
        # Formulário
        'Age':                      float(age),
        'DistanceFromHome':         float(distance),
        'EnvironmentSatisfaction':  float(env_satisfaction),
        'JobSatisfaction':          float(job_satisfaction),
        'MonthlyIncome':            float(monthly_income),
        'NumCompaniesWorked':       float(num_companies),
        'RelationshipSatisfaction': float(rel_satisfaction),
        'TotalWorkingYears':        float(total_working),
        'OverTime':                 overtime,
        'BusinessTravel':           business_travel,
        'MaritalStatus':            marital_status,
        # Features derivadas
        'IncomePerYear':            float(income_per_year),
        'YearsPerCompany':          float(years_per_company),
        'PromocaoAtrasada':         float(promocao_atrasada),
        'JovemSemExperiencia':      float(jovem_sem_experiencia),
        # Medianas para features não expostas
        'DailyRate':                MEDIANAS['DailyRate'],
        'Education':                MEDIANAS['Education'],
        'HourlyRate':               MEDIANAS['HourlyRate'],
        'JobInvolvement':           MEDIANAS['JobInvolvement'],
        'MonthlyRate':              MEDIANAS['MonthlyRate'],
        'PercentSalaryHike':        MEDIANAS['PercentSalaryHike'],
        'StockOptionLevel':         MEDIANAS['StockOptionLevel'],
        'TrainingTimesLastYear':    MEDIANAS['TrainingTimesLastYear'],
        'WorkLifeBalance':          MEDIANAS['WorkLifeBalance'],
        'YearsAtCompany':           MEDIANAS['YearsAtCompany'],
        'YearsSinceLastPromotion':  MEDIANAS['YearsSinceLastPromotion'],
        'Gender':                   MEDIANAS['Gender'],
        'Department':               MEDIANAS['Department'],
        'EducationField':           MEDIANAS['EducationField'],
        'JobRole':                  MEDIANAS['JobRole'],
    }])

    # ── Predição — pipeline faz encoding + scaling + predict ──
    proba = float(pipeline_final.predict_proba(dados)[0, 1])
    pred  = int(proba >= THRESHOLD)

    # ── Resultado ─────────────────────────────────────────────
    st.markdown("<hr style='border-color:#30363d;'>",
                unsafe_allow_html=True)
    st.markdown("### Resultado da Predição")

    cor   = "#ff7b72" if pred == 1 else "#3fb950"
    nivel = "⚠️ ALTO RISCO DE SAÍDA" if pred == 1 else "✅ BAIXO RISCO DE SAÍDA"

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Score de Risco", f"{proba*100:.1f}%")
    col_r2.metric("Threshold",      f"{THRESHOLD*100:.0f}%")
    col_r3.metric("Predição",
                  "⚠️ RISCO" if pred == 1 else "✅ ESTÁVEL")

    st.markdown(f"""
    <div style="background:#161b22; border:2px solid {cor};
                border-radius:12px; padding:28px;
                text-align:center; margin-top:16px;">
        <div style="font-size:3.5rem; font-weight:800;
                    color:{cor};">{proba*100:.1f}%</div>
        <div style="font-size:1.2rem; color:{cor};
                    margin-top:8px; font-weight:600;">{nivel}</div>
        <div style="background:#21262d; border-radius:8px;
                    height:14px; margin:20px 0; overflow:hidden;">
            <div style="background:{cor}; width:{proba*100:.1f}%;
                        height:100%; border-radius:8px;"></div>
        </div>
        <div style="color:#8b949e; font-size:0.85rem;">
            Score {proba*100:.1f}%
            {'≥' if pred==1 else '<'}
            threshold {THRESHOLD*100:.0f}%
            →
            {'Attrition previsto' if pred==1 else
             'Sem attrition previsto'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fatores de risco detectados
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Fatores de risco detectados neste perfil:**")
    fatores = []
    if overtime == "Yes":
        fatores.append("⚠️ Faz hora extra — maior preditor (SHAP=0.588)")
    if business_travel == "Travel_Frequently":
        fatores.append("⚠️ Viagens frequentes (SHAP=0.486)")
    if years_per_company < 2:
        fatores.append("⚠️ Baixo YearsPerCompany — histórico instável (SHAP=0.387)")
    if job_satisfaction <= 2:
        fatores.append("⚠️ Baixa satisfação no trabalho (SHAP=0.380)")
    if env_satisfaction <= 2:
        fatores.append("⚠️ Baixa satisfação com ambiente (SHAP=0.378)")
    if marital_status == "Single":
        fatores.append("⚠️ Solteiro — maior mobilidade (SHAP=0.356)")
    if distance > 14:
        fatores.append("⚠️ Distância de casa alta > 14km (SHAP=0.317)")
    if age < 30:
        fatores.append("⚠️ Funcionário jovem < 30 anos")
    if rel_satisfaction <= 2:
        fatores.append("⚠️ Baixa satisfação nos relacionamentos (SHAP=0.248)")

    if fatores:
        for f in fatores:
            st.markdown(f"- {f}")
    else:
        st.success("✅ Nenhum fator de risco crítico identificado.")

    # Features derivadas calculadas
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Features derivadas calculadas automaticamente:**")
    col1, col2, col3 = st.columns(3)
    col1.metric("YearsPerCompany",
                f"{years_per_company:.2f}")
    col2.metric("IncomePerYear",
                f"{income_per_year:.1f}")
    col3.metric("JovemSemExperiencia",
                "Sim" if jovem_sem_experiencia else "Não")

    st.markdown("""
    <p style='color:#8b949e; font-size:0.8rem; margin-top:12px;'>
    ⚠️ Pipeline completo: ImbPipeline(preprocessor + RUS + LogReg).
    PR-AUC=0.532 · ROC-AUC=0.794 · Recall=79% · Threshold=0.35
    </p>
    """, unsafe_allow_html=True)
