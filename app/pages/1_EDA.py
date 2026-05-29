import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, mannwhitneyu, chi2_contingency
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import GITHUB_DARK_CSS, FIG_DIR, carregar_features, PALETTE

st.markdown(GITHUB_DARK_CSS, unsafe_allow_html=True)

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
    'text.color': '#e6edf3', 'grid.color': '#21262d',
    'grid.linewidth': 0.6, 'font.family': 'monospace',
})

st.markdown("## 📊 Análise Exploratória de Dados")
st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)

# ── Carregar dados processados (target já = 0/1) ──────────────
df     = carregar_features()
TARGET = 'Attrition'

# ── Métricas ──────────────────────────────────────────────────
st.markdown("### Visão Geral do Dataset")
n_at  = int(df[TARGET].sum())
n_ok  = int((df[TARGET] == 0).sum())
taxa  = df[TARGET].mean() * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Funcionários",     f"{len(df):,}")
col2.metric("Attrition (1)",    f"{n_at}")
col3.metric("Sem Attrition (0)",f"{n_ok}")
col4.metric("Taxa Attrition",   f"{taxa:.1f}%")
col5.metric("Desbalanceamento", f"{n_ok/n_at:.1f}:1")

# ── Figuras EDA ───────────────────────────────────────────────
st.markdown("### Distribuição do Target")
fig_path = FIG_DIR / 'eda_01_target.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

st.markdown("### Distribuições Numéricas por Attrition")
fig_path = FIG_DIR / 'eda_02_distribuicoes_numericas.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

st.markdown("### Análise Bivariada — Numéricas")
fig_path = FIG_DIR / 'eda_03_bivar_numericas.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

st.markdown("### Análise Bivariada — Categóricas")
fig_path = FIG_DIR / 'eda_04_bivar_categoricas.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

st.markdown("### Correlação")
fig_path = FIG_DIR / 'eda_05_correlacao.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

# ── Explorador Interativo ─────────────────────────────────────
st.markdown("### 🔎 Explorador Interativo")

# Features numéricas disponíveis no parquet
COLS_NUM = [c for c in [
    'Age', 'MonthlyIncome', 'TotalWorkingYears',
    'YearsAtCompany', 'DistanceFromHome',
    'YearsPerCompany', 'IncomePerYear',
    'NumCompaniesWorked', 'YearsSinceLastPromotion',
] if c in df.columns]

col_x = st.selectbox("Feature X", COLS_NUM, index=0)

if col_x in df.columns:
    g0 = df.loc[df[TARGET] == 0, col_x].dropna()
    g1 = df.loc[df[TARGET] == 1, col_x].dropna()
    _, p = mannwhitneyu(g0, g1, alternative='two-sided')
    sig  = '***' if p < 0.001 else ('**' if p < 0.01
           else ('*' if p < 0.05 else 'ns'))
    r, _ = pointbiserialr(df[TARGET], df[col_x].fillna(df[col_x].median()))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(g0, bins=35, alpha=0.6, color=PALETTE[1],
            label=f'Sem Attrition (n={len(g0)})', density=True)
    ax.hist(g1, bins=35, alpha=0.6, color=PALETTE[2],
            label=f'Attrition (n={len(g1)})', density=True)
    ax.axvline(g0.median(), color=PALETTE[1],
               linestyle='--', linewidth=1.5,
               label=f'Med: {g0.median():.1f}')
    ax.axvline(g1.median(), color=PALETTE[2],
               linestyle='--', linewidth=1.5,
               label=f'Med: {g1.median():.1f}')
    ax.set_title(f'{col_x} vs Attrition — '
                 f'MW {sig} (p={p:.2e}) | r={r:.3f}',
                 color='#e6edf3')
    ax.set_xlabel(col_x)
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Mediana — Sem Attrition", f"{g0.median():.2f}")
    c2.metric(f"Mediana — Attrition",     f"{g1.median():.2f}")
    c3.metric(f"Diferença",
              f"{((g1.median()-g0.median())/g0.median()*100):+.1f}%")
