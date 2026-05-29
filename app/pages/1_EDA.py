import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import GITHUB_DARK_CSS, FIG_DIR, carregar_dados_raw

st.set_page_config(page_title="EDA — IBM Attrition", page_icon="📊", layout="wide")
st.markdown(GITHUB_DARK_CSS, unsafe_allow_html=True)

# ── Estilo matplotlib ─────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',   'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',      'ytick.color': '#8b949e',
    'text.color': '#e6edf3',       'grid.color': '#21262d',
    'grid.linewidth': 0.6,         'font.family': 'monospace',
})
PALETTE = ['#58a6ff', '#ff7b72', '#3fb950', '#d2a8ff', '#ffa657']

# ── Header ────────────────────────────────────────────────────
st.markdown("## 📊 Análise Exploratória de Dados")
st.markdown("<hr>", unsafe_allow_html=True)

df = carregar_dados_raw()
TARGET = 'Attrition'

# ── Seção 1: Visão Geral ──────────────────────────────────────
st.markdown("### Visão Geral do Dataset")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Funcionários",  f"{df.shape[0]:,}")
col2.metric("Variáveis",     df.shape[1])
col3.metric("Saídas (Yes)",  df[TARGET].value_counts()['Yes'])
col4.metric("Taxa Attrition", f"{(df[TARGET]=='Yes').mean()*100:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📋 Ver primeiras linhas do dataset"):
    st.dataframe(df.head(10), use_container_width=True)

# ── Seção 2: Distribuição do Target ───────────────────────────
st.markdown("### Distribuição do Target")
fig_path = FIG_DIR / 'eda_01_target.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

# ── Seção 3: Distribuições Univariadas ────────────────────────
st.markdown("### Distribuições Univariadas")
fig_path = FIG_DIR / 'eda_02_distribuicoes_numericas.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

# ── Seção 4: Bivariada ────────────────────────────────────────
st.markdown("### Features Numéricas vs Attrition (Mann-Whitney U)")
fig_path = FIG_DIR / 'eda_03_bivar_numericas.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

st.markdown("### Taxa de Attrition por Variável Categórica (Qui-Quadrado)")
fig_path = FIG_DIR / 'eda_04_bivar_categoricas.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

# ── Seção 5: Correlação ───────────────────────────────────────
st.markdown("### Matriz de Correlação")
fig_path = FIG_DIR / 'eda_05_correlacao.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

# ── Seção 6: Explorador interativo ───────────────────────────
st.markdown("### 🔎 Explorador Interativo")
st.markdown("<p style='color:#8b949e'>Selecione uma variável numérica para comparar entre os grupos.</p>",
            unsafe_allow_html=True)

COLS_DROP = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
numericas = df.select_dtypes(include='number').drop(columns=COLS_DROP, errors='ignore').columns.tolist()

col_sel = st.selectbox("Variável", numericas, index=numericas.index('MonthlyIncome'))

grupo_no  = df.loc[df[TARGET] == 'No',  col_sel]
grupo_yes = df.loc[df[TARGET] == 'Yes', col_sel]
stat, p   = mannwhitneyu(grupo_no, grupo_yes, alternative='two-sided')
sig = '*** p<0.001' if p < 0.001 else ('** p<0.01' if p < 0.01 else ('* p<0.05' if p < 0.05 else 'ns'))

fig, ax = plt.subplots(figsize=(8, 4))
for j, (grupo, dados, cor) in enumerate(zip(['No','Yes'], [grupo_no, grupo_yes], [PALETTE[0], PALETTE[1]])):
    ax.boxplot(dados, positions=[j], widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=cor, alpha=0.7),
               medianprops=dict(color='#e6edf3', linewidth=2),
               whiskerprops=dict(color='#8b949e'),
               capprops=dict(color='#8b949e'),
               flierprops=dict(marker='o', color=cor, alpha=0.3, markersize=3))
ax.set_xticks([0, 1]); ax.set_xticklabels(['No', 'Yes'])
ax.set_title(f'{col_sel}  —  Mann-Whitney U: {sig}', color='#e6edf3')
st.pyplot(fig, use_container_width=True)
plt.close()

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric(f"Mediana (No)",  f"{grupo_no.median():.1f}")
col_m2.metric(f"Mediana (Yes)", f"{grupo_yes.median():.1f}")
col_m3.metric("p-value",        f"{p:.4f}  {sig}")
