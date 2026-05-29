import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import GITHUB_DARK_CSS, FIG_DIR

st.set_page_config(page_title="SHAP — IBM Attrition", page_icon="🔍", layout="wide")
st.markdown(GITHUB_DARK_CSS, unsafe_allow_html=True)

st.markdown("## 🔍 Interpretabilidade — SHAP")
st.markdown("<hr>", unsafe_allow_html=True)

# ── Seção 1: O que é SHAP ─────────────────────────────────────
st.markdown("### O que é SHAP?")
st.markdown("""
<p style='color:#8b949e'>
SHAP (SHapley Additive exPlanations) quantifica a contribuição de cada feature
para a predição de cada instância. Baseado na teoria dos jogos cooperativos,
garante propriedades matemáticas de <b>consistência</b> e <b>completude</b>.
</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
col1.markdown("""
**Interpretação dos valores:**
- **Valor positivo** → feature aumenta a probabilidade de attrition
- **Valor negativo** → feature reduz a probabilidade de attrition
- **|valor| maior** → maior impacto na predição
""")
col2.markdown("""
**Cores no Summary Plot:**
- 🔴 **Rosa/Vermelho** → valor alto da feature
- 🔵 **Azul** → valor baixo da feature
""")

# ── Seção 2: Summary Plot Global ─────────────────────────────
st.markdown("### Importância Global das Features")
st.markdown("<p style='color:#8b949e'>Top features ordenadas por impacto médio absoluto no modelo.</p>",
            unsafe_allow_html=True)
fig_path = FIG_DIR / 'nb05_shap_summary.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

# ── Seção 3: Top insights ─────────────────────────────────────
st.markdown("### Top 5 Insights do SHAP")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**1. OverTime (SHAP=0.681)**
Hora extra é o maior fator de risco.
Funcionários com OT têm ~3x mais chance de sair.

**2. YearsPerCompany (SHAP=0.454)**
Feature derivada — quem muda muito de empresa
tende a sair novamente. Instabilidade histórica.

**3. StockOptionLevel (SHAP=0.414)**
Sem stock options → menos vínculo financeiro.
Nível 0 é forte preditor de saída.
""")
with col2:
    st.markdown("""
**4. EnvironmentSatisfaction (SHAP=0.310)**
Insatisfação com o ambiente é sinal de alerta precoce
e acionável pelo RH antes da saída.

**5. MonthlyIncome (SHAP=0.267)**
Salário baixo aumenta risco — especialmente
em Sales e Human Resources.
""")

# ── Seção 4: Waterfall ────────────────────────────────────────
st.markdown("### Explicação Individual — Waterfall Plot")
st.markdown("<p style='color:#8b949e'>Comparação entre um caso detectado (True Positive) e um não detectado (False Negative).</p>",
            unsafe_allow_html=True)
fig_path = FIG_DIR / 'nb05_shap_waterfall.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

# ── Seção 5: Avaliação ────────────────────────────────────────
st.markdown("### Avaliação Final no Conjunto de Teste")
col1, col2, col3, col4 = st.columns(4)
col1.metric("ROC-AUC",  "0.784")
col2.metric("PR-AUC",   "0.534")
col3.metric("Recall",   "0.510",  delta="+0.21 vs baseline")
col4.metric("F1",       "0.530",  delta="+0.09 vs baseline")

fig_path = FIG_DIR / 'nb05_confusion_matrix.png'
if fig_path.exists():
    col_img, col_text = st.columns([1, 1])
    col_img.image(str(fig_path), width=400)
    with col_text:
        st.markdown("""
**Leitura da matriz (threshold=0.35):**

- **0.92** dos funcionários que ficaram → corretamente classificados
- **0.51** dos funcionários que saíram → detectados pelo modelo
- **0.49** dos casos reais → ainda não detectados (False Negatives)

O threshold 0.35 foi escolhido para maximizar o F1,
priorizando o Recall em detrimento da Precision —
correto para o contexto de RH onde perder um caso
real é mais custoso que um falso alarme.
""")

fig_path = FIG_DIR / 'nb05_threshold.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)
