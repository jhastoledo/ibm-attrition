import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import GITHUB_DARK_CSS, FIG_DIR, REPORTS, carregar_comparativo

st.set_page_config(page_title="Modelagem — IBM Attrition", page_icon="🤖", layout="wide")
st.markdown(GITHUB_DARK_CSS, unsafe_allow_html=True)

st.markdown("## 🤖 Modelagem e Tuning")
st.markdown("<hr>", unsafe_allow_html=True)

# ── Seção 1: Comparativo Baseline ────────────────────────────
st.markdown("### Comparativo de Modelos — Baseline (5-fold CV)")
st.markdown("<p style='color:#8b949e'>5 modelos avaliados com validação cruzada estratificada.</p>",
            unsafe_allow_html=True)

df_comp = carregar_comparativo()

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: #1f4d25; color: #3fb950' if v else '' for v in is_max]

st.dataframe(
    df_comp.style
        .apply(highlight_max, subset=['ROC-AUC', 'PR-AUC', 'Recall', 'F1'])
        .format({'ROC-AUC': '{:.3f}', 'ROC-AUC±': '{:.3f}',
                 'PR-AUC': '{:.3f}', 'Recall': '{:.3f}', 'F1': '{:.3f}'}),
    use_container_width=True
)

fig_path = FIG_DIR / 'nb03_comparativo_modelos.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

# ── Seção 2: Decisão ─────────────────────────────────────────
st.markdown("### Decisão — Modelo Selecionado")
col1, col2 = st.columns(2)
with col1:
    st.info("""
**XGBoost** foi selecionado para tuning pelos seguintes motivos:

- Melhor potencial de melhoria com otimização de hiperparâmetros
- `scale_pos_weight` nativo para desbalanceamento
- Suporte a SHAP via `TreeExplainer` (interpretabilidade)
- Recall de 0.411 no baseline → esperado > 0.50 pós-tuning
""")
with col2:
    st.warning("""
**Logistic Regression** liderou o baseline (ROC-AUC 0.832), mas:

- Modelo mais simples, menor capacidade de capturar interações
- Menos flexível para otimização de threshold
- Não gera SHAP values nativamente
""")

# ── Seção 3: Tuning Optuna ────────────────────────────────────
st.markdown("### Tuning com Optuna — 80 Trials")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Trials",          "80")
col2.metric("Melhor Trial",    "61")
col3.metric("ROC-AUC (CV)",    "0.8368")
col4.metric("Tempo",           "~22s")

fig_path = FIG_DIR / 'nb04_optuna_analise.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

st.markdown("### Melhores Hiperparâmetros")
params = {
    'n_estimators': 479, 'max_depth': 4,
    'learning_rate': 0.216, 'subsample': 0.627,
    'colsample_bytree': 0.761, 'min_child_weight': 9,
    'gamma': 4.358, 'reg_alpha': 0.719,
    'reg_lambda': 2.531, 'scale_pos_weight': 1.054,
}
df_params = pd.DataFrame(params.items(), columns=['Hiperparâmetro', 'Valor'])
st.dataframe(df_params, use_container_width=True, hide_index=True)
