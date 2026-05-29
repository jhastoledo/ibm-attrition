import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import GITHUB_DARK_CSS, FIG_DIR, carregar_comparativo

st.markdown(GITHUB_DARK_CSS, unsafe_allow_html=True)

st.markdown("## 🤖 Modelagem e Otimização")
st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)

df_comp = carregar_comparativo()

def highlight(s):
    if s.name in ['PR-AUC', 'ROC-AUC', 'Recall', 'F1']:
        best = s == s.max()
    else:
        return ['' for _ in s]
    return ['background-color:#1f4d25;color:#3fb950'
            if v else '' for v in best]

# ── Comparativo ───────────────────────────────────────────────
st.markdown("### Comparativo — 7 Modelos com ImbPipeline + RUS")
st.markdown("""
<p style='color:#8b949e;'>
Todos os modelos usam ImbPipeline (preprocessor + RandomUnderSampler + classifier).
PR-AUC é a métrica principal — desbalanceamento 5.2:1.
Dois preprocessadores: com StandardScaler (LR, SVC, KNN) e sem (DT, LGBM, XGB).
</p>
""", unsafe_allow_html=True)

st.dataframe(
    df_comp.style.apply(highlight,
                        subset=['PR-AUC', 'ROC-AUC', 'Recall', 'F1'])
               .format({'PR-AUC':'{:.4f}', 'PR-AUC±':'{:.4f}',
                        'ROC-AUC':'{:.4f}', 'Recall':'{:.4f}',
                        'F1':'{:.4f}'}),
    use_container_width=True
)

fig_path = FIG_DIR / 'nb03_comparativo.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

# ── Optuna ────────────────────────────────────────────────────
st.markdown("### Tuning — Optuna 80 Trials")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Trials",       "80")
col2.metric("Convergiu em", "Trial 38")
col3.metric("Penalty",      "ElasticNet")
col4.metric("l1_ratio",     "0.947 (≈L1)")

fig_path = FIG_DIR / 'nb04_optuna.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

# ── Avaliação final ───────────────────────────────────────────
st.markdown("### Avaliação Final — LogReg Tuned (threshold=0.35)")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("PR-AUC",    "0.532")
col2.metric("ROC-AUC",   "0.794")
col3.metric("Recall",    "78.7%")
col4.metric("Precision", "28.5%")
col5.metric("Threshold", "0.35")

fig_path = FIG_DIR / 'nb04_avaliacao.png'
if fig_path.exists():
    st.image(str(fig_path), use_container_width=True)

# ── Explicação threshold ──────────────────────────────────────
st.markdown("""
<div style="background:#161b22;border-left:4px solid #58a6ff;
            padding:14px 18px;border-radius:0 8px 8px 0;margin-top:16px;">
    <b style="color:#58a6ff;">Por que Threshold=0.35?</b>
    <p style="color:#8b949e;margin:6px 0 0;">
    Em RH, o custo de <strong>não identificar</strong> um funcionário que vai sair (FN)
    é muito maior que o custo de uma conversa de retenção desnecessária (FP).
    Threshold baixo prioriza Recall — capturar o máximo de casos reais.
    TP=37 × funcionários retidos &gt;&gt; FP=93 × conversas desnecessárias.
    </p>
</div>
""", unsafe_allow_html=True)
