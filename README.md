# 👥 IBM HR Attrition — Employee Turnover Predictor

[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-RUS-6c5ce7?style=for-the-badge)](https://imbalanced-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Interpretabilidade-00b894?style=for-the-badge)](https://shap.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-80%20trials-189fdd?style=for-the-badge)](https://optuna.org)
[![License](https://img.shields.io/badge/License-MIT-8b949e?style=for-the-badge)](LICENSE)

> Pipeline completa de Machine Learning para previsão de turnover voluntário
> em dados de RH da IBM — com **ImbPipeline + RandomUnderSampler**,
> dois ColumnTransformers especializados e interpretabilidade via SHAP.

🔗 **[Acessar App no Streamlit Cloud](https://ibm-attrition-mzpihkegk29fxsnoa7w6hn.streamlit.app/)**

---

## 📋 Sumário

- [Contexto](#contexto)
- [Resultados](#resultados)
- [Visualizações](#visualizações)
- [Pipeline](#pipeline)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Como Executar](#como-executar)
- [Principais Insights](#principais-insights)
- [Tecnologias](#tecnologias)
- [Autor](#autor)

---

## Contexto

O dataset IBM HR Analytics contém informações de **1.470 funcionários**
com 35 features de RH. O objetivo é prever quais funcionários têm maior
risco de saída voluntária — permitindo ações preventivas de retenção.

| Dado | Valor |
|---|---|
| Fonte | IBM HR Analytics (Kaggle) |
| Registros | 1.470 funcionários |
| Taxa de Attrition | 16.1% (237 saídas) |
| Desbalanceamento | 5.2:1 |
| Features originais | 35 |
| Features após eng. | 30 input → 44 após encoding |

---

## Resultados

### Comparativo de Modelos — 5-fold StratifiedKFold + RUS

| Modelo | PR-AUC | ROC-AUC | Recall |
|---|---|---|---|
| **LogisticRegression** | **0.609** | **0.830** | **0.768** |
| SVC | 0.569 | 0.812 | 0.684 |
| LGBMClassifier | 0.516 | 0.776 | 0.647 |
| XGBClassifier | 0.482 | 0.767 | 0.668 |
| KNN | 0.317 | 0.697 | 0.574 |
| DecisionTree | 0.212 | 0.617 | 0.632 |
| Dummy | 0.168 | 0.518 | 0.532 |

### Modelo Final — LogisticRegression Tuned (ElasticNet)

| Métrica | Valor |
|---|---|
| PR-AUC (teste) | 0.532 |
| ROC-AUC (teste) | 0.794 |
| Recall | 0.787 |
| Threshold | 0.35 |
| TP | 37 |
| FP | 93 |
| FN | 10 |
| TN | 154 |

> **Por que Recall > Precision?** Em RH, o custo de perder um
> funcionário qualificado (FN) é muito maior que o custo de uma
> conversa de retenção desnecessária (FP). Threshold=0.35 prioriza
> não deixar passar casos reais.

### Diferenciais da Refatoração

| Componente | Antes | Depois |
|---|---|---|
| Encoding | `LabelEncoder` + `pd.get_dummies` | `OrdinalEncoder` + `OHE` no `ColumnTransformer` |
| Balanceamento | Externo ao CV | `RUS` dentro do `ImbPipeline` |
| Preprocessadores | 1 genérico | **2 especializados** (linear vs tree-based) |
| Artefatos | 3 pkl separados | **1 `pipeline_final.joblib`** |
| Inferência | encoder → scaler → modelo | `pipeline.predict_proba(X_raw)` |

---

## Visualizações

### Comparativo de Modelos Baseline
> LogisticRegression lidera em PR-AUC com ImbPipeline + RUS

![Comparativo Modelos](reports/figures/nb03_comparativo.png)

---

### Análise Optuna — Convergência + Importância dos HPs
> 80 trials · ElasticNet com l1_ratio≈0.95 · convergiu no trial 38

![Optuna](reports/figures/nb04_optuna.png)

---

### Avaliação Final — Curvas PR + ROC + Matriz de Confusão
> PR-AUC=0.532 · ROC-AUC=0.794 · Recall=0.787 · Threshold=0.35

![Avaliação](reports/figures/nb04_avaliacao.png)

---

### SHAP Summary — Top Preditores
> OverTime lidera (0.588) — consistente com análise original

![SHAP Summary](reports/figures/nb05_shap_summary.png)

---

### SHAP Waterfall — Verdadeiro Positivo vs Falso Positivo
> P=0.984 (TP) vs P=0.954 (FP) — fatores que influenciam cada predição

![SHAP Waterfall](reports/figures/nb05_shap_waterfall.png)

---

### Distribuição do Target + EDA
> Análise exploratória — OverTime e BusinessTravel como principais preditores

![EDA Target](reports/figures/eda_01_target.png)

---

## Pipeline

```
NB01 → NB02 → NB03 → NB04 → NB05 → NB06
 EDA   Feat.  Base   Tuning  SHAP   Rel.
       Eng.   line
```

| Notebook | Descrição | Entregável |
|---|---|---|
| `01_eda.ipynb` | EDA com testes estatísticos + análise de OverTime | 5 figuras |
| `02_feature_engineering.ipynb` | **2 ColumnTransformers** (linear + tree) + 5 features derivadas | Splits + artefatos |
| `03_modelagem_baseline.ipynb` | **ImbPipeline + RUS** · 7 modelos · StratifiedKFold | Comparativo |
| `04_tuning.ipynb` | Optuna 80 trials · ElasticNet · saga solver | `pipeline_final.joblib` |
| `05_interpretabilidade.ipynb` | SHAP LinearExplainer · Waterfall · Dependence plots | 4 figuras |
| `06_relatorio_final.ipynb` | Consolidação + recomendações de RH | HTML GitHub Dark |

---

## Estrutura do Repositório

```
ibm_attrition/
├── app/
│   ├── main.py
│   ├── utils.py                    # carregar_pipeline() — objeto único
│   └── pages/
│       ├── 1_EDA.py
│       ├── 2_Modelagem.py
│       ├── 3_SHAP.py
│       └── 4_Inferencia.py         # pipeline.predict_proba(X_raw)
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/                        # employee_attrition.csv
│   └── processed/                  # parquets
├── models/
│   ├── pipeline_final.joblib       # ImbPipeline completo
│   ├── pipeline_linear.pkl         # ColumnTransformer (com scaler)
│   └── pipeline_arvore.pkl         # ColumnTransformer (sem scaler)
├── notebooks/                      # 6 notebooks
├── reports/
│   ├── figures/                    # 15 figuras
│   ├── comparativo_modelos.csv
│   └── relatorio_final.html
├── src/
│   ├── config.py
│   ├── viz_config.py               # GitHub Dark Theme
│   └── __init__.py
├── requirements.txt
└── environment.yml
```

---

## Como Executar

### 1. Clonar o repositório

```bash
git clone https://github.com/jhastoledo/ibm-attrition.git
cd ibm-attrition
```

### 2. Criar o ambiente

```bash
conda env create -f environment.yml
conda activate singularity
pip install -e .
```

### 3. Obter o dataset

Baixar `employee_attrition.csv` do
[Kaggle — IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
e colocar em `data/raw/`.

### 4. Rodar os notebooks

```bash
jupyter lab
```

### 5. Rodar o app

```bash
streamlit run app/main.py
```

---

## Principais Insights

### OverTime é o preditor #1 (SHAP=0.588)
Funcionários que fazem hora extra têm risco de attrition
dramaticamente maior. Política de horas extras é a alavanca
de retenção mais impactante segundo o modelo.

### ImbPipeline garante CV sem data leakage
O `RandomUnderSampler` aplicado **dentro** de cada fold do
cross-validation evita que a distribuição manipulada vaze
para os dados de validação — erro comum em projetos anteriores.

### Dois ColumnTransformers especializados
Modelos lineares precisam de `StandardScaler` para convergir.
Modelos tree-based não precisam de scaling — e podem performar
melhor sem ele. A separação dos preprocessadores reflete
essa diferença arquitetural.

### LogisticRegression superou LGBM e XGBoost
Com o preprocessamento correto (StandardScaler + OHE + OrdinalEncoder)
e balanceamento via RUS, a Regressão Logística obteve PR-AUC=0.609
vs LGBM=0.516 e XGB=0.482 no cross-validation.

### Top 7 Preditores SHAP

| # | Feature | SHAP | Ação |
|---|---|---|---|
| 1 | OverTime | 0.588 | Limitar horas extras |
| 2 | BusinessTravel_Frequently | 0.486 | Reduzir viagens |
| 3 | YearsPerCompany | 0.387 | Monitorar histórico instável |
| 4 | JobSatisfaction | 0.380 | Pesquisas de clima |
| 5 | EnvironmentSatisfaction | 0.378 | Melhorar ambiente físico |
| 6 | MaritalStatus_Single | 0.356 | Programas de integração |
| 7 | DistanceFromHome | 0.317 | Home office / transporte |

---

## Tecnologias

**Linguagem**

![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=for-the-badge&logo=python&logoColor=white)

**Dados e Análise**

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

**Machine Learning**

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-6c5ce7?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-00b894?style=for-the-badge)
![Optuna](https://img.shields.io/badge/Optuna-189fdd?style=for-the-badge)

**Visualização e App**

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**Ambiente**

![Conda](https://img.shields.io/badge/Conda-44A833?style=for-the-badge&logo=anaconda&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

---

## Autor

<div align="center">

### Jhonnes Toledo

**Físico (BSc & MSc — UFJF) | Pós-graduando em Data Science (UNINASSAU)**

Data Science practitioner com background em física, estatística e
machine learning. Experiência em Python, análise exploratória,
modelagem preditiva e deploy de aplicações.

<br>

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jhonnestoledo/)
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jhastoledo)
[![Outlook](https://img.shields.io/badge/-Outlook-0078D4?style=for-the-badge&logo=microsoft-outlook&logoColor=white)](mailto:jas_toledo@hotmail.com)

</div>

---

<div align="center">
  <i>1.470 funcionários · Recall=79% · ImbPipeline + RUS ·
  OverTime preditor #1 · Logistic Reg. > LGBM + XGBoost</i>
</div>
