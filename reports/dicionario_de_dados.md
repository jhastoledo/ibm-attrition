# 📖 Dicionário de Dados — IBM HR Attrition

Dataset sintético criado pela IBM para análise de turnover voluntário
de funcionários — amplamente usado como benchmark em projetos de RH Analytics.

**Fonte:** [IBM HR Analytics — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
**Arquivo:** `employee_attrition.csv`
**Registros:** 1.470 funcionários
**Colunas:** 35
**Nota:** Dataset sintético — criado pela IBM para fins educacionais.

---

## Features Originais

### Identificação e Target

| Coluna | Tipo | Descrição | Valores |
|---|---|---|---|
| `EmployeeNumber` | int | Identificador único — **descartado** | 1–2068 |
| `Attrition` | str | **Target** — saída voluntária | Yes=1, No=0 |

> Taxa de Attrition: 16.1% (237 de 1.470). Desbalanceamento 5.2:1.

### Perfil Demográfico

| Coluna | Tipo | Descrição | Valores |
|---|---|---|---|
| `Age` | int | Idade do funcionário | 18–60 |
| `Gender` | str | Gênero | Male, Female |
| `MaritalStatus` | str | Estado civil | Single, Married, Divorced |
| `Education` | int | Nível de escolaridade (ordinal) | 1=Abaixo Faculdade, 2=Faculdade, 3=Bacharel, 4=Mestre, 5=Doutor |
| `EducationField` | str | Área de formação | Life Sciences, Medical, Marketing, Technical Degree, Human Resources, Other |
| `DistanceFromHome` | int | Distância de casa para o trabalho (km) | 1–29 |

### Informações Profissionais

| Coluna | Tipo | Descrição | Valores |
|---|---|---|---|
| `Department` | str | Departamento | Sales, Research & Development, Human Resources |
| `JobRole` | str | Cargo | 9 cargos (ver abaixo) |
| `JobLevel` | int | Nível hierárquico | 1–5 |
| `BusinessTravel` | str | Frequência de viagens | Non-Travel, Travel_Rarely, Travel_Frequently |
| `OverTime` | str | Faz hora extra? | Yes, No |
| `YearsAtCompany` | int | Anos na empresa atual | 0–40 |
| `YearsInCurrentRole` | int | Anos no cargo atual | 0–18 |
| `YearsSinceLastPromotion` | int | Anos desde última promoção | 0–15 |
| `YearsWithCurrManager` | int | Anos com gestor atual | 0–17 |
| `TotalWorkingYears` | int | Total de anos de experiência | 0–40 |
| `NumCompaniesWorked` | int | Número de empresas anteriores | 0–9 |
| `TrainingTimesLastYear` | int | Treinamentos no último ano | 0–6 |

> **Cargos disponíveis:** Sales Executive, Research Scientist, Laboratory Technician,
> Manufacturing Director, Healthcare Representative, Manager,
> Sales Representative, Research Director, Human Resources.

### Satisfação e Engajamento

| Coluna | Tipo | Descrição | Escala |
|---|---|---|---|
| `JobSatisfaction` | int | Satisfação com o trabalho | 1=Baixa, 2=Média, 3=Alta, 4=Muito Alta |
| `EnvironmentSatisfaction` | int | Satisfação com o ambiente físico | 1=Baixa, 2=Média, 3=Alta, 4=Muito Alta |
| `RelationshipSatisfaction` | int | Satisfação nos relacionamentos no trabalho | 1=Baixa, 2=Média, 3=Alta, 4=Muito Alta |
| `JobInvolvement` | int | Envolvimento com o trabalho | 1=Baixo, 2=Médio, 3=Alto, 4=Muito Alto |
| `WorkLifeBalance` | int | Equilíbrio trabalho-vida | 1=Ruim, 2=Bom, 3=Melhor, 4=Ótimo |
| `PerformanceRating` | int | Avaliação de desempenho | 1=Baixo, 2=Bom, 3=Excelente, 4=Excepcional |

> `PerformanceRating` foi descartada — apenas valores 3 e 4 no dataset
> (variância quase zero, sem poder preditivo).

### Remuneração

| Coluna | Tipo | Descrição | Range |
|---|---|---|---|
| `MonthlyIncome` | int | Salário mensal | $1.009–$19.999 |
| `MonthlyRate` | int | Taxa mensal (componente variável) | $2.094–$26.999 |
| `DailyRate` | int | Taxa diária | $102–$1.499 |
| `HourlyRate` | int | Taxa horária | $30–$100 |
| `PercentSalaryHike` | int | Percentual de aumento no último ano | 11–25% |
| `StockOptionLevel` | int | Nível de stock options | 0–3 |

### Colunas Constantes (descartadas)

| Coluna | Valor | Motivo |
|---|---|---|
| `EmployeeCount` | 1 (sempre) | Constante — sem valor preditivo |
| `Over18` | Y (sempre) | Constante — todos maiores de 18 |
| `StandardHours` | 80 (sempre) | Constante — sem variação |

---

## Features Derivadas (criadas no NB02)

| Feature | Fórmula | Motivação | SHAP |
|---|---|---|---|
| `YearsPerCompany` | `TotalWorkingYears / (NumCompaniesWorked + 1)` | Estabilidade histórica — quanto mais baixo, mais instável | 0.387 (#3) |
| `IncomePerYear` | `MonthlyIncome / (TotalWorkingYears + 1)` | Remuneração relativa à experiência | — |
| `SatisfacaoMedia` | `mean(JobSat, EnvSat, RelSat)` | Índice geral de satisfação — **removida por alta correlação** | — |
| `PromocaoAtrasada` | `YearsSinceLastPromotion > 3` | Flag de promoção atrasada | — |
| `JovemSemExperiencia` | `Age < 30 AND TotalWorkingYears < 3` | Perfil de alto risco por instabilidade | — |

---

## Encoding aplicado no Pipeline (NB02)

### Preprocessador Linear (pipeline_linear.pkl)
*Para: LogisticRegression, SVC, KNeighborsClassifier*

| Feature | Transformação | Detalhes |
|---|---|---|
| `Gender` | OrdinalEncoder | Female=0, Male=1 |
| `OverTime` | OrdinalEncoder | No=0, Yes=1 |
| `BusinessTravel` | OneHotEncoder (drop='first') | Referência: Non-Travel |
| `Department` | OneHotEncoder (drop='first') | Referência: Human Resources |
| `EducationField` | OneHotEncoder (drop='first') | Referência: Human Resources |
| `JobRole` | OneHotEncoder (drop='first') | Referência: Healthcare Representative |
| `MaritalStatus` | OneHotEncoder (drop='first') | Referência: Divorced |
| Numéricas contínuas | StandardScaler | Age, MonthlyIncome, TotalWorkingYears... |
| Ordinais numéricas | Passthrough | Education, JobSatisfaction, StockOptionLevel... |
| Features derivadas binárias | Passthrough | PromocaoAtrasada, JovemSemExperiencia |

### Preprocessador Árvore (pipeline_arvore.pkl)
*Para: DecisionTree, LGBMClassifier, XGBClassifier*

| Feature | Transformação | Detalhes |
|---|---|---|
| `Gender`, `OverTime` | OrdinalEncoder | Igual ao linear |
| Nominais | OneHotEncoder (drop='first') | Igual ao linear |
| Numéricas contínuas | **Passthrough** | Tree-based não precisa de scaling |
| Ordinais numéricas | Passthrough | Igual ao linear |

---

## Colunas Removidas por Alta Correlação (NB01)

| Coluna | Correlação com | r | Ação |
|---|---|---|---|
| `SatisfacaoMedia` | `JobSatisfaction`, `EnvironmentSatisfaction` | >0.8 | Removida — derivada das originais |
| `YearsInCurrentRole` | `YearsAtCompany` | 0.76 | Removida — multicolinearidade |
| `YearsWithCurrManager` | `YearsAtCompany` | 0.77 | Removida — multicolinearidade |

---

## Limpeza Realizada (NB02)

| Ação | Critério | Registros afetados |
|---|---|---|
| Remoção colunas constantes | `EmployeeCount`, `Over18`, `StandardHours` | 3 colunas |
| Remoção identificador | `EmployeeNumber` | 1 coluna |
| Remoção por correlação | `SatisfacaoMedia`, `YearsInCurrentRole`, `YearsWithCurrManager` | 3 colunas |
| Remoção `PerformanceRating` | Variância quase zero (só valores 3 e 4) | 1 coluna |
| **Nulos** | Nenhum | 0 registros |
| **Duplicatas** | Nenhuma | 0 registros |
| **Total após limpeza** | | **1.470 registros × 30 features** |

---

## Métricas do Modelo Final

| Parâmetro | Valor | Descrição |
|---|---|---|
| Algoritmo | LogisticRegression | ElasticNet · saga solver |
| C | 0.957 | Regularização moderada |
| penalty | elasticnet | l1_ratio=0.947 → quase L1 puro |
| Balanceamento | RandomUnderSampler | Dentro do ImbPipeline — sem data leakage |
| Threshold | 0.35 | Otimizado para maximizar Recall |
| PR-AUC | 0.532 | Area Under Precision-Recall Curve |
| ROC-AUC | 0.794 | Area Under ROC Curve |
| Recall | 0.787 | 79% dos casos reais identificados |
| Precision | 0.285 | 28% dos alertas são casos reais |

> **Interpretação do threshold=0.35:** em RH, o custo de não
> identificar um funcionário que vai sair (FN) é muito maior que
> o custo de uma conversa de retenção desnecessária (FP).
> Threshold baixo prioriza Recall em detrimento de Precision.
