# Previsão de incidência de diabetes com Kedro

Este documento embasa a nossa tomada de decisão e descreve como contruímos os pipelines usando o notebook `diabetes-prediction.ipynb` como ponto de partida.

## Como rodar

```bash
uv sync
uv run kedro run
uv run kedro viz run
```

Cada pipeline também pode ser executado separadamente, nesta ordem:

```bash
uv run kedro run --pipeline data_engineering
uv run kedro run --pipeline modelling
uv run kedro run --pipeline inference
```

No macOS, o XGBoost e o LightGBM precisam do OpenMP (`brew install libomp`).

## Estrutura

| Pipeline | Entrada | Saída |
|---|---|---|
| `data_engineering` | `data/01_raw/diabetes-dataset-modelling.csv` | `master_table`, encoders e scalers |
| `modelling` | `master_table` | 9 modelos treinados, métricas e `best_model` |
| `inference` | `data/01_raw/diabetes-dataset-inference.csv` | `data/07_model_output/inferences.json` |

## 1. Engenharia de dados

Nós, em ordem:

1. `clean_data`: seleciona as colunas originais e converte para numérico.
2. `feature_extraction_columns`: cria as variáveis derivadas do notebook (`NEW_AGE_CAT`, `NEW_BMI`, `NEW_GLUCOSE`, `NEW_AGE_BMI_NOM`, `NEW_AGE_GLUCOSE_NOM`, `NEW_INSULIN_SCORE`, `NEW_GLUCOSE_X_INSULIN`, `NEW_GLUCOSE_X_PREGNANCIES`).
3. `add_split_column`: marca cada linha como `train` (70%), `test` (15%) ou `validate` (15%), com semente fixa (`seed: 42`).
4. `fit_encoders` / `transform_encoders`: `LabelEncoder` para as colunas categóricas.
5. `fit_scalers` / `transform_scalers`: `StandardScaler` para as colunas numéricas.

### Decisões

- As variáveis originais não precisam de encoding, porque todas as colunas são `int64` ou `float64`. A única categórica é `Outcome`, que é a target e já vem como 0/1, então não precisa de encoding. O encoding é aplicado posteriormente com as novas variáveis categóricas criadas.

- Os encoders e scalers foram ajustados só no split `train` para que não houvesse vazamento de informação para as bases de teste e validação.

- A variável target (`Outcome`) tem 65% de "negativos" e 35% de "positivos". Para replicar o pipeline visto em aula, introduzimos acurácia, recall, precision, F1 macro e ROC AUC para avaliação dos modelos.

- O notebook diz que não há valores faltantes, mas a análise das variáveis mostrou zeros que não fazem sentido clínico, como visto na tabela abaixo. A remoção dessas linhas foi testada em `transform_scalers`, mas cerca de 30% das linhas seriam perdidas, o que nos levou a optar por mantê-los nesta versão.

| Variável | Linhas com 0 |
|---|---|
| Glucose | 5 |
| BloodPressure | 27 |
| SkinThickness | 192 |
| Insulin | 314 |
| BMI | 8 |

O notebook de referência usou imputação por KNN (k = 5) para esses dados estranhos, mas optamos por não adicionar ao código para não aumentar a complexidade desta entrega.

- Por fim, os encoders e scalers foram salvos em disco (`data/06_models/`) para que o pipeline de inferência use exatamente a mesma transformação do treino e possa rodar sozinho.

## 2. Treinamento

Nós:

1. `train_model`: treina os 9 classificadores usados no notebook: Logistic Regression, Random Forest, KNN, SVC, Decision Tree, AdaBoost, Gradient Boosting, XGBoost e LightGBM.
2. `evaluate_model`: calcula as métricas em `train`, `test` e `validate`.
3. `select_best_model`: escolhe o modelo com maior `roc_auc` no split `validate`.

### Decisões

- Adotamos ROC AUC como a métrica principal de avaliação dos modelos, no validate, para que possamos discutir o threshold e entender melhor quantos casos de diabetes deixamos de detectar (falsos negativos) em troca de menos alarmes falsos (falsos positivos).

- O modelo usa 16 colunas, sendo elas: as 8 numéricas originais, as duas interações `NEW_GLUCOSE_X_INSULIN` e `NEW_GLUCOSE_X_PREGNANCIES` e as 6 categóricas derivadas (`NEW_AGE_CAT`, `NEW_BMI`, `NEW_GLUCOSE`, `NEW_AGE_BMI_NOM`, `NEW_AGE_GLUCOSE_NOM`, `NEW_INSULIN_SCORE`).

- No notebook de referência, essas categóricas são criadas a partir da combinação linear das variáveis numéricas (idade, IMC, glicose e insulina). Como o objetivo do exercício é replicar o notebook em Kedro, e não rediscutir ou otimizar a modelagem, elas foram mantidas e entram no treino, mas com a expectativa de terem baixo poder preditivo, uma vez que são derivadas das variáveis numéricas que o modelo já recebe.

### Resultados

| Modelo | ROC AUC train | ROC AUC validate | ROC AUC test | F1 macro validate |
|---|---|---|---|---|
| **AdaBoostClassifier** | 0.912 | **0.872** | 0.727 | 0.829 |
| LogisticRegression | 0.873 | 0.867 | 0.710 | 0.815 |
| LGBMClassifier | 1.000 | 0.867 | 0.697 | 0.848 |
| RandomForestClassifier | 1.000 | 0.866 | 0.716 | 0.798 |
| SVC | 0.909 | 0.863 | 0.708 | 0.801 |
| XGBClassifier | 1.000 | 0.862 | 0.702 | 0.790 |
| GradientBoostingClassifier | 0.996 | 0.855 | 0.707 | 0.772 |
| KNeighborsClassifier | 0.919 | 0.831 | 0.680 | 0.716 |
| DecisionTreeClassifier | 1.000 | 0.667 | 0.616 | 0.674 |

Tamanho dos splits: 452 train, 109 test, 91 validate.

O modelo escolhido foi o AdaBoost, mas devido ao tamanho da base (652 linhas na base de modelagem, 452 no split de treino, e 91 linhas no `validate`), os modelos tiveram uma performance muito parecida. Adicionalmente, não executamos o processo de validação cruzada, uma vez que não foi solicitada no enunciado, mas entendemos que seria o passo óbvio no desenvolvimento de um modelo real. Observamos também que alguns modelos chegam a ROC AUC 1.0 no treino, o que indica que eles apenas "decoraram" os padrões da base de treino.

## 3. Inferência

Nós:

1. `clean_inference_data` e `enrich_inference_data`: mesmas funções da engenharia de dados.
2. `encode_inference_data` e `scale_inference_data`: aplicam os encoders e scalers salvos no treino, sem reajustar.
3. `predict`: usa o `best_model` e gera, para cada linha, a classe prevista e a probabilidade de diabetes.

### Decisões

- Utilizamos, mais uma vez, as funções do pipeline de data engineering, para que a limpeza e a criação de features sejam padronizadas.
- O resultado final é um JSON com as colunas `index`, `prediction` e `probability`: na base de inferência (116 linhas), 38 foram classificadas como diabetes.
