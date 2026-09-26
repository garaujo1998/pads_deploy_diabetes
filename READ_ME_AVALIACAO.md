# Previsão de incidência de diabetes com Kedro

Este documento descreve como os pipelines foram construídos a partir do notebook `diabetes-prediction.ipynb` e por que cada decisão foi tomada.

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

**Sem encoding nas variáveis originais.** Segundo o notebook, todas as colunas originais são `int64` ou `float64`. A única categórica é `Outcome`, que é a target e já vem como 0/1, então não precisa de encoding. O encoding só se aplica às variáveis categóricas criadas no passo 2.

**Encoders e scalers ajustados só no split `train`.** O parâmetro `split_to_fit: ['train']` garante que média, desvio e categorias venham apenas do treino, sem vazamento de informação de teste e validação.

**Encoders e scalers salvos em disco.** Ficam em `data/06_models/` como `.pkl` para que o pipeline de inferência use exatamente a mesma transformação do treino e possa rodar sozinho.

**Valores zero mantidos.** A análise das variáveis mostrou zeros que não fazem sentido clínico:

| Variável | Linhas com 0 |
|---|---|
| Glucose | 5 |
| BloodPressure | 27 |
| SkinThickness | 192 |
| Insulin | 314 |
| BMI | 8 |

O notebook diz que não há valores faltantes, então esses zeros provavelmente são NAs preenchidos com 0. A remoção dessas linhas foi testada em `transform_scalers` (o código está comentado): a base cairia de 652 para 451 linhas, uma perda de cerca de 30%. Por isso os zeros foram mantidos nesta versão. O notebook de referência usou imputação por KNN (k = 5), que fica como próximo passo.

**Balanceamento.** A target tem 65% de "não" e 35% de "sim". Não foi aplicado rebalanceamento. Por isso, além da acurácia, são reportados recall, precision, F1 macro e ROC AUC.

## 2. Treinamento

Nós:

1. `train_model`: treina os 9 classificadores usados no notebook: Logistic Regression, Random Forest, KNN, SVC, Decision Tree, AdaBoost, Gradient Boosting, XGBoost e LightGBM.
2. `evaluate_model`: calcula as métricas em `train`, `test` e `validate`.
3. `select_best_model`: escolhe o modelo com maior `roc_auc` no split `validate`.

### Decisões

**Modelos definidos por configuração.** A lista de modelos fica em `conf/base/parameters_modelling.yml` (`class_path`), e a classe é importada dinamicamente. Isso permite adicionar ou remover modelos sem mexer no código. Os argumentos de cada modelo ficam em `init_args`, separados por modelo. O SVC precisa de `probability: true` para calcular a ROC AUC.

**Métricas iguais às do notebook:** accuracy, recall, precision, F1 macro e ROC AUC, mais `n_samples` para conferir o tamanho de cada split.

**Critério de escolha: ROC AUC no `validate`.** A ROC AUC não depende de um ponto de corte e é menos sensível ao desbalanceamento de 65/35 do que a acurácia. O `validate` é usado para escolher o modelo, e o `test` fica como verificação independente.

**Features usadas.** O modelo usa as 10 colunas numéricas (as 8 originais e as duas interações `NEW_GLUCOSE_X_INSULIN` e `NEW_GLUCOSE_X_PREGNANCIES`). As categóricas derivadas são criadas e codificadas, mas não entram no treino nesta versão.

### Resultados

| Modelo | ROC AUC train | ROC AUC validate | ROC AUC test | F1 macro validate |
|---|---|---|---|---|
| **LGBMClassifier** | 1.000 | **0.887** | 0.691 | 0.798 |
| LogisticRegression | 0.855 | 0.884 | 0.723 | 0.787 |
| AdaBoostClassifier | 0.911 | 0.871 | 0.726 | 0.826 |
| SVC | 0.918 | 0.869 | 0.700 | 0.826 |
| RandomForestClassifier | 1.000 | 0.863 | 0.700 | 0.801 |
| XGBClassifier | 1.000 | 0.856 | 0.690 | 0.776 |
| GradientBoostingClassifier | 0.995 | 0.855 | 0.717 | 0.812 |
| KNeighborsClassifier | 0.919 | 0.798 | 0.640 | 0.732 |
| DecisionTreeClassifier | 1.000 | 0.707 | 0.639 | 0.712 |

Tamanho dos splits: 452 train, 109 test, 91 validate.

O modelo escolhido foi o LightGBM. Pontos de atenção:

- **A base é pequena.** São 652 linhas na base de modelagem, e só 452 no split de treino. Com 91 linhas no `validate`, a diferença entre os primeiros colocados fica dentro do ruído.
- **Não foi feita validação cruzada.** Ela não foi pedida no enunciado, mas seria o passo óbvio no desenvolvimento de um modelo real, justamente por causa do tamanho da base.
- **Parte dos modelos decora o treino.** LightGBM, Random Forest, XGBoost e Decision Tree chegam a ROC AUC 1.0 no treino. Com os parâmetros padrão, eles decoram os dados, e escolher pelo `validate` acaba premiando quem teve sorte no ruído.
- A Regressão Logística fica quase empatada no `validate` (0.884), tem o menor overfitting e é melhor no `test` que o LightGBM.

## 3. Inferência

Nós:

1. `clean_inference_data` e `enrich_inference_data`: mesmas funções da engenharia de dados.
2. `encode_inference_data` e `scale_inference_data`: aplicam os encoders e scalers salvos no treino, sem reajustar.
3. `predict`: usa o `best_model` e gera, para cada linha, a classe prevista e a probabilidade de diabetes.

### Decisões

**Reuso das funções de engenharia de dados.** A limpeza e a criação de features são as mesmas funções do treino, então os dados de inferência passam exatamente pela mesma transformação.

**Saída em JSON** com `index`, `prediction` e `probability`. Na base de inferência (116 linhas), 47 foram classificadas como diabetes.

## Problemas encontrados

- **Só 3 dos 9 modelos rodavam.** O XGBoost não carregava porque faltava o OpenMP no macOS (`brew install libomp`). Depois disso, a avaliação quebrava no SVC, que não tem `predict_proba` por padrão. A solução foi separar `init_args` por modelo e passar `probability: true` ao SVC.
- **A inferência não rodava sozinha.** Encoders e scalers ficavam só em memória. Foram adicionados ao `catalog.yml` como pickle.

## Próximos passos

- Imputar os zeros com KNN (k = 5), como no notebook, em vez de mantê-los.
- Tratar outliers.
- Escolher o modelo por validação cruzada estratificada em vez de um único split.
- Otimizar hiperparâmetros, principalmente dos modelos com overfitting.
- Avaliar o uso das variáveis categóricas derivadas no treino.
