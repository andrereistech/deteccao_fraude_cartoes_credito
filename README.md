# 💳 Detecção de Fraudes em Transações com Cartão de Crédito

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Highly%20Optimized-green.svg)
![SHAP](https://img.shields.io/badge/XAI-SHAP%20Explainer-red.svg)

Este repositório consolida o desenvolvimento de uma infraestrutura completa de Engenharia de Dados e Machine Learning para a detecção precoce de fraudes financeiras, visando otimizar a precisão operacional e mitigar o impacto de falsos positivos. A solução aborda o desafio clássico de trabalhar com conjuntos de dados massivos e severamente desbalanceados, avaliando e comparando o desempenho de técnicas estatísticas de reamostragem frente a modelos preditivos avançados baseados em `Gradient Boosting`.

A base de dados real utilizada contém transações feitas por portadores de cartões europeus e está disponível publicamente em: [GoogleApis (TensorFlow Data)](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv).

---

## 👥 Autoria e Créditos

### 👨‍💻 Desenvolvedor
- **André Reis**
- 🌐 [LinkedIn](https://www.linkedin.com/in/andre-reis-tech)
- 📧 [GitHub](https://github.com/andrereistech)

### 👩‍🏫 Orientação e Coordenação
- **Professora:** Dra. Isadora Garcia Ferrão
- 🌐 [LinkedIn](https://www.linkedin.com/in/isadora-ferrao/)
- 📧 [GitHub](https://github.com/IsadoraFerrao)

### 🏫 Instituição
- 🌐 [DIO](https://www.dio.me/)
- 🌐 [Bootcamp Bradesco - GenAI, Dados & Cyber](https://web.dio.me/track/bradesco-dados-ciberseguranca-genai)
- 📧 [LinkedIn](https://www.linkedin.com/school/dio-me/)

---

## 📈 Impacto de Negócio & Objetivos
Em sistemas financeiros, um modelo ingénuo focado apenas em Acurácia falharia drasticamente ao ignorar a classe minoritária. Este projeto foca em otimizar a relação entre:
* **Recall (Sensibilidade):** Minimizar os Falsos Negativos (fraudes não detectadas que geram prejuízo direto).
* **Precision (Precisão):** Minimizar os Falsos Positivos (alarmes falsos que bloqueiam cartões de clientes legítimos e saturam a equipa de análise manual).

---

## 📌 Sumário
1. [Visão Geral do Problema e Dados](#1-visão-geral-do-problema-e-dados)
2. [Análise do Desbalanceamento da Classe Target](#2-análise-do-desbalanceamento-da-classe-target)
3. [Feature Engineering e Pré-processamento](#3-feature-engineering-e-pré-processamento)
4. [Modelo Baseline (Regressão Logística)](#4-modelo-baseline-regressão-logística)
5. [Métricas de Avaliação de Modelos](#5-métricas-de-avaliação-de-modelos)
6. [Técnicas de Balanceamento de Dados](#6-técnicas-de-balanceamento-de-dados)
7. [Pipelines e Ajuste de Threshold](#7-pipelines-e-ajuste-de-threshold)
8. [Modelo Avançado (XGBoost)](#8-modelo-avançado-xgboost)
9. [Análise de Feature Importance](#9-análise-de-feature-importance)
10. [Otimização de Hiperparâmetros (GridSearchCV)](#10-otimização-de-hiperparâmetros-gridsearchcv)
11. [Explicabilidade do Modelo com SHAP](#11-explicabilidade-do-modelo-com-shap)

---

## 🛠️ Como Executar o Projeto

```bash
# Clonar o repositório
git clone [https://github.com/andrereistech/seu-repositorio.git](https://github.com/andrereistech/seu-repositorio.git)

# Instalar as dependências necessárias
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib shap

```
---

## 1. Visão Geral do Problema e Dados

Em problemas de detecção de fraude, o volume de transações legítimas supera drasticamente o volume de transações fraudulentas. Os dados fornecidos possuem transformações PCA (Componentes Principais) aplicadas sobre as features originais por motivos de confidencialidade (representadas por $V_1, V_2, \dots, V_{28}$), restando apenas as colunas `Time` e `Amount` com seus valores originais intactos.

### Carregamento inicial do Dataset:

```python

import pandas as pd

url = "[https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)"

df = pd.read_csv(url)

# Visualizando as primeiras linhas

df.head(10)

```

---

## 2. Análise do Desbalanceamento da Classe Target

### A distribuição da variável Class revela o tamanho do desafio: as fraudes constituem uma fração minúscula de todo o conjunto de dados.

```python

df["Class"].value_counts(normalize=True)

```

### Distribuição Percentual aproximada:

> Classe 0 (Legítima): $99.82\%$

> Classe 1 (Fraudulenta): $0.17\%$

---

> [!Tip]
> Se criarmos um modelo ingênuo que classifique 100% das transações como legítimas (Classe 0), sua acurácia será de $99.82\%$, porém o modelo será completamente inútil para o negócio porque não detectará nenhuma fraude.

---

## 3. Feature Engineering e Pré-processamentoPara mitigar a variância e a diferença de escalas nas colunas originais (Amount), aplicamos técnicas de transformação logarítmica e padronização (Z-score normalization).

```python

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

# 1. Transformação Logarítmica para suavizar distribuições muito distorcidas

df["Amount_log"] = np.log1p(df["Amount"])

# 2. Padronização da escala da variável Amount

scaler = StandardScaler()

df["Amount_Scaled"] = scaler.fit_transform(df[["Amount"]])

# 3. Divisão Estratificada em treino e teste (mantendo a proporção de fraudes nas duas frações)

X = df.drop("Class", axis=1)

y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(

    X, y, stratify=y, test_size=0.3, random_state=42

)

```

---

## 4. Modelo de Linha de Base (Baseline) - Regressão Logística

Como primeiro experimento, treinamos um estimador linear simples de Regressão Logística para servir como referencial de performance.

```python

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

# Instanciação e ajuste do modelo

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predição nos dados de validação/teste

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

```

---

## 5. Métricas de Avaliação de Modelos

Dado o forte desbalanceamento, a acurácia é descartada. Focamos nosso esforço analítico nas seguintes ferramentas:

- Recall (Sensibilidade): Mede a proporção de fraudes reais que o modelo conseguiu identificar. É a métrica de negócio mais crítica (minimiza Falsos Negativos).

- Precision (Precisão): Avalia a taxa de acerto do modelo quando ele dispara um alarme de fraude (minimiza Falsos Positivos e retrabalho de análise manual).

- F1-Score: Média harmônica ponderada entre Precision e Recall.

- Curvas ROC (AUC) e Precision-Recall:

````Python

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

import matplotlib.pyplot as plt

# Obtendo probabilidades da classe positiva

y_probs = model.predict_proba(X_test)[:, 1]

# Curva ROC

fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, label=f"AUC: {roc_auc_score(y_test, y_probs):.4f}")

plt.title("ROC Curve")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.show()

# Curva Precision-Recall

precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(6,4))

plt.plot(recall, precision)

plt.title("Precision-Recall Curve")

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.show()

`````

---

## 6. Técnicas de Balanceamento de Dados

Para evitar que os algoritmos sejam enviesados pela classe majoritária, aplicamos duas técnicas clássicas de reamostragem:

### - A. Random Undersampling

Reduz o número de transações legítimas para se igualar à quantidade de fraudes existentes.

```Python

fraudes = df[df["Class"] == 1]

nao_fraudes = df[df["Class"] == 0].sample(len(fraudes), random_state=42)

df_under = pd.concat([fraudes, nao_fraudes])

```

### - B. SMOTE (Synthetic Minority Over-sampling Technique)Gera dados artificiais sintéticos da classe minoritária (fraudes) baseando-se em vizinhos mais próximos ($k$-NN).

```Python

from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_res, y_res = smote.fit_resample(X, y)

```

### - C. Abordagem por Pesos Intrinsecamente Balanceados (Random Forest)

Podemos também delegar ao próprio algoritmo o ajuste dos pesos associados a cada classe. 

```Python

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(

    n_estimators=50,

    max_depth=10,

    class_weight="balanced",

    random_state=42

)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))

```

---

## 7. Técnicas Avançadas: Pipelines e Ajuste de Threshold

### Uso de Pipelines do Scikit-Learn

Garante o encapsulamento do pré-processamento evitando vazamento de dados (data leakage).

```Python

from sklearn.pipeline import Pipeline

pipeline = Pipeline([

    ("scaler", StandardScaler()),

    ("model", LogisticRegression(max_iter=1000))

])

pipeline.fit(X_train, y_train)

y_pred_pipeline = pipeline.predict(X_test)

```

### Customização do Limiar de Decisão (Threshold Tuning)

Por padrão, classificadores usam $0.5$ como ponto de corte. Reduzir esse valor nos ajuda a capturar mais fraudes ocultas (aumentando o Recall).

```Python

threshold = 0.3

y_pred_custom = (y_probs > threshold).astype(int)

print(classification_report(y_test, y_pred_custom))

```

---

## 8. Modelo Avançado - XGBoostO

XGBoost (Extreme Gradient Boosting) destaca-se pelo alto poder preditivo através do aprendizado sequencial de árvores de decisão fracas.

```Python

from xgboost import XGBClassifier

xgb = XGBClassifier(

    scale_pos_weight=10,  # Fator corretivo de peso para equilibrar a classe positiva

    use_label_encoder=False,

    eval_metric="logloss"

)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print(classification_report(y_test, y_pred_xgb))

```

---

## 9. Importância das Variáveis (Feature Importance)

Mapeamento estrutural interno para visualizar quais componentes latentes do PCA ou transformações de valores carregam maior impacto na segregação das classes.

```Python

importances = xgb.feature_importances_

plt.figure(figsize=(10, 5))

plt.bar(range(len(importances)), importances)

plt.title("Importância das Variáveis no XGBoost")

plt.xlabel("Índice da Feature")

plt.ylabel("Ganho de Importância Relative")

plt.show()

```

---

## 10. Otimização de Hiperparâmetros (GridSearchCV)

Varredura sistemática no espaço amostral de hiperparâmetros com o objetivo explícito de maximizar o Recall.

```Python

from sklearn.model_selection import GridSearchCV

param_grid = {

    "max_depth": [3, 5],

    "n_estimators": [50, 100]

}

grid = GridSearchCV(

    XGBClassifier(eval_metric="logloss"),

    param_grid,

    scoring="recall",

    cv=3

)

grid.fit(X_train, y_train)

print("Melhores Hiperparâmetros Identificados:", grid.best_params_)

```

---

## 11. Explicabilidade do Modelo com SHAP

A biblioteca SHAP (SHapley Additive exPlanations) quebra a característica de "caixa preta" do modelo baseado em árvores complexas, atribuindo a cada variável um valor de contribuição exato para a decisão final de score da transação.


```Python

import shap

# Computando explicações locais utilizando a teoria dos jogos de Shapley

explainer = shap.Explainer(xgb)

shap_values = explainer(X_test[:100])

# Plot estruturado de impacto de barra

shap.plots.bar(shap_values)

```

---

> [!TIP]
> Sinta-se livre para clonar este projeto, criar novas colunas derivadas e testar novas arquiteturas de redes neurais profundas!
