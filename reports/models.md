
# Models

## Pre-Processing

Foi utilizado 20% dos dados para divisão dos dados de teste, conforme a seguir: 

```python
# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=40)
X_train2, X_test2= train_test_split(X_2,test_size=0.2, random_state=40)
```

Foi aplicado para todas as variáveis a normalização usando o `MinMaxScaler` do pacote `scikit-learn`.

```python
# Apply standard to each X's data: 
scaler = MinMaxScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.fit_transform(X_test)
X_train_scaler2 = scaler.fit_transform(X_train2)
X_test_scaler2 = scaler.fit_transform(X_test2)
```
Os dados ficaram portanto divididos em 4 subsets para o treinamento e teste, sendo os respectivos dados para cada grupo de dados `data.csv` e `data2.csv`. 

A seguir foi realizado a seleção de variáveis utilizando a seleção das melhores importancias com o uso do RandomForest com critério gini, calculando a média de todas importancias e selecionando as variaveis que possuem valores acima dessa média. Os resutlados podem ser observados conforme as figuras abaixo:  

* X_train - data.csv:

<p align="center">
  <img src="figures/feature_data.png" width="500" height="400">
</p>

* X_train - data2.csv: 

<p align="center">
  <img src="figures/feature_data2.png" width="500" height="400">
</p>

Para o conjunto de dados `data.csv` 9 variáveis apresentaram destaque no grau de importancia contra 7 variáveis para o `data2.csv`. Os respectivos dados de treinamento e teste para cada conjunto preprocessado foram salvos no diretório `./data/processed/`. Dados que serão utilizados pelo algoritmo para realizar as seleções de modelo, treinamentos, otimizações e avaliações finais do modelo. Abaixo segue um resumo da estrutura. 

* `data.csv`: 
    * X_data: 
        * X_train.csv
        * X_test.csv
        * X_train_feature.csv
        * X_test_feature.csv

* `data2.csv`:
    * X_data2:
        * X_train2.csv
        * X_test2.csv
        * X_train2_features.csv
        * X_test2_features.csv

* Variável alvo: 
    * y_train.csv
    * y_test.csv

Com todos os conjuntos de dados previamente processados, iremos prosseguir para a construções dos modelos, seleções e suas avaliações.

## Select Models

Para realizarmos a seleção do modelo de acordo com cada agrupamento de dados, os que foram feito seleção de variáveis e os que permaneceram com todas as variáveis dado os dois sub conjunto de dados em `data.csv`e `data2.csv` será realizado a divisão e nomeado em dois grupos principais de modelos: Grupo Alpha para todos os dados referente a `data.csv` e Grupo  Beta para todos em `data2.csv`. 

Em todos os casos será aplicado a cross-validation sobre os dados de treinamento com avaliação sobre a métrica AUC scoring, utilizando o método k-fold com n_splits=10 para a lista de modelos a seguir: 

```python
models = [RandomForestClassifier, 
          MultinomialNB, 
          LogisticRegression,
          KNeighborsClassifier,
          xgb.XGBClassifier,
          SVC]
```
Função para a aplicação da cross-validation:

```python 
def evaluate_models(models, X_train, y_train):
    for model in models:
        cls = model()
        kfold = KFold(n_splits=10, random_state=42, shuffle=True)
        s = cross_val_score(cls, X_train, y_train, scoring='roc_auc', cv=kfold)
        print(f"{model.__name__:22} AUC: "
              f"{s.mean():.3f} STD: {s.std():.2f}")
```

A seguir segue os resultados de acordo com o agrupamento dos modelos Alpha e Beta:

### Modelos grupo Alpha: 

```python
# All data: X_train.
RandomForestClassifier AUC: 0.988 STD: 0.02
MultinomialNB          AUC: 0.936 STD: 0.04
LogisticRegression     AUC: 0.994 STD: 0.01
KNeighborsClassifier   AUC: 0.988 STD: 0.02
XGBClassifier          AUC: 0.994 STD: 0.01
SVC                    AUC: 0.995 STD: 0.01
```
  
```python
# Feature data: X_train_feature.
RandomForestClassifier AUC: 0.984 STD: 0.02
MultinomialNB          AUC: 0.852 STD: 0.07
LogisticRegression     AUC: 0.986 STD: 0.02
KNeighborsClassifier   AUC: 0.976 STD: 0.02
XGBClassifier          AUC: 0.982 STD: 0.02
SVC                    AUC: 0.986 STD: 0.02
```

### Modelos grupo Beta:

```python
# All data: X_train2
RandomForestClassifier AUC: 0.994 STD: 0.01
MultinomialNB          AUC: 0.962 STD: 0.03
LogisticRegression     AUC: 0.988 STD: 0.02
KNeighborsClassifier   AUC: 0.979 STD: 0.03
XGBClassifier          AUC: 0.995 STD: 0.01
SVC                    AUC: 0.994 STD: 0.01
```

```python
# Feature data: X_train2_feature
RandomForestClassifier AUC: 0.989 STD: 0.01
MultinomialNB          AUC: 0.736 STD: 0.10
LogisticRegression     AUC: 0.973 STD: 0.04
KNeighborsClassifier   AUC: 0.967 STD: 0.04
XGBClassifier          AUC: 0.988 STD: 0.01
SVC                    AUC: 0.985 STD: 0.02
```

Contudo, os modelos ensamble apresentaram bons desempenhos os praticamente todos os casos, os modelos que foram selecionados para treinamentos e otimizações de parametros foram: RandomForest, SVC e XGBC. A seguir apresentamos os parametros encontrados na otimização.

## Traning models 

Para a etapa de otimização dos parametros dos modelos, neste caso, utilizamos GridSearch, onde será criado um modelo para cada combinação de parametros e selecionado o melhor. A escolha de aplicar este método para esse caso foi pela simplicidade e também pelo tamanho dos dados disponíveis para treinamento. Observando que para modelos mais complexos com superiores quantidades de dados, outros métodos de otimização randomica passam a ser mais interessantes. 

### Parametros para modelos Grupo Alpha: 

```python
# RandomForest to X_train
{'criterion': 'gini', 'max_features': 'log2', 'n_estimators': 200}
Model Saved: ../models/model_alpha_rf.pkl

# RandomForest to X_train_feature
{'criterion': 'entropy', 'max_features': 'sqrt', 'n_estimators': 100}
Model Saved: ../models/model_alpha_rf_feature.pkl

# XGBC to X_train
{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 150, 'subsample': 0.5}
Model Saved: ../models/model_alpha_xgb.pkl

# XGBC to X_train_feature
{'learning_rate': 0.001, 'max_depth': 3, 'n_estimators': 150, 'subsample': 0.5}
Model Saved: ../models/model_alpha_xgb_feature.pkl

# SVC to X_train
{'C': 10, 'degree': 2, 'kernel': 'rbf'}
Model Saved: ../models/model_alpha_svc.pkl

# SVC to X_train_feature
{'C': 10, 'degree': 2, 'kernel': 'poly'}
Model Saved: ../models/model_alpha_svc_feature.pkl
```

### Parametros para modelos Grupo Beta:

```python
# RandomForest to X_train2
{'criterion': 'entropy', 'max_features': 'sqrt', 'n_estimators': 100}
Model Saved: ../models/model_beta_rf.pkl

# RandomForest to X_train2_feature
{'criterion': 'gini', 'max_features': 'sqrt', 'n_estimators': 200}
Model Saved: ../models/model_beta_rf_feature.pkl

# XGBC to X_train2
{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 150, 'subsample': 0.5}
Model Saved: ../models/model_beta_xgb.pkl

# XGBC to X_train2_feature
{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.5}
Model Saved: ../models/model_beta_xgb_feature.pkl

# XGBC to X_train2
{'C': 10, 'degree': 2, 'kernel': 'rbf'}
Model Saved: ../models/model_beta_svc.pkl

#XGBC to X_train2_feature
{'C': 10, 'degree': 2, 'kernel': 'rbf'}
Model Saved: ../models/model_beta_svc_feature.pkl
```
Definido os melhores parametros para esse grupo de configurações, os modelos foram treinados com os dados de treino de seus respectivos grupos e salvos em suas versões .pkl. A seguir os modelos foram carregados e avaliados com seus respectivos dados de testes. 

## Evaluation models


# Conclusions