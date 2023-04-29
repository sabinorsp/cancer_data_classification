
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




## Traning models 

# Evaluation models

# Conclusions