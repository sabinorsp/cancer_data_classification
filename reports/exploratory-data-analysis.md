# Exploratory Data Analysis:

## About the project and data: 
O objetivo desse projeto foi realizar várias etapas de um projeto de ciencia de dados, passando pela etapa de análise exploratória, realizando o pré-processamento dos dados, realizar seleção de modelos, avaliar os modelos e selecionar o modelo final de classifição. Também teve como objetivo a implementação e utilização de um padrão de projeto para ciencia de dados embasado no projeto  cookiecutter data science project template.

Os dados selecionados foram retirados do Kaggle ref: [source]('https://www.kaggle.com/datasets/erdemtaha/cancer-data'). Representam caratetisticas geométricas de células cancerígenas que foram classificados em benigina e malígina. 
## Struture data and some informations: 

O conjunto original do dataset é composto por 30 colunas/caracteristicas e 570 registros. Todas as caracteristicas exceto a variável alvo `diagnosis` são variáveis numéricas contínuas. A seguir avaliaremos os dados ausentes e ou duplicações.  

### Missing Values and duplicated

O conjunto de dados não possui valores ausentes ou valores duplicados, conforme observado na imagem abaixo. 

<p align="center">
  <img src="figures/fig1-missing-values.png" width="500" height="300">
</p>

As variáveis id e Unnamed não são representativas para as análises e modelos, portanto serão excluidas. Para a variável alvo `diagnosis` será feito a mudança das classes pelo seguinte acordo:  
* B = 0  
* M = 1.

Em seguida iremos avaliar os resumos estatísticos, distribuições e correlações dos dados. 

### Summarize Statistics

Os resumos estatisticos como média, desvio padrão e mediana podem ser observados conforme as imagens a seguir: 

```python
summarize = data.describe()
summarize.T[:14]
```

<p align="center">
  <img src="figures/describe0-14.png" width="700" height="300">
</p>


```python
summarize = data.describe()
summarize.T[14:30]
```

<p align="center">
  <img src="figures/describe-14-30.png" width="700" height="300">
</p>

Podemos notar que há diferentes escalas entre os dados, como exemplo valores médios de 0.002 até um máximo de 880. Essas diferenças de escalas podem afetar a importancia que o modelo irá calcular cada variável, na etapa de pré-processamento será tratado essas diferenças para normalizar os dados. 

### Distributions
Procurando realizar uma verificação rápida sobre as distribuições dos dados, realizamos o teste de Shapiro-Wilk para verificação de normalidade de cada distribuição. O código a seguir retorna somente as variáveis que falhamos em rejeitar h0= a distribuição da variável segue uma distribuição normal. 

```python
# Verify to all columns Shapiro-Wilk test, print just the variable that we can reject the H0: 
for col in data.columns:
    stat, p = shapiro(data[col])
    if p >= 0.05:
        print(f'{col}:p-value = {p:.6f}')
```
Como podemos notar, não foi retornado nenhum valor. Portanto rejeitamos a hipotese de que os dados seguem uma distribuição normal. A seguir seguem os plots para as variáveis com os valores de simetria e curtose: 

* Simetria:

<p align="center">
  <img src="fig2-skewness.png" width="500" height="300">
</p>

* Curtosi:

<p align="center">
  <img src="figures/fig3-kurtosis.png" width="500" height="300">
</p>

Para criar uma visualização da distribuição dos dados, como há muitas variáveis e também diferentes tipos de escalas, vamos dividir em grupos de semelhança de acordo com suas médias. Portanto iremos criar os seguintes grupos: 

    * Group A : 0 > mean <=1  
    * Group B : 1 > mean <=20  
    * Group C : 20 > mean <=100  
    * Group D : 100 > mean <= 1000 

Segue o plot de violino para as variáveis a seguir: 

<p align="center">
  <img src="figures/features-violin-plot.png" width="500" height="900">
</p>

### About Correlations: 

<p align="center">
  <img src="figures/heat-map-correlations.png" width="550" height="450">
</p>

O mapa de calor mostra que há a presença de várias colunas com alto grau de correlação. Esse fator gera redundancia de informação ou multicolinearidade o que prejudica a identificação de importancia das variáveis para predizer o alvo quando treinado o modelo.

Para uma representação gráfica das relações entre as multiplas variáveis temos o plot RadViz: 

<p align="center">
  <img src="figures/radviz-30-features.png" width="550" height="450">
</p>



# Models

## Pre-Processing

## Select Models

## Traning models 

# Evaluation models

# Conclusions