# Exploratory Data Analysis:

## About the project and data: 
O objetivo desse projeto foi realizar várias etapas de um projeto de ciencia de dados, passando pela etapa de análise exploratória, realizando o pré-processamento dos dados, realizar seleção de modelos, avaliar os modelos e selecionar o modelo final de classifição. Também teve como objetivo a implementação e utilização de um padrão de projeto para ciencia de dados embasado no projeto  cookiecutter data science project template.

Os dados selecionados foram retirados do Kaggle ref: [source]('https://www.kaggle.com/datasets/erdemtaha/cancer-data'). Representam caratetisticas geométricas de células cancerígenas que foram classificados em benigina e malígina. 
## Struture data and some informations: 

O conjunto original do dataset é composto por 30 colunas/caracteristicas e 570 registros. Todas as caracteristicas exceto a variável alvo `diagnosis` são variáveis numéricas contínuas. A seguir avaliaremos os dados ausentes e ou duplicações.  

### Missing Values and duplicated

O conjunto de dados não possui valores ausentes ou valores duplicados, conforme observado na imagem abaixo. 

<p align="center">
  <img src="figures/fig1-missing-values.png" width="700" height="400">
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
  <img src="figures/fig2-skewness.png" width="700" height="400">
</p>

* Curtosi:

<p align="center">
  <img src="figures/fig3-kurtosis.png" width="700" height="400">
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

O mapa de calor mostra que há a presença de várias colunas com alto grau de correlação. Esse fator gera redundancia de informação ou multicolinearidade o que prejudica a identificação de importancia das variáveis para predizer o alvo quando treinado o modelo. O que pode explicar a alta correção entre as variáveis é devido praticamente todas serem derivadas de uma caracteristica geométrica da célula. 

Para uma representação gráfica das relações entre as multiplas variáveis temos o plot RadViz: 

<p align="center">
  <img src="figures/radviz-30-features.png" width="900" height="500">
</p>

Com o objetivo de reduzir essa multicolinearidade, iremos realizar o filtro para colunas que possuem alta correção, acima de 0.95 e excluir a colinearidade escolhendo somente um grupo de colunas para permanecer no conjunto de dados. Será criado um novo dataset para essa transformação e ficará salvo em data2 para etapas posteriores de preprocessamento e treinamento de modelos. 

A seguir as colunas que apresentam uma alta taxa de correlação: 

```python
correleted_columns = correlated_columns(data.drop('diagnosis', axis=1))
correleted_columns
```
           level_0          level_1   pearson
    0    perimeter_mean      radius_mean  0.997855
    1         area_mean      radius_mean  0.987357
    2         area_mean   perimeter_mean  0.986507
    3      perimeter_se        radius_se  0.972794
    4           area_se        radius_se  0.951830
    5      radius_worst      radius_mean  0.969539
    6      radius_worst   perimeter_mean  0.969476
    7      radius_worst        area_mean  0.962746
    8   perimeter_worst      radius_mean  0.965137
    9   perimeter_worst   perimeter_mean  0.970387
    10  perimeter_worst        area_mean  0.959120
    11  perimeter_worst     radius_worst  0.993708
    12       area_worst        area_mean  0.959213
    13       area_worst     radius_worst  0.984015
    14       area_worst  perimeter_worst  0.977578


```python
# Create a second dataset data2 with contain the drop about correlated columns level1
data2 = data.drop(correleted_columns.level_1.unique(), axis=1)
```

Portanto ficamos com dois modelos de dados para a etapa de pre-processamento e treinamento dos modelos. Os datasets que sofreram transformações ficaram salvos no diretório `./data/interim`. 
  * `data.csv`: Não contém colunas `id`, alteração classes `diagnosis` para (0,1).  
  * `data2.csv`: Cópia de `data.csv` com a exclusão de colunas com altas correlações.   

### Sobre o balanceamento de classe variável alvo: 

Para verificação do balanço de classe na variável alvo temos: 

<p align="center">
  <img src="figures/balance_diagnosis.png" width="500" height="400">
</p>

# END