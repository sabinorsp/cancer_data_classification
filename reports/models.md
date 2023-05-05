
# Models

## Pre-Processing

20% of the data was used to split the test data, as follows:

```python
# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=40)
X_train2, X_test2= train_test_split(X_2,test_size=0.2, random_state=40)
```

Normalization using the `MinMaxScaler` package from `scikit-learn` was applied to all variables.

```python
# Apply standard to each X's data: 
scaler = MinMaxScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.fit_transform(X_test)
X_train_scaler2 = scaler.fit_transform(X_train2)
X_test_scaler2 = scaler.fit_transform(X_test2)
```
Therefore, the data was divided into 4 subsets for training and testing, with the respective data for each data group being `data.csv` and `data2.csv`.

Next, variable selection was performed using the selection of the best importances with the use of RandomForest with gini criteria, calculating the average of all importances and selecting the variables that have values above this average. The results can be observed in the figures below: 

* X_train - data.csv:

<p align="center">
  <img src="figures/feature_data.png" width="500" height="400">
</p>

* X_train - data2.csv: 

<p align="center">
  <img src="figures/feature_data2.png" width="500" height="400">
</p>

For the dataset `data.csv`, 9 variables showed importance compared to 7 variables for `data2.csv`. The respective training and test data for each preprocessed set were saved in the directory `./data/processed/`. These data will be used by the algorithm to perform model selections, training, optimizations, and final model evaluations. Below is a summary of the structure.

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

* Target variable: 
    * y_train.csv
    * y_test.csv

With all the previously processed datasets, we will proceed to building the models, selections, and their evaluations.

## Select Models

To perform model selection according to each data grouping, those that had variable selection and those that remained with all variables given the two subsets of data in `data.csv` and `data2.csv`, a division will be made and named into two main groups of models: Alpha Group for all data related to `data.csv` and Beta Group for all in `data2.csv`.

In all cases, cross-validation will be applied to the training data with evaluation on the AUC scoring metric, using the k-fold method with n_splits=10 for the following list of models: 

```python
models = [RandomForestClassifier, 
          MultinomialNB, 
          LogisticRegression,
          KNeighborsClassifier,
          xgb.XGBClassifier,
          SVC]
```
Function for applying cross-validation:

```python 
def evaluate_models(models, X_train, y_train):
    for model in models:
        cls = model()
        kfold = KFold(n_splits=10, random_state=42, shuffle=True)
        s = cross_val_score(cls, X_train, y_train, scoring='roc_auc', cv=kfold)
        print(f"{model.__name__:22} AUC: "
              f"{s.mean():.3f} STD: {s.std():.2f}")
```

The following are the results according to the grouping of Alpha and Beta models:

### Alpha group models:: 

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

### Beta group models:

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

However, the ensemble models showed good performance in practically all cases. The models that were selected for training and parameter optimization were: RandomForest, SVC and XGBC. Below we present the parameters found in the optimization.

## Traning models 

For the parameter optimization step of the models, in this case, we used GridSearch, where a model will be created for each combination of parameters and the best one will be selected. The choice of applying this method for this case was due to its simplicity and also due to the size of the data available for training. It should be noted that for more complex models with larger amounts of data, other methods of random optimization become more interesting.

### Parameters for Alpha Group models: 

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

### Parameters for Beta Group models:

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
After defining the best parameters for this configuration group, the models were trained with their respective training data and saved in their .pkl versions. Next, the models were loaded and evaluated with their respective test data.

## Final Model

All models were tested with their respective test data and compared using the AUC metric. The table below shows the results: 

                          AUC
    model_beta_xgb           0.947
    model_alpha_svc          0.947
    model_beta_rf            0.941
    model_alpha_xgb          0.941
    model_beta_svc           0.941
    model_alpha_rf           0.934
    model_alpha_svc_feature  0.934
    model_beta_rf_feature    0.927
    model_alpha_xgb_feature  0.927
    model_beta_svc_feature   0.927
    model_alpha_rf_feature   0.921
    model_beta_xgb_feature   0.921


- The `model_beta_xgb` and `model_alpha_svc` models have the best AUC scores. To select a model, I will choose `model_beta_xgb` because it has variable reduction due to the elimination of columns with high correlation.

```python
model_beta_xgb:
              precision    recall  f1-score   support

          B        0.99      0.92      0.95        75
          M        0.86      0.97      0.92        39

    accuracy                           0.94       114
   macro avg       0.92      0.95      0.93       114
weighted avg       0.94      0.94      0.94       114

AUC:0.947
```

- In this problem, the worst error the model can make is to predict a cell as benign when it is actually malignant. The current recall rate is 97%, compared to a precision rate of 86%.


<button><a href="https://github.com/sabinorsp/cancer_data_classification" class="btn btn-primary" >Return Page</a></button>_____________________________________________________________________________________________________________
<button><a href="https://github.com/sabinorsp/cancer_data_classification/blob/master/reports/exploratory-data-analysis.md" class="btn btn-primary" >Exploratory Data Analysis</a></button>