# Module 2 — Analytics Pipeline

## Overview

This module performs an end-to-end analytics and machine-learning workflow
using the Titanic dataset.

The dataset was loaded exactly once using Seaborn in `01_eda.ipynb` and
immediately saved as `titanic.csv`. The modeling notebook reads this saved
CSV rather than loading the dataset again.

## Part A — Exploratory Data Analysis

### Dataset Profiling

The dataset was examined using:

- `df.info()`
- `df.describe()`
- `df.shape`

### Missing Values

Measured missing-value percentages:

age            19.87
embarked        0.22
deck           77.22
embark_town     0.22

The required threshold strategy was applied:

- Under 5% missing: affected rows were dropped.
- 5%–30% missing: values were imputed.
- Very high missingness: the column was dropped when reliable imputation
  was not appropriate.

For this dataset, age was median-imputed, rows with missing embarkation
information were removed, and the deck column was dropped because of its
very high missing percentage.

### Univariate Analysis

Histograms and box plots were produced for `age` and `fare`.

Outlier counts were calculated using the IQR rule:

`Q1 - 1.5 × IQR` to `Q3 + 1.5 × IQR`.

Fare mean, median and mode were also calculated to determine the shape of
the fare distribution.

### Bivariate Analysis

Survival rates were calculated by:

- sex
- passenger class
- sex and passenger class together

The correlation matrix used exactly:

- survived
- pclass
- age
- sibsp
- parch
- fare

The derived Boolean columns `adult_male` and `alone` were intentionally
excluded.

### Multivariate Analysis

Four charts were created:

1. Survival by sex
2. Survival by passenger class
3. Survival by sex and passenger class
4. Fare distribution by survival

Each chart is accompanied by a written interpretation in `01_eda.ipynb`.

### Standardization

Age and fare were standardized using `StandardScaler`.

The transformed variables were checked to confirm approximately:

- mean = 0
- standard deviation = 1

This was only an exploratory EDA check and was not reused for model
training.

---

## Part B — Classification

The classification target is `survived`.

Features used:

- pclass
- sex
- age
- sibsp
- parch
- fare
- embarked

A stratified train/test split was performed before preprocessing.

All preprocessing is contained inside scikit-learn pipelines, ensuring
that imputation, encoding and scaling are fitted only on training data.

### Models

Three classifiers were trained:

- Logistic Regression
- Decision Tree
- Random Forest

### Classification Model Comparison

| Model               |   Accuracy |   Precision |   Recall |     F1 |    AUC |
|:--------------------|-----------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.8045 |      0.7931 |   0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |     0.7933 |      0.8636 |   0.5507 | 0.6726 | 0.8292 |
| Random Forest       |     0.8156 |      0.8    |   0.6957 | 0.7442 | 0.83   |

Each model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- ROC curve
- AUC

The Decision Tree was also visualized using `plot_tree`.

---

## Imbalance Handling

Three approaches were compared:

1. Baseline
2. `class_weight="balanced"`
3. SMOTE

SMOTE was applied only to the training data through an imbalanced-learn
pipeline.

### Imbalance Comparison

| Strategy              |   Precision |   Recall |     F1 |
|:----------------------|------------:|---------:|-------:|
| Baseline              |      0.7931 |   0.6667 | 0.7244 |
| Class Weight Balanced |      0.7297 |   0.7826 | 0.7552 |
| SMOTE                 |      0.7397 |   0.7826 | 0.7606 |

The strategy with the strongest F1 score was considered the best balance
between precision and recall in this experiment.

---

## Random Forest Hyperparameter Tuning

`GridSearchCV` was used to tune:

- n_estimators
- max_depth
- max_features

Best parameters:

`{'model__max_depth': 5, 'model__max_features': 'sqrt', 'model__n_estimators': 100}`

Best cross-validation F1:

`0.7459`

OOB score:

`0.8272`

The Random Forest was constructed with `oob_score=True`.

---

## Regression Task

Multivariate Linear Regression was used to predict `fare`.

Results:

- MAE: 20.8094
- RMSE: 30.4731
- R²: 0.3999
- Adjusted R²: 0.3679

A residual plot was produced in `02_modeling.ipynb` to examine whether the
residual spread indicates heteroscedasticity.

---

## Final Classification Selection

The saved classifier is:

**Logistic Regression**

AUC:

**0.8437**

The classifier was selected by comparing performance metrics, particularly
AUC and F1 score.

---

## Saved Pipeline

The complete preprocessing and classification pipeline was saved as:

`best_model.joblib`

The artifact contains both preprocessing and the final estimator.

It was reloaded with `joblib.load()` and successfully produced a prediction
from raw, unprocessed passenger data.

## Files

- `01_eda.ipynb`
- `02_modeling.ipynb`
- `titanic.csv`
- `titanic_cleaned.csv`
- `best_model.joblib`
- `requirements.txt`
- `README.md`

## Running the Module

Install:

`pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib`

Run `01_eda.ipynb` first and then `02_modeling.ipynb`.
