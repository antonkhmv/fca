## Formal Concept Analysis HW

By: Khomyakov Anton

# Task 1

Dataset sources (Kaggle):

1. [Housing Prices](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)
2. [Student Performance](https://www.kaggle.com/datasets/devansodariya/student-performance-data)
3. [Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

The datasets are copied to folders `housing-price-prediction/`, `mobile-price-classification/`, `student-performance-data/`

# Task 2

## Data preparation

Numerical attributes were tranformed using Standard Sclaer from sklearn. Boolean and Categorical attributes were one-hot encoded.

## Paramter tuning

Best parameters were chosen using param grid searches using GridSearchCV

## Code

Data perparation and classification using sklearn, xgboost and CatBoost can be found in the notebook 
- `LazyFCA-sklearn-xgb-catboost.ipynb`

# Task 3 & 4 

## Metrics

I chose `f1_macro` as a metric for comaring the predictions as it captures recall and precision for both classes. `AUC ROC` can't be used as we can't calculate probabilities for each class.

## Data Preparation

The mobile price dataset is downsampled to 512 objects (in pandas: `df.sample(512, random_state=42)`) to speed the calculations up

For datasets with a prediciton task, the target is binnized using `pd.qcut(..., 2)`.

## Parameter tuning

Parameter tuning concerned binarization of data:

- Binning with a parameter of number of bins (Using quantiles)

- One-hot encoding with a parameter of subset of attributes

- Ordinal Encoding with a parameter of subset of attributes

## Code

The code for cross-validation using this metric and grid search implementation can be found in `fca_utils.py`

Data preparation and param tuning for binary and pattern classifiers can be found in the notebooks:

1. `LazyFCA-housing.ipynb`
2. `LazyFCA-mobile.ipynb`
3. `LazyFCA-students.ipynb`

`fcalc/` contains the modified code for LazyFCA, speedup using Pytorch.

## Results table

The comparison between these methods for the best parameters found (cross-validation results):

|    | classifier          |   f1_macro_mobile |   f1_macro_housing |   f1_macro_student |
|---:|:--------------------|------------------:|-------------------:|-------------------:|
|  0 | Naive Bayes         |          0.931445 |           0.783526 |           0.597235 |
|  1 | Decision Tree       |          0.919705 |           0.722525 |           0.486537 |
|  2 | Random Forest       |          0.953039 |           0.751659 |           0.559067 |
|  3 | Logistic Regression |      **0.992171** |           0.783507 |           0.529248 |
|  4 | K Nearest Neighbors |          0.929526 |           0.749931 |           0.564667 |
|  5 | Catboost            |          0.929478 |           0.746898 |           0.54321  |
|  6 | Xgboost             |          0.962794 |           0.777596 |           0.599721 |
|  7 | Binary Classifier   |          0.89936  |           0.807106 |           0.559277 |
|  8 | Pattern Classifier  |          0.915377 |       **0.807762** |       **0.624295** |

So, the pattern classifier gave the best results for 2 of the datasets - Housing and Student.

For the Mobile and Student datasets the most important intersections counts for patterns are in the range 4-6, which might mean that the algorithm overfits on these datasets. The Housing dataset seems to give the most robust solution (relative to other methods) with a lot of support.

The Mobile dataset is best classified by a simple Logistic Regression.