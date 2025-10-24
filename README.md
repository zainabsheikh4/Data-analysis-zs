# Data-analysis-zs

## Introduction

This project focuses on predicting key performance indicators of 3D‑printed parts—**surface roughness**, **tensile strength**, and **elongation**—based on various printer settings and material parameters. We explore a variety of regression techniques, compare their performance, and interpret their behaviour through quantitative metrics and visual diagnostics.

Key objectives:

1. Clean and prepare the dataset for modelling.  
2. Implement baseline and advanced regression models (Linear, Ridge, Lasso, Polynomial, Decision Tree, KNN, SVR).  
3. Apply cross‑validation and hyperparameter tuning to improve generalisation.  
4. Analyse feature importance and residuals to understand each model’s decision logic.  
5. Provide at least one custom implementation “from scratch” to demonstrate core algorithm understanding.

---

## Notebook Structure

1. **data_preprocessing.ipynb**  
   - Loads the raw CSV.  
   - Checks for missing values, duplicates, and outliers.  
   - Encodes categorical features and scales numerical inputs.  
   - Saves the cleaned dataset as `clean_dataset.csv`.

2. **linear_regression.ipynb**  
   - Baseline linear models on all three targets.  
   - Evaluation using train/test R², MAE, RMSE, and residual plots.

3. **ridge_regression.ipynb**  
   - Linear regression with L2 regularisation.  
   - 5‑fold cross‑validation and discussion of regularisation effects.

4. **lasso_regression.ipynb**  
   - Linear regression with L1 regularisation.  
   - Coefficient sparsity analysis and cross‑validation.

5. **polynomial_regression.ipynb**  
   - Polynomial regression (degrees 2 and 3).  
   - Residual diagnostics and cross‑validation for each degree.

6. **decision_tree.ipynb**  
   - Decision Tree regression with max‑depth tuning.  
   - Feature importance analysis and cross‑validation.

7. **knn_regression.ipynb**  
   - K‑Nearest Neighbours regression with scaling.  
   - Grid‑search hyperparameter tuning and cross‑validation.

8. **svr_model.ipynb**  
   - Support Vector Regression with RBF kernel.  
   - Hyperparameter tuning via GridSearchCV and cross‑validation.

9. **manual_poly_regression.ipynb**  
   - Custom 2nd‑degree polynomial regression implemented in NumPy.  
   - Comparison with scikit‑learn’s `PolynomialFeatures` + `LinearRegression`.

---

## Graphs & Visualisations

- **Correlation Heatmap**: Pairwise correlation of all features.  
- **Distribution Plots**: Histograms/KDEs of each variable pre- and post-cleaning.  
- **Residual Histograms**: Error distributions for each model and target.  
- **Actual vs. Predicted Scatter**: For each model/target combination.  
- **Cross‑Validation Score Plots**: R² per fold and mean/std indicators.  
- **Feature Importance Bars**: For tree‑based models and coefficient magnitudes.  
- **Hyperparameter Tuning Curves**: Model performance vs. parameter values.

---

## How to Use

1. **Clone or unzip** this repository so that all `.ipynb` files and `clean_dataset.csv` are in a single folder.  
2. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
