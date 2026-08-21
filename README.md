# Mercedes-Benz Greener Manufacturing

## Machine Learning Prediction System

A machine learning project developed to predict the target variable in the Mercedes-Benz Greener Manufacturing dataset.

The project explores the complete machine learning workflow, from data preparation and preprocessing to feature selection, dimensionality reduction, model training, evaluation, tuning, and deployment.

---

## 🚗 Project Overview

The Mercedes-Benz Greener Manufacturing dataset contains a large number of manufacturing-related features. A major challenge is the high-dimensional feature space, which makes feature selection and dimensionality reduction important.

Our project investigates different approaches to reduce the feature space and compares several machine learning models to identify a strong predictive solution.

The final model is deployed as an interactive **Streamlit web application**.

---

## 📊 Project Pipeline

The project consists of four main phases:

### 1. Data Preparation & EDA

- Loaded and inspected the dataset
- Examined the dataset structure and feature types
- Identified numerical and categorical features
- Checked for missing values
- Investigated constant features
- Split the data into training and validation sets

The dataset was divided into:

- **Training:** 3,367 samples
- **Validation:** 842 samples

---

### 2. Data Preprocessing

#### Constant Feature Removal

Constant features were removed because they contain no variation and therefore cannot provide useful predictive information.

After removing constant features:

- **364 features remained**

#### One-Hot Encoding

Categorical variables were transformed using `OneHotEncoder`.

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
```

    handle_unknown="ignore",
 This increased the feature space to:

549 processed features
Feature Scaling

The processed features were standardized using StandardScaler.

The scaler was fitted only on the training data and then applied to the validation data.

Lasso Feature Selection

Lasso regression was used as a feature-selection method.

Features whose Lasso coefficients were zero were removed.

Results:

Original processed features: 549
Selected features: 78
Removed features: 471

This reduced the dimensionality substantially while maintaining strong predictive performance.

4. PCA Bonus Challenge

Principal Component Analysis (PCA) was also investigated as a dimensionality-reduction method.

Different variance-retention levels were tested, including:

80%
85%
90%
95%
99%

At 95% retained variance, PCA reduced the feature space from 549 dimensions to 239 components.

The PCA approach was compared with Lasso feature selection using validation performance.

Dimensionality Reduction Comparison
Method	Dimensions	Validation R²
Baseline	549	0.549930
Lasso	78	0.596536
PCA (95% variance)	239	0.586905

Lasso achieved the strongest validation R² while using substantially fewer features than PCA.

5. Models

Several regression models were evaluated:

Linear Regression
Ridge Regression
Random Forest
Tuned Random Forest
XGBoost

Lasso-selected features were used for the model comparison.

The models were evaluated using:

R²
RMSE
MAE
6. Model Comparison
Model	Features	Validation R²	Validation RMSE	Validation MAE
Linear Regression + Lasso	78	0.596536	7.924587	5.293676
Ridge + Lasso	78	0.596124	7.928641	5.290851
Random Forest + Lasso	78	0.466431	9.113173	5.868031
Tuned Random Forest + Lasso	78	0.593053	7.958720	5.310659
XGBoost + Lasso	78	0.580659	8.079009	5.358029
Final Model

The final selected model is:

Lasso Feature Selection + Linear Regression

Its validation performance was:

R²: 0.596536
RMSE: 7.924587
MAE: 5.293676
7. Deployment

The final model and preprocessing components were saved using joblib.

The saved model artifact contains:

OneHotEncoder
StandardScaler
Lasso feature-selection mask
Selected feature names
Linear Regression model
Numerical feature names
Categorical feature names

This allows the deployed application to reproduce the same preprocessing pipeline used during model development.

The project was deployed using Streamlit.

Live Application

Open the Mercedes-Benz Greener Manufacturing App

The application contains:

Home — project overview and final model performance
Prediction — generate predictions using the trained model
Compare — compare the evaluated models
Dashboard — visualize project and model information
8. Repository Structure
Mercedes-Benz-Greener-Manufacturing/
│
├── deployment/
│   ├── app.py
│   ├── final_model.pkl
│   └── requirements.txt
│
├── README.md
├── .gitignore
└── ...

The complete analysis and modeling process are documented in the project notebook.

9. Technologies

The project was developed using:

Python
Pandas
NumPy
Scikit-learn
XGBoost
Streamlit
Joblib
Jupyter Notebook
10. Project Deliverables

The final project includes:

Data preparation and EDA
Data preprocessing
Lasso feature selection
PCA dimensionality-reduction comparison
Multiple regression models
Hyperparameter tuning
Model evaluation
Streamlit deployment
Public GitHub repository   sparse_output=False
)
