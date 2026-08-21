# Mercedes-Benz Greener Manufacturing

## Machine Learning Prediction System

This project develops a machine learning system for predicting the target variable in the Mercedes-Benz Greener Manufacturing dataset.

The project covers the complete machine learning workflow:

- Data Preparation & Exploratory Data Analysis
- Data Preprocessing
- Feature Selection and Dimensionality Reduction
- Model Training and Tuning
- Model Comparison
- Streamlit Deployment

---

## 1. Project Objective

The objective of this project is to build a machine learning model capable of predicting the target variable from manufacturing-related data while investigating effective approaches for handling a high-dimensional feature space.

Because the dataset contains a large number of features, dimensionality reduction and feature selection were important parts of the project.

---

## 2. Data Preparation & EDA

The first phase focused on understanding and preparing the dataset.

The main steps included:

- Inspecting the dataset structure
- Examining numerical and categorical features
- Checking for missing values
- Exploring feature distributions
- Investigating constant features
- Splitting the dataset into training and validation sets

The data was divided into:

- Training set: 3,367 samples
- Validation set: 842 samples

---

## 3. Data Preprocessing

### Constant Feature Removal

Constant features were removed because they contain no variation and therefore provide no useful information for prediction.

After this step, the training and validation datasets contained:

- 364 features

### One-Hot Encoding

Categorical features were transformed using `OneHotEncoder`.

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)
