import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mercedes-Benz Greener Manufacturing",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# LOAD MODEL ARTIFACT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_model.pkl"

artifact = joblib.load(MODEL_PATH)

encoder = artifact["encoder"]
scaler = artifact["scaler"]
lasso_selected_mask = artifact["lasso_selected_mask"]
selected_feature_names = artifact["selected_feature_names"]
linear_model = artifact["linear_model"]

numerical_features = artifact["numerical_features"]
categorical_features = artifact["categorical_features"]


# ============================================================
# MODEL COMPARISON DATA
# ============================================================

model_comparison = pd.DataFrame([
    {
        "Model": "Linear Regression + Lasso",
        "Features": 78,
        "Validation R²": 0.596536,
        "Validation RMSE": 7.924587,
        "Validation MAE": 5.293676
    },
    {
        "Model": "Ridge + Lasso",
        "Features": 78,
        "Validation R²": 0.596124,
        "Validation RMSE": 7.928641,
        "Validation MAE": 5.290851
    },
    {
        "Model": "Random Forest + Lasso",
        "Features": 78,
        "Validation R²": 0.466431,
        "Validation RMSE": 9.113173,
        "Validation MAE": 5.868031
    },
    {
        "Model": "Tuned Random Forest + Lasso",
        "Features": 78,
        "Validation R²": 0.593053,
        "Validation RMSE": 7.958720,
        "Validation MAE": 5.310659
    },
    {
        "Model": "XGBoost + Lasso",
        "Features": 78,
        "Validation R²": 0.580659,
        "Validation RMSE": 8.079009,
        "Validation MAE": 5.358029
    }
])


# ============================================================
# PCA / FEATURE REDUCTION DATA
# ============================================================

reduction_comparison = pd.DataFrame([
    {
        "Method": "Baseline",
        "Dimensions": 549,
        "Validation R²": 0.549930
    },
    {
        "Method": "Lasso",
        "Dimensions": 78,
        "Validation R²": 0.596536
    },
    {
        "Method": "PCA (95% variance)",
        "Dimensions": 239,
        "Validation R²": 0.586905
    }
])


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to:",
    [
        "Home",
        "Prediction",
        "Compare",
        "Dashboard"
    ]
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.title("Mercedes-Benz Greener Manufacturing")

    st.subheader("Machine Learning Prediction System")

    st.write(
        """
        Welcome to our Mercedes-Benz Greener Manufacturing
        machine learning project.

        This application uses machine learning to predict the
        target variable from manufacturing data.
        """
    )

    st.info(
        "Our final model uses Lasso-based feature selection "
        "followed by Linear Regression."
    )

    st.markdown("---")

    st.subheader("Final Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Validation R²",
            "0.5965"
        )

    with col2:
        st.metric(
            "Validation RMSE",
            "7.9246"
        )

    with col3:
        st.metric(
            "Validation MAE",
            "5.2937"
        )

    st.markdown("---")

    st.subheader("Feature Reduction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Original Features",
            "376"
        )

    with col2:
        st.metric(
            "Processed Features",
            "549"
        )

    with col3:
        st.metric(
            "Lasso Selected",
            "78"
        )

    st.write(
        """
        Lasso reduced the processed feature space from 549 features
        to 78 informative features while maintaining strong
        validation performance.
        """)


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "Prediction":

    st.title("Prediction")

    st.write(
        "Enter the manufacturing features below to generate "
        "a prediction using the trained Linear Regression model."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Numerical inputs
    # --------------------------------------------------------

    input_data = {}

    if numerical_features:

        st.subheader("Numerical Features")

        numerical_columns = st.columns(3)

        for i, feature in enumerate(numerical_features):

            with numerical_columns[i % 3]:

                input_data[feature] = st.number_input(
                    feature,
                    value=0.0,
                    step=1.0
                )

    # --------------------------------------------------------
    # Categorical inputs
    # --------------------------------------------------------

    if categorical_features:

        st.subheader("Categorical Features")

        categorical_columns = st.columns(3)

        for i, feature in enumerate(categorical_features):

            categories = list(
                encoder.categories_[
                    categorical_features.index(feature)
                ]
            )

            with categorical_columns[i % 3]:

                input_data[feature] = st.selectbox(
                    feature,
                    categories
                )

    st.markdown("---")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if st.button(
        "Predict",
        type="primary",
        use_container_width=True
    ):

        try:

            input_df = pd.DataFrame(
                [input_data],
                columns=numerical_features + categorical_features
            )

            # One-hot encode categorical variables
            encoded = encoder.transform(
                input_df[categorical_features]
            )

            # Numerical data
            numerical_data = input_df[
                numerical_features
            ].to_numpy()

            # Combine numerical + encoded categorical features
            processed_input = np.hstack(
                [
                    numerical_data,
                    encoded
                ]
            )

            # Scale using training scaler
            scaled_input = scaler.transform(
                processed_input
            )

            # Apply Lasso feature-selection mask
            lasso_input = scaled_input[
                :,
                lasso_selected_mask
            ]

            # Final prediction
            prediction = linear_model.predict(
                lasso_input
            )[0]

            st.success(
                f"Predicted target value: **{prediction:.4f}**"
            )

        except Exception as e:

            st.error(
                "An error occurred while generating the prediction."
            )

            st.exception(e)


# ============================================================
# COMPARE PAGE
# ============================================================

elif page == "Compare":

    st.title("Model Comparison")

    st.write(
        "Comparison of the machine learning models evaluated "
        "during Phase 3."
    )

    st.markdown("---")

    st.subheader("Model Performance")

    st.dataframe(
        model_comparison.style.format({
            "Validation R²": "{:.4f}",
            "Validation RMSE": "{:.4f}",
            "Validation MAE": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("Validation R²")

    r2_chart = model_comparison.set_index(
        "Model"
    )["Validation R²"]

    st.bar_chart(r2_chart)

    st.subheader("Validation RMSE")

    rmse_chart = model_comparison.set_index(
        "Model"
    )["Validation RMSE"]

    st.bar_chart(rmse_chart)

    st.subheader("Validation MAE")

    mae_chart = model_comparison.set_index(
        "Model"
    )["Validation MAE"]

    st.bar_chart(mae_chart)

    st.markdown("---")

    st.subheader("Dimensionality Reduction")

    st.dataframe(
        reduction_comparison,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        reduction_comparison.set_index(
            "Method"
        )["Dimensions"]
    )

    st.info(
        """
        Lasso achieved the strongest validation performance among
        the dimensionality-reduction approaches tested while
        reducing the feature space from 549 to 78 features.
        """
    )


# ============================================================
# DASHBOARD PAGE
# ============================================================

elif page == "Dashboard":

    st.title("Project Dashboard")

    st.write(
        "Overview of the dataset, preprocessing, dimensionality "
        "reduction, and final model performance."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Dataset metrics
    # --------------------------------------------------------

    st.subheader("Dataset & Preprocessing")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Training Samples",
            "3,367"
        )

    with col2:
        st.metric(
            "Validation Samples",
            "842"
        )

    with col3:
        st.metric(
            "Processed Features",
            "549"
        )

    with col4:
        st.metric(
            "Lasso Features",
            "78"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Final model metrics
    # --------------------------------------------------------

    st.subheader("Final Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Validation R²",
            "0.5965"
        )

    with col2:
        st.metric(
            "Validation RMSE",
            "7.9246"
        )

    with col3:
        st.metric(
            "Validation MAE",
            "5.2937"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    st.subheader("Model Performance Comparison")

    st.bar_chart(
        model_comparison.set_index(
            "Model"
        )[
            [
                "Validation R²"
            ]
        ]
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Feature reduction
    # --------------------------------------------------------

    st.subheader("Feature Reduction")

    st.bar_chart(
        reduction_comparison.set_index(
            "Method"
        )[
            [
                "Dimensions"
            ]
        ]
    )

    st.markdown("---")

    st.subheader("Selected Lasso Features")

    st.write(
        f"Lasso selected {len(selected_feature_names)} "
        "features from the 549 processed features."
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Selected Feature": selected_feature_names
            }
        ),
        use_container_width=True,
        hide_index=True
    )