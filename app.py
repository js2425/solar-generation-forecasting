
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Solar Energy Forecasting",
    page_icon="☀️",
    layout="wide"
)

# -----------------------------------
# 2. Dashboard Heading
# -----------------------------------

st.title("☀️ Solar Generation & Household Load Forecasting")

st.markdown("""
An interactive machine learning dashboard for analysing household
electricity consumption, estimating solar generation, and forecasting
net grid load with prediction uncertainty.
""")

st.divider()

# -----------------------------------
# 3. Load Project Files
# -----------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(
        "solar_forecasting_data.csv",
        index_col=0,
        parse_dates=True
    )

@st.cache_resource
def load_model():
    return joblib.load("solar_forecasting_model.joblib")

try:
    df = load_data()
    model_package = load_model()
    st.success("Dataset and trained model loaded successfully!")

except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()

# -----------------------------------
# 4. Sidebar
# -----------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Overview",
        "Energy Analysis",
        "Load Forecasting",
        "Model Information"
    ]
)

# -----------------------------------
# 5. Overview
# -----------------------------------

if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Average Household Consumption",
        f"{df['Global_active_power'].mean():.2f} kW"
    )

    col3.metric(
        "Average Solar Generation",
        f"{df['Solar_Generation_kWh'].mean():.2f} kWh"
    )

    st.subheader("Project Dataset")

    st.dataframe(df.head(10), use_container_width=True)

# -----------------------------------
# 6. Energy Analysis
# -----------------------------------

elif page == "Energy Analysis":

    st.header("Household Energy Analysis")

    st.subheader("Household Electricity Consumption")

    st.line_chart(df["Global_active_power"])

    st.subheader("Estimated Solar Generation")

    st.line_chart(df["Solar_Generation_kWh"])

    st.subheader("Net Grid Load")

    st.line_chart(df["Net_Grid_Load"])

# -----------------------------------
# 7. Load Forecasting
# -----------------------------------

elif page == "Load Forecasting":

    st.header("One-Hour-Ahead Load Forecasting")

    st.write(
        "This section uses the saved trained model "
        "to estimate the next hour's net grid load."
    )

    st.info(
        "We will add the interactive forecasting inputs "
        "after verifying that the dashboard loads correctly."
    )

# -----------------------------------
# 8. Model Information
# -----------------------------------

elif page == "Model Information":

    st.header("Machine Learning Model")

    st.write(
        "Best overall model: ",
        model_package["best_model_name"]
    )

    st.write(
        "The trained model and preprocessing scaler "
        "are loaded from the saved model package."
    )

    st.write(
        "Prediction interval width:",
        round(model_package["interval_width"], 4)
    )

    st.write(
        "Features used:",
        model_package["feature_columns"]
    )

st.divider()

st.caption(
    "Solar Generation & Household Load Forecasting | "
    "Machine Learning Project"
)
