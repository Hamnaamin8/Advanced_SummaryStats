import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Data Cleaning Function
# =========================
def clean_adidas_data(df):
    # Convert date
    df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")

    # Clean numeric columns
    for col in ["Price per Unit", "Units Sold", "Total Sales", "Operating Profit"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("[$,]", "", regex=True)
            .astype(float)
        )

    # Clean Operating Margin (%)
    df["Operating Margin"] = (
        df["Operating Margin"].astype(str).str.replace("%", "", regex=True).astype(float)
    )

    # Extra calculated columns
    df["Profit Margin %"] = (df["Operating Profit"] / df["Total Sales"]) * 100
    df["Revenue per Unit"] = df["Total Sales"] / df["Units Sold"]

    return df


# =========================
# Streamlit App
# =========================
st.title("📊 Advanced Adidas Sales Analysis Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload Adidas Sales CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = clean_adidas_data(df)

    # Tabs for navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📋 Overview", "🔢 Numeric Analysis", "🛍 Categorical Analysis", "📈 Time Series", "⚡ Cross Analysis"]
    )

    # =========================
    # Tab 1: Overview
    # =========================
    with tab1:
        st.subheader("Dataset Preview")
        st.write(df.head())

        st.subheader("General Info")
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.write("Missing values per column:")
        st.write(df.isnull().sum())

        st.subheader("Data Types")
        st.write(df.dtypes)

    # =========================
    # Tab 2: Numeric Analysis
    # =========================
    with tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col = st.selectbox("Select numeric column", numeric_cols)

        st.write(df[col].describe())
        st.write(f"Skewness: {df[col].skew():.2f}")
        st.write(f"Kurtosis: {df[col].kurtosis():.2f}")

        # Histogram + KDE
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)

        # Boxplot
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # =========================
    # Tab 3: Categorical Analysis
    # =========================
    with tab3:
        cat_cols = df.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()
        col = st.selectbox("Select categorical column", cat_cols)

        st.write(df[col].value_counts())

        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        df[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

        # Pie chart
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        ax.set_title(f"Pie Chart of {col}")
        st.pyplot(fig)

    # =========================
    # Tab 4: Time Series
    # =========================
    with tab4:
        st.subheader("Sales Trend Over Time")
        df_time = df.groupby("Invoice Date")["Total Sales"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df_time, x="Invoice Date", y="Total Sales", ax=ax)
        ax.set_title("Daily Total Sales Trend")
        st.pyplot(fig)

        st.subheader("Profit Trend Over Time")
        df_profit = df.groupby("Invoice Date")["Operating Profit"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df_profit, x="Invoice Date", y="Operating Profit", ax=ax)
        ax.set_title("Daily Operating Profit Trend")
        st.pyplot(fig)

    # =========================
    # Tab 5: Cross Analysis
    # =========================
    with tab5:
        st.subheader("Grouped Statistics")
        group_col = st.selectbox("Group by column", ["Region", "State", "Product", "Retailer", "Sales Method"])
        metric = st.selectbox("Select metric", ["Total Sales", "Operating Profit", "Units Sold"])

        grouped = df.groupby(group_col)[metric].sum().sort_values(ascending=False)

        st.write(grouped)

        fig, ax = plt.subplots(figsize=(10, 6))
        grouped.plot(kind="bar", ax=ax)
        ax.set_title(f"{metric} by {group_col}")
        st.pyplot(fig)

        st.subheader("Scatterplot Analysis")
        x_col = st.selectbox("X-axis", numeric_cols, index=numeric_cols.index("Units Sold"))
        y_col = st.selectbox("Y-axis", numeric_cols, index=numeric_cols.index("Total Sales"))

        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_col], y=df[y_col], hue=df["Region"], ax=ax)
        ax.set_title(f"{y_col} vs {x_col}")
        st.pyplot(fig)
