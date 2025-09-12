import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Title
st.title("Advanced Data Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Show basic info
    st.subheader("Summary Statistics")
    st.write(df.describe(include="all"))

    # Column selection
    st.subheader("Column-wise Analysis")
    column = st.selectbox("Select a column for analysis", df.columns)

    # If numeric column
    if pd.api.types.is_numeric_dtype(df[column]):
        st.write(f"Summary of {column}:")
        st.write(df[column].describe())

        # Histogram
        fig, ax = plt.subplots()
        df[column].hist(ax=ax, bins=20)
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)

        # Line Chart
        fig, ax = plt.subplots()
        df[column].reset_index(drop=True).plot(kind="line", ax=ax)
        ax.set_title(f"Line Chart of {column}")
        st.pyplot(fig)

        # Scatterplot (vs another numeric column)
        other_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(other_num_cols) > 1:
            y_col = st.selectbox(
                "Select another numeric column for scatterplot",
                [c for c in other_num_cols if c != column],
            )
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[column], y=df[y_col], ax=ax)
            ax.set_title(f"Scatterplot of {column} vs {y_col}")
            st.pyplot(fig)

        # Boxplot
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], ax=ax)
        ax.set_title(f"Boxplot of {column}")
        st.pyplot(fig)

    # If categorical column
    else:
        st.write(f"Value counts of {column}:")
        st.write(df[column].value_counts())

        # Bar Chart
        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Bar Chart of {column}")
        st.pyplot(fig)

        # Pie Chart
        fig, ax = plt.subplots()
        df[column].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_title(f"Pie Chart of {column}")
        ax.set_ylabel("")
        st.pyplot(fig)

    # Heatmap (for correlations if multiple numeric cols exist)
    if df.select_dtypes(include=[np.number]).shape[1] > 1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
