import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Title
st.title("Advanced Data Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Function to generate PDF report
def generate_pdf_report(df, summary):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 770, "EDA Report")

    c.setFont("Helvetica", 12)
    c.drawString(30, 740, f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Add Summary Stats
    c.drawString(30, 720, "Summary Statistics:")
    text = c.beginText(30, 700)
    text.setFont("Helvetica", 9)
    for line in summary.split("\n")[:40]:  # limit rows
        text.textLine(line)
    c.drawText(text)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show dataset
    st.subheader("Explore Dataset")
    st.write("Full Dataset:")
    st.write(df)

    # Show summary stats
    st.subheader("Summary Statistics")
    summary_stats = df.describe(include="all").to_string()
    st.text(summary_stats)

    # Auto EDA Mode
    st.subheader("Exploratory Data Analysis Options")
    auto_mode = st.checkbox("Enable Auto EDA Mode (Generate All Suitable Charts Automatically)")

    if auto_mode:
        st.success("Auto EDA Mode Enabled ✅")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        # Numeric Columns Analysis
        if numeric_cols:
            st.subheader("Numeric Columns Analysis")
            for col in numeric_cols:
                st.markdown(f"### {col}")

                # Histogram
                fig, ax = plt.subplots()
                ax.hist(df[col].dropna(), bins=20, color="skyblue", edgecolor="black")
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)

                # Boxplot
                fig, ax = plt.subplots()
                sns.boxplot(y=df[col], ax=ax, color="lightgreen")
                ax.set_title(f"Boxplot of {col}")
                st.pyplot(fig)

                # Line Chart
                fig, ax = plt.subplots()
                df[col].reset_index(drop=True).plot(kind="line", ax=ax, color="purple")
                ax.set_title(f"Line Chart of {col}")
                st.pyplot(fig)

        # Categorical Columns Analysis
        if cat_cols:
            st.subheader("Categorical Columns Analysis")
            for col in cat_cols:
                st.markdown(f"### {col}")

                # Bar Chart
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind="bar", ax=ax, color="orange", edgecolor="black")
                ax.set_title(f"Bar Chart of {col}")
                st.pyplot(fig)

                # Pie Chart
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax, colormap="tab20")
                ax.set_ylabel("")
                ax.set_title(f"Pie Chart of {col}")
                st.pyplot(fig)

        # Correlation Analysis
        if len(numeric_cols) >= 2:
            st.subheader("Relationships Between Numeric Variables")

            # Scatterplot
            fig, ax = plt.subplots()
            ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6, color="blue")
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            ax.set_title(f"Scatterplot of {numeric_cols[0]} vs {numeric_cols[1]}")
            st.pyplot(fig)

            # Heatmap
            fig, ax = plt.subplots()
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

            # Pairplot
            st.subheader("Pairplot of Numeric Columns")
            pairplot_fig = sns.pairplot(df[numeric_cols], diag_kind="hist", corner=True)
            st.pyplot(pairplot_fig)

    else:
        # Manual Mode
        st.subheader("Manual Column-wise Analysis")
        columns = st.multiselect("Select one or more columns for analysis", df.columns)

        if columns:
            for column in columns:
                st.markdown(f"### Analysis of **{column}**")

                if pd.api.types.is_numeric_dtype(df[column]):
                    st.write(df[column].describe())

                    # Histogram
                    fig, ax = plt.subplots()
                    ax.hist(df[column].dropna(), bins=20, color="skyblue", edgecolor="black")
                    ax.set_title(f"Histogram of {column}")
                    st.pyplot(fig)

                    # Boxplot
                    fig, ax = plt.subplots()
                    sns.boxplot(y=df[column], ax=ax, color="lightgreen")
                    ax.set_title(f"Boxplot of {column}")
                    st.pyplot(fig)

                    # Line Chart
                    fig, ax = plt.subplots()
                    df[column].reset_index(drop=True).plot(kind="line", ax=ax, color="purple")
                    ax.set_title(f"Line Chart of {column}")
                    st.pyplot(fig)

                else:
                    st.write(df[column].value_counts())

                    # Bar Chart
                    fig, ax = plt.subplots()
                    df[column].value_counts().plot(kind="bar", ax=ax, color="orange", edgecolor="black")
                    ax.set_title(f"Bar Chart of {column}")
                    st.pyplot(fig)

                    # Pie Chart
                    fig, ax = plt.subplots()
                    df[column].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax, colormap="tab20")
                    ax.set_ylabel("")
                    ax.set_title(f"Pie Chart of {column}")
                    st.pyplot(fig)

            # Extra charts
            numeric_cols = df[columns].select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) >= 2:
                st.subheader("Relationships Between Selected Numeric Variables")

                # Scatterplot
                fig, ax = plt.subplots()
                ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6, color="blue")
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel(numeric_cols[1])
                ax.set_title(f"Scatterplot of {numeric_cols[0]} vs {numeric_cols[1]}")
                st.pyplot(fig)

                # Heatmap
                fig, ax = plt.subplots()
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)

                # Pairplot
                st.subheader("Pairplot of Selected Numeric Columns")
                pairplot_fig = sns.pairplot(df[numeric_cols], diag_kind="hist", corner=True)
                st.pyplot(pairplot_fig)

    # --- Download Report ---
    st.subheader("Download EDA Report")
    pdf_buffer = generate_pdf_report(df, summary_stats)
    st.download_button(
        label="Download EDA Report as PDF",
        data=pdf_buffer,
        file_name="EDA_Report.pdf",
        mime="application/pdf"
    )

