# streamlit_advanced_eda.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")
st.set_page_config(page_title="Notebook-like EDA App", layout="wide")

st.title("Notebook-style Advanced EDA App 🚀")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

def basic_info(df):
    st.markdown("**Shape & Basic Info**")
    st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
    with st.expander("Show dtypes & non-null counts"):
        info_df = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "non-null count": df.notnull().sum(),
            "null count": df.isnull().sum()
        })
        st.dataframe(info_df)

    with st.expander("Head (5 rows)"):
        st.dataframe(df.head())

    with st.expander("Descriptive statistics"):
        st.text(df.describe(include="all").transpose().to_string())

def detect_columns(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    return numeric_cols, cat_cols

def plot_numeric_overview(df, numeric_cols):
    st.subheader("Numeric Columns Overview")
    for col in numeric_cols:
        st.markdown(f"**{col}**")
        c1, c2, c3 = st.columns([1,1,1])
        # Histogram (plotly)
        with c1:
            fig = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}", marginal="rug")
            st.plotly_chart(fig, use_container_width=True)
        # Box + Violin (plotly)
        with c2:
            fig2 = px.box(df, y=col, points="outliers", title=f"Boxplot of {col}")
            st.plotly_chart(fig2, use_container_width=True)
        with c3:
            fig3 = px.violin(df, y=col, box=True, points="all", title=f"Violin of {col}")
            st.plotly_chart(fig3, use_container_width=True)

def plot_categorical_overview(df, cat_cols):
    st.subheader("Categorical Columns Overview")
    for col in cat_cols:
        st.markdown(f"**{col}**")
        vc = df[col].value_counts(dropna=False)
        c1, c2 = st.columns([1,1])
        with c1:
            fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f"Bar chart of {col}", labels={"x":col, "y":"count"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            # Pie (top 10)
            top = vc[:10]
            fig2 = px.pie(values=top.values, names=top.index.astype(str), title=f"Pie chart (top 10) of {col}")
            st.plotly_chart(fig2, use_container_width=True)

def correlation_and_pairwise(df, numeric_cols):
    if len(numeric_cols) < 2:
        st.info("At least 2 numeric columns required for correlation and pairwise visuals.")
        return
    st.subheader("Correlation Matrix")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Interactive Pairwise (Scatter Matrix)")
    fig2 = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter matrix")
    st.plotly_chart(fig2, use_container_width=True)

def pca_visualization(df, numeric_cols):
    st.subheader("PCA (2 components) visualization")
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for PCA.")
        return
    scaled = StandardScaler().fit_transform(df[numeric_cols].dropna())
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(pca_res, columns=["PC1","PC2"])
    maybe_label = None
    # If there's a categorical column, try to use the first one for coloring
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    if len(cat_cols) >= 1:
        maybe_label = df[cat_cols[0]].astype(str).reset_index(drop=True)
        pca_df["label"] = maybe_label
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="label", title="PCA 2D colored by " + cat_cols[0])
    else:
        fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA 2D (no categorical label found)")
    st.plotly_chart(fig, use_container_width=True)

def generate_pdf_report(df, summary_text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(180, 770, "Notebook-style EDA Report")
    c.setFont("Helvetica", 10)
    c.drawString(30, 740, f"Rows: {df.shape[0]}  Columns: {df.shape[1]}")
    text = c.beginText(30, 720)
    text.setFont("Helvetica", 9)
    for line in summary_text.split("\n")[:45]:
        text.textLine(line)
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    st.sidebar.header("Preprocessing & Options")
    # Missing value strategy
    mv_strategy = st.sidebar.selectbox("Missing Value Handling", ["Show as-is", "Drop rows with any NA", "Fill NA (median for numeric, mode for categorical)"])
    if mv_strategy == "Drop rows with any NA":
        df = df.dropna()
    elif mv_strategy == "Fill NA (median for numeric, mode for categorical)":
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "")

    basic_info(df)
    numeric_cols, cat_cols = detect_columns(df)

    st.markdown("---")
    st.sidebar.header("Auto EDA Controls")
    auto_eda = st.sidebar.checkbox("Run Auto EDA (full notebook-like workflow)", value=True)
    if auto_eda:
        if numeric_cols:
            plot_numeric_overview(df, numeric_cols)
        if cat_cols:
            plot_categorical_overview(df, cat_cols)
        correlation_and_pairwise(df, numeric_cols)
        pca_visualization(df, numeric_cols)
    else:
        # Manual selection mode
        st.subheader("Manual Analysis Mode")
        cols = st.multiselect("Select columns to analyze", df.columns.tolist())
        if cols:
            sel_num = df[cols].select_dtypes(include=np.number).columns.tolist()
            sel_cat = df[cols].select_dtypes(exclude=np.number).columns.tolist()
            if sel_num:
                plot_numeric_overview(df[cols], sel_num)
            if sel_cat:
                plot_categorical_overview(df[cols], sel_cat)
            correlation_and_pairwise(df, sel_num)

    # Quick insights (automated)
    st.markdown("---")
    st.subheader("Quick Automated Insights")
    with st.expander("Top quick checks"):
        # Missing
        st.write("Missing values per column:")
        st.write(df.isnull().sum())
        # Duplicates
        dup_count = df.duplicated().sum()
        st.write(f"Duplicate rows: {dup_count}")
        # Skewness for numeric
        if numeric_cols:
            skewness = df[numeric_cols].skew().sort_values(ascending=False)
            st.write("Skewness (numeric columns):")
            st.write(skewness)

    # Download PDF report
    st.markdown("---")
    st.subheader("Download Report")
    summary_stats = df.describe(include="all").to_string()
    pdf_buffer = generate_pdf_report(df, summary_stats)
    st.download_button(
        label="Download EDA Report as PDF",
        data=pdf_buffer,
        file_name="notebook_style_EDA_report.pdf",
        mime="application/pdf"
    )
