# ==============================
# home.py — PilgrimageAI Dashboard
# ==============================

import streamlit as st
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import StringIO
from streamlit_autorefresh import st_autorefresh
from deep_translator import GoogleTranslator
from transformers import pipeline
import pdfplumber
import documentation

# ---------------------------------------
# --- PAGE CONFIGURATION ---
# ---------------------------------------
st.set_page_config(page_title="PILGRIMAGE DEMOGRAPHICS DASHBOARD", layout="wide")


# ---------------------------------------
# --- HELPER FUNCTIONS ---
# ---------------------------------------
def get_base64(fp):
    """Encodes image for Streamlit background use."""
    with open(fp, "rb") as f:
        return base64.b64encode(f.read()).decode()


def add_bg_from_local(image_file):
    """Adds a static background image for the analyze page."""
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------
# --- HOME PAGE ---
# ---------------------------------------
def home():
    img_b64 = get_base64("pilgrimage.png")

    st.markdown(
        f"""
        <style>
          .stApp {{
            background-image: url("data:image/png;base64,{img_b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
          }}
          .overlay {{
            background-color: rgba(255,255,255,0.85);
            padding: 2rem;
            border-radius: 1rem;
            max-width: 650px;
            margin: 8vh auto;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            font-family: Arial, sans-serif;
            text-align: center;
          }}
          .overlay h1 {{ color: #DAA520; text-decoration: underline; }}
          .overlay h2 {{ color: #DAA520; font-weight: bold; font-style: italic; }}
          .overlay p {{ color: #333; margin: 0.5rem 0; text-align: justify; }}
        </style>

        <div class="overlay">
          <h1>PILGRIMAGEAI</h1>
          <h2>Voice of the Pilgrims</h2>
          <p>PILGRIMAGEAI is an AI-powered platform that automatically analyzes and categorizes large-scale pilgrim feedback data.</p>
          <ul>
            <li>Automatically categorizes feedback across key service areas</li>
            <li>Performs sentiment analysis to assess satisfaction levels</li>
            <li>Provides authorities with data-driven insights to enhance pilgrim experience</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("View Dashboard"):
        st.session_state.page = "dashboard"
    if st.button("Analyze Comments"):
        st.session_state.page = "analyze"
    if st.button("Documentation: Instructions to use the App"):
        st.session_state.page = "documentation"


# ---------------------------------------
# --- DASHBOARD PAGE ---
# ---------------------------------------
def dashboard():
    st.title("📊 Real-Time Demographic Dashboard")
    st_autorefresh(interval=10 * 1000, limit=None, key="datarefresh")

    # --- Data Input ---
    data_source = st.radio(
        "Select Data Source", ["Upload CSV", "Enter API URL", "Paste Raw CSV Text"]
    )
    dataset = None

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload your data file", type=["csv", "xlsx", "xls", "ods"]
        )
        if uploaded_file:
            ext = uploaded_file.name.split(".")[-1].lower()
            if ext in ["csv", "txt"]:
                dataset = pd.read_csv(uploaded_file, encoding="utf-8", errors="replace")
            elif ext in ["xls", "xlsx", "ods"]:
                dataset = pd.read_excel(uploaded_file)

    elif data_source == "Enter API URL":
        api_url = st.text_input("Enter API URL returning CSV data")
        if api_url:
            try:
                r = requests.get(api_url)
                r.raise_for_status()
                dataset = pd.read_csv(StringIO(r.text))
            except Exception as e:
                st.error(f"Failed to load API data: {e}")

    elif data_source == "Paste Raw CSV Text":
        raw_csv = st.text_area("Paste your CSV data here")
        if raw_csv:
            dataset = pd.read_csv(StringIO(raw_csv))

    if dataset is None:
        st.info("Please upload or enter data to continue.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
        return

    # --- Validate columns ---
    dataset.columns = dataset.columns.str.strip()
    required_cols = ["العمر Age", "الجنسية Nationality", "الجنس Gender"]
    if not all(col in dataset.columns for col in required_cols):
        st.error("❌ Required columns not found in dataset.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
        return

    # --- Gender mapping ---
    gender_map = {"أنثى": "Female", "ذكر": "Male"}
    dataset["Gender_English"] = dataset["الجنس Gender"].map(gender_map)

    # --- Filters ---
    st.markdown("### 🔍 Filter Data")
    col1, col2 = st.columns(2)
    with col1:
        genders = dataset["الجنس Gender"].dropna().unique()
        selected_genders = st.multiselect(
            "Select Gender", options=genders, default=list(genders)
        )
    with col2:
        nationalities = dataset["الجنسية Nationality"].dropna().unique()
        selected_nationalities = st.multiselect(
            "Select Nationality", options=nationalities, default=list(nationalities)
        )

    # Apply filters
    fdf = dataset[
        dataset["الجنس Gender"].isin(selected_genders)
        & dataset["الجنسية Nationality"].isin(selected_nationalities)
    ]

    if fdf.empty:
        st.warning("No data available after filtering.")
        return

    # --- KPI SUMMARY CARDS ---
    total_respondents = len(fdf)
    distinct_nationalities = fdf["الجنسية Nationality"].nunique()
    gender_ratio = (
        f"{(fdf['Gender_English'].value_counts(normalize=True) * 100).round(1).to_dict()}"
    )

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("👥 Total Respondents", f"{total_respondents:,}")
    kpi2.metric("🌍 Distinct Nationalities", distinct_nationalities)
    kpi3.metric("⚖️ Gender Ratio (%)", gender_ratio)

    st.markdown("---")

    # ---------------- AGE DISTRIBUTION ----------------
    df_age = (
        fdf["العمر Age"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Age", "العمر Age": "Count"})
        .sort_values("Age")
    )
    df_rep = df_age["Age"].repeat(df_age["Count"]).astype(float)

    stats = {
        "mean": df_rep.mean(),
        "median": df_rep.median(),
        "mode": df_rep.mode().iloc[0],
        "min": df_rep.min(),
        "max": df_rep.max(),
        "skewness": df_rep.skew(),
        "kurtosis": df_rep.kurt(),
    }

    st.subheader("📈 Age Distribution with Statistical Markers")
    fig_age = px.histogram(
        fdf,
        x="العمر Age",
        nbins=20,
        title="Age Distribution",
        color_discrete_sequence=["#3498db"],
    )
    fig_age.add_vline(x=stats["mean"], line_dash="dot", line_color="red", annotation_text="Mean")
    fig_age.add_vline(x=stats["median"], line_dash="dash", line_color="orange", annotation_text="Median")
    fig_age.add_vline(x=stats["mode"], line_dash="dashdot", line_color="purple", annotation_text="Mode")
    st.plotly_chart(fig_age, use_container_width=True)
    st.info(
        f"""
        **Interpretation:**  
        - Mean ({stats['mean']:.2f}) shows the average pilgrim age.  
        - Median ({stats['median']:.2f}) splits ages evenly.  
        - Mode ({stats['mode']:.2f}) is the most frequent age.  
        - Skewness ({stats['skewness']:.2f}) shows asymmetry — positive → younger-dominated.  
        - Kurtosis ({stats['kurtosis']:.2f}) indicates whether ages are tightly grouped or widely spread.
        """
    )

    st.markdown("---")

    # ---------------- FREQUENCY DISTRIBUTIONS ----------------
    st.subheader("🧮 Frequency Distribution Analysis")

    # NATIONALITY
    freq_nat = fdf["الجنسية Nationality"].value_counts().reset_index()
    freq_nat.columns = ["Nationality", "Frequency"]
    freq_nat["Percentage"] = (
        freq_nat["Frequency"] / freq_nat["Frequency"].sum() * 100
    ).round(2)
    st.markdown("#### 🌍 Nationality Frequency Table")
    st.dataframe(freq_nat)
    csv_nat = freq_nat.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Nationality Data", csv_nat, "nationality_distribution.csv")
    st.info("Higher frequency indicates greater representation from that nationality group.")

    # GENDER
    freq_gender = fdf["Gender_English"].value_counts().reset_index()
    freq_gender.columns = ["Gender", "Frequency"]
    freq_gender["Percentage"] = (
        freq_gender["Frequency"] / freq_gender["Frequency"].sum() * 100
    ).round(2)
    st.markdown("#### 🚻 Gender Frequency Table")
    st.dataframe(freq_gender)
    csv_gender = freq_gender.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Gender Data", csv_gender, "gender_distribution.csv")
    st.info("Shows participation ratio by gender — a balanced ratio implies equal involvement.")

    # AGE
    freq_age = fdf["العمر Age"].value_counts().reset_index()
    freq_age.columns = ["Age", "Frequency"]
    freq_age["Percentage"] = (
        freq_age["Frequency"] / freq_age["Frequency"].sum() * 100
    ).round(2)
    st.markdown("#### ⏳ Age Frequency Table")
    st.dataframe(freq_age)
    csv_age = freq_age.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Age Data", csv_age, "age_distribution.csv")
    st.info("Observe which age groups dominate to identify key pilgrim demographics.")

    st.markdown("---")

    # ---------------- VISUAL REPRESENTATIONS ----------------
    st.subheader("📊 Graphical Representation of Distributions")

    # Nationality
    fig_nat = px.bar(
        freq_nat,
        x="Nationality",
        y="Frequency",
        title="Nationality: Frequency Distribution",
        color="Frequency",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_nat, use_container_width=True)
    st.info("Taller bars indicate more pilgrims from that nationality.")

    # Gender
    fig_g = px.pie(
        freq_gender,
        names="Gender",
        values="Frequency",
        title="Gender: Proportional Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    st.plotly_chart(fig_g, use_container_width=True)
    st.info("Pie chart shows gender proportionality — near-equal slices suggest balance.")

    # Age
    fig_a = px.histogram(
        fdf, x="العمر Age", nbins=15, title="Age: Frequency Distribution"
    )
    st.plotly_chart(fig_a, use_container_width=True)
    st.info(
        "Histogram shape reveals whether age distribution is uniform, skewed, or concentrated around certain ranges."
    )

    st.markdown("---")

    # ---------------- GENDER-LANGUAGE CHART (OPTIONAL) ----------------
    if "اللغة Language" in fdf.columns:
        st.subheader("🌍 Gender–Language Interaction Analysis")

        language_translation = {
            "Bahasa Indonesia": "Indonesian",
            "Français": "French",
            "Türkçe": "Turkish",
            "বাংলা (Bengali)": "Bengali",
            "اردو": "Urdu",
            "English": "English",
            "فارسی": "Persian (Farsi)",
            "العربية": "Arabic",
        }

        gender_translation = {"أنثى": "Female", "ذكر": "Male"}
        fdf["Gender_English"] = fdf["الجنس Gender"].map(gender_translation)

        # Cross-tab between gender and language
        ct = pd.crosstab(fdf["Gender_English"], fdf["اللغة Language"])

        bars = []
        for lang in ct.columns:
            label = language_translation.get(lang, lang)
            bars.append(
                go.Bar(
                    name=label,
                    x=ct.index,
                    y=ct[lang],
                    hovertext=[
                        f"Gender: {g}<br>Language: {label}<br>Count: {v}"
                        for g, v in zip(ct.index, ct[lang])
                    ],
                    hoverinfo="text",
                )
            )

        fig_lang = go.Figure(data=bars)
        fig_lang.update_layout(
            barmode="stack",
            title="Distribution of Gender and Language",
            xaxis_title="Gender",
            yaxis_title="Count",
            legend_title="Language",
            template="plotly_white",
        )
        st.plotly_chart(fig_lang, use_container_width=True)
        st.info(
            "This stacked bar shows how gender groups use different languages — useful for communication and translation service planning."
        )

    # ---------------- Back Button ----------------
    st.markdown("---")
    if st.button("Back to Home"):
        st.session_state.page = "home"


# ---------------------------------------
# --- ANALYZE COMMENTS PAGE ---
# ---------------------------------------
def analyze():
    add_bg_from_local("background.png")
    st.title("💬 Sentiment Classification with Primary Model")
    if st.button("Back to Home"):
        st.session_state.page = "home"
        return


# ---------------------------------------
# --- MAIN ROUTING ---
# ---------------------------------------
def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if st.session_state.page == "home":
        home()
    elif st.session_state.page == "dashboard":
        dashboard()
    elif st.session_state.page == "analyze":
        analyze()
    elif st.session_state.page == "documentation":
        documentation.show()


if __name__ == "__main__":
    main()
