import streamlit as st
# --- PAGE CONFIG ---
st.set_page_config(page_title="PILGRIMAGE DEMOGRAPHICS DASHBOARD", layout="wide")

import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import StringIO
from streamlit_autorefresh import st_autorefresh
from deep_translator import GoogleTranslator
from transformers import pipeline
import pdfplumber
import documentation


# --- HELPER: Background Setup ---
def get_base64(fp):
    with open(fp, "rb") as f:
        return base64.b64encode(f.read()).decode()


# --- HOME PAGE ---
def home():
    img_b64 = get_base64("pilgrimage.png")
    st.markdown(f"""
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
    """, unsafe_allow_html=True)

    if st.button("View Dashboard"):
        st.session_state.page = "dashboard"
    if st.button("Analyze Comments"):
        st.session_state.page = "analyze"
    if st.button("Documentation:Instructions to use the App"):
        st.session_state.page = "documentation"


# --- DASHBOARD PAGE ---
def dashboard():
    st.title("Real-Time Demographic Dashboard")
    st_autorefresh(interval=10 * 1000, limit=None, key="datarefresh")

    # --- Data input ---
    data_source = st.radio("Select Data Source", ['Upload CSV', 'Enter API URL', 'Paste Raw CSV Text'])
    dataset = None

    if data_source == 'Upload CSV':
        uploaded_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx', 'xls', 'ods'])
        if uploaded_file:
            ext = uploaded_file.name.split('.')[-1].lower()
            if ext in ['csv', 'txt']:
                dataset = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
            elif ext in ['xls', 'xlsx', 'ods']:
                dataset = pd.read_excel(uploaded_file)

    elif data_source == 'Enter API URL':
        api_url = st.text_input("Enter API URL returning CSV data")
        if api_url:
            try:
                r = requests.get(api_url)
                r.raise_for_status()
                dataset = pd.read_csv(StringIO(r.text))
            except Exception as e:
                st.error(f"Failed to load API data: {e}")

    elif data_source == 'Paste Raw CSV Text':
        raw_csv = st.text_area("Paste your CSV data here")
        if raw_csv:
            dataset = pd.read_csv(StringIO(raw_csv))

    if dataset is None:
        st.info("Please upload or enter data to continue.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
        return

    # --- Column checks ---
    dataset.columns = dataset.columns.str.strip()
    required_cols = ['العمر Age', 'الجنسية Nationality', 'الجنس Gender']
    if not all(col in dataset.columns for col in required_cols):
        st.error("❌ Required columns not found in dataset.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
        return

    # --- Map Gender labels ---
    gender_map = {'أنثى': 'Female', 'ذكر': 'Male'}
    dataset['Gender_English'] = dataset['الجنس Gender'].map(gender_map)

    # --- FILTERS ---
    st.markdown("### 🔍 Data Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        genders = dataset['الجنس Gender'].dropna().unique()
        selected_genders = st.multiselect("Select Gender", options=genders, default=list(genders))
    with col2:
        nationalities = dataset['الجنسية Nationality'].dropna().unique()
        selected_nationalities = st.multiselect("Select Nationality", options=nationalities, default=list(nationalities))
    with col3:
        st.write("Optional date filter not implemented for this dataset.")

    filtered_df = dataset[
        dataset['الجنس Gender'].isin(selected_genders) &
        dataset['الجنسية Nationality'].isin(selected_nationalities)
    ]

    if filtered_df.empty:
        st.warning("No data after applying filters.")
        return

    # --- Age distribution stats (from your base code) ---
    df_Age = filtered_df["العمر Age"].value_counts().reset_index()
    df_Age.columns = ['العمر Age', 'count']
    df_Age = df_Age.sort_values('العمر Age')
    df_repeated = df_Age['العمر Age'].repeat(df_Age['count']).astype(float)

    stats = {
        "max": df_repeated.max(),
        "min": df_repeated.min(),
        "mean": df_repeated.mean(),
        "median": df_repeated.median(),
        "mode": df_repeated.mode().iloc[0],
        "std": df_repeated.std(),
        "q1": df_repeated.quantile(0.25),
        "q3": df_repeated.quantile(0.75),
        "skewness": df_repeated.skew(),
        "kurtosis": df_repeated.kurt()
    }

    st.markdown("### 📈 Age Distribution with Statistical Markers")
    fig_age_dist = px.histogram(filtered_df, x='العمر Age', nbins=20, title="Age Distribution (Filtered Data)",
                                color_discrete_sequence=['#3498db'])
    fig_age_dist.add_vline(x=stats["mean"], line_dash="dot", line_color="red", annotation_text="Mean")
    fig_age_dist.add_vline(x=stats["median"], line_dash="dash", line_color="orange", annotation_text="Median")
    fig_age_dist.add_vline(x=stats["mode"], line_dash="dashdot", line_color="purple", annotation_text="Mode")
    st.plotly_chart(fig_age_dist, use_container_width=True)

    st.info(f"""
    **Interpretation:**  
    - **Mean ({stats['mean']:.2f})** shows the average age.  
    - **Median ({stats['median']:.2f})** divides the population into two halves.  
    - **Mode ({stats['mode']:.2f})** is the most common age.  
    - **Skewness ({stats['skewness']:.2f})** indicates asymmetry: right-skew means more younger pilgrims.  
    - **Kurtosis ({stats['kurtosis']:.2f})** shows whether ages are tightly or widely spread.  
    """)

    st.markdown("---")

    # ---- Frequency Distribution Analysis ----
    st.subheader("🧮 Frequency Distribution Analysis (Filtered Data)")

    # 1. Nationality
    freq_nat = filtered_df['الجنسية Nationality'].value_counts().reset_index()
    freq_nat.columns = ["Nationality", "Frequency"]
    freq_nat["Percentage"] = (freq_nat["Frequency"] / freq_nat["Frequency"].sum() * 100).round(2)
    st.markdown("#### Nationality Frequency Table")
    st.dataframe(freq_nat)
    st.info("Higher frequency indicates a larger representation of that nationality in the dataset.")

    # 2. Gender
    freq_gender = filtered_df['Gender_English'].value_counts().reset_index()
    freq_gender.columns = ["Gender", "Frequency"]
    freq_gender["Percentage"] = (freq_gender["Frequency"] / freq_gender["Frequency"].sum() * 100).round(2)
    st.markdown("#### Gender Frequency Table")
    st.dataframe(freq_gender)
    st.info("Gender distribution shows the proportion of male and female pilgrims. A balanced ratio suggests equal participation.")

    # 3. Age
    freq_age = filtered_df['العمر Age'].value_counts().reset_index()
    freq_age.columns = ["Age", "Frequency"]
    freq_age["Percentage"] = (freq_age["Frequency"] / freq_age["Frequency"].sum() * 100).round(2)
    st.markdown("#### Age Frequency Table")
    st.dataframe(freq_age)
    st.info("Observe which age groups dominate the distribution to identify the primary demographic age range.")

    st.markdown("---")

    # ---- Graphical Representations ----
    st.subheader("📊 Graphical Representation of Distributions")

    # Nationality - Bar Chart
    fig_nat = px.bar(freq_nat, x="Nationality", y="Frequency", title="Nationality: Frequency Distribution",
                     color="Frequency", color_continuous_scale="Viridis")
    st.plotly_chart(fig_nat, use_container_width=True)
    st.info("Taller bars indicate higher representation by nationality. Observe dominant or underrepresented national groups.")

    # Gender - Pie Chart
    fig_gender = px.pie(freq_gender, names="Gender", values="Frequency", title="Gender: Proportional Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_gender, use_container_width=True)
    st.info("Pie chart reveals gender proportionality — near equal slices suggest gender balance, while skewed distribution highlights dominance.")

    # Age - Histogram
    fig_age = px.histogram(filtered_df, x="العمر Age", nbins=15, title="Age: Frequency Distribution",
                           color_discrete_sequence=["#1f77b4"])
    st.plotly_chart(fig_age, use_container_width=True)
    st.info("""
    **How to Interpret:**  
    - The **shape** of the histogram reveals whether the distribution is uniform, skewed, or normal.  
    - **Wide spread** indicates age diversity.  
    - **Right skew** → younger dominant; **left skew** → older dominant.  
    - Variance & Standard Deviation quantify how spread out ages are.  
    """)

    st.markdown("---")

    # Back button
    if st.button("Back to Home"):
        st.session_state.page = "home"


# --- ANALYZE COMMENTS PAGE ---
def analyze():
    add_bg_from_local("background.png")
    st.title("Sentiment Classification with Primary Model")
    if st.button("Back to Home"):
        st.session_state.page = "home"
        return


# --- BACKGROUND CSS FOR ANALYZE PAGE ---
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)


# --- MAIN ROUTING ---
def main():
    if 'page' not in st.session_state:
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
