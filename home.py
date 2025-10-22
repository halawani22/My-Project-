import streamlit as st
# --- PAGE CONFIG ---
st.set_page_config(page_title="PILGRIMAGE DEMOGRAPHICS DASHBOARD", layout="wide")

import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns  # kept for future use
import matplotlib.pyplot as plt  # kept for future use
import requests
from io import StringIO
from streamlit_autorefresh import st_autorefresh
from deep_translator import GoogleTranslator
from transformers import pipeline
import pdfplumber
import documentation


# -------------------------
# HELPER: Background Setup
# -------------------------
def get_base64(fp):
    """Return base64 string for an image file path (used for backgrounds)."""
    with open(fp, "rb") as f:
        return base64.b64encode(f.read()).decode()


# -------------------------
# HOME PAGE
# -------------------------
def home():
    """Render the home / landing page with background and navigation buttons."""
    img_b64 = get_base64("pilgrimage.png")

    st.markdown(f"""
    <style>
      .stApp {{
        background-image: url("data:image/png;base64,{img_b64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        margin: 0;
        padding: 0;
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
      .overlay ul {{ color: #000; padding-left: 1rem; text-align: left; }}
      .overlay .stButton > button {{
        margin-top: 1rem;
        background-color: #fff;
        color: #000;
        padding: 0.75rem 1.5rem;
        border: 2px solid #000;
        border-radius: 0.5rem;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
      }}
      .overlay .stButton > button:hover {{
        background-color: #000;
        color: #fff;
        cursor: pointer;
      }}
    </style>

    <div class="overlay">
      <h1>PILGRIMAGEAI</h1>
      <h2>Voice of the Pilgrims</h2>
      <p>PILGRIMAGEAI is an AI-powered platform that automatically analyzes and categorizes large-scale pilgrim feedback data.</p>
      <ul>
        <li>Automatically categorizes feedback across key service areas</li>
        <li>Performs sentiment analysis to assess overall satisfaction levels</li>
        <li>Provides authorities with data-driven insights to enhance service quality and pilgrim experience</li>
      </ul>
      <p>By adopting this NLP-powered approach, Hajj and Umrah authorities can make informed decisions, prioritize improvements, and ensure a more fulfilling pilgrimage.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("View Dashboard"):
        st.session_state.page = "dashboard"

    if st.button("Analyze Comments"):
        st.session_state.page = "analyze"

    if st.button("Documentation:Instructions to use the App"):
        st.session_state.page = "documentation"


# -------------------------
# DASHBOARD PAGE
# -------------------------
def dashboard():
    """Real-Time Demographic Dashboard with filters, frequency analysis, and interactive charts."""
    st.title("Real-Time Demographic Dashboard")

    img_b64 = get_base64("analysis.png")
    st.markdown(f"""
    <style>
      .custom-container {{
        background: url("data:image/png;base64,{img_b64}") no-repeat center;
        background-size: cover;
        padding: 2rem;
        border-radius: 1rem;
        max-width: 900px;
        margin: 2rem auto;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        background-attachment: local;
        color: #000;
        font-family: 'Segoe UI', sans-serif;
      }}
      .custom-container h2 {{
        color: #DAA520;
        text-align: center;
        margin-bottom: 1rem;
      }}
      .custom-container p,
      .custom-container ul {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 1rem;
        border-radius: 0.5rem;
      }}
      .custom-container ul {{ padding-left: 2rem; }}
    </style>

    <div class="custom-container">
      <h2>AI-Powered Demographic Insights</h2>
      <p>This AI-powered dashboard delivers comprehensive insights into the demographics of Hajj and Umrah pilgrims.</p>
      <ul>
        <li><strong>Age Distribution</strong>: Interactive visualizations illustrating the range and concentration of pilgrims’ ages.</li>
        <li><strong>Statistical Overview</strong>: Key metrics including min, max, mean, median, quartiles, and mode of pilgrim ages.</li>
        <li><strong>Nationality & Gender Breakdown</strong>: Detailed analysis of visitor nationalities segmented by gender.</li>
        <li><strong>Cross-Demographic Insights</strong>: Visualizations combining age, gender, nationality, and language where available.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    st_autorefresh(interval=10 * 1000, limit=None, key="datarefresh")

    # ------------- DATA INPUT -------------
    data_source = st.radio("Select Data Source", ['Upload CSV', 'Enter API URL', 'Paste Raw CSV Text'])
    dataset = None

    if data_source == 'Upload CSV':
        uploaded_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx', 'xls', 'ods', 'txt'])
        if uploaded_file:
            ext = uploaded_file.name.split('.')[-1].lower()
            if ext in ['csv', 'txt']:
                dataset = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
            elif ext in ['xls', 'xlsx', 'ods']:
                dataset = pd.read_excel(uploaded_file)

    elif data_source == 'Enter API URL':
        api_url = st.text_input("Enter API URL returning CSV data")
        if api_url:
            r = requests.get(api_url)
            dataset = pd.read_csv(StringIO(r.text))

    elif data_source == 'Paste Raw CSV Text':
        raw_csv = st.text_area("Paste your CSV text here")
        if raw_csv:
            dataset = pd.read_csv(StringIO(raw_csv))

    if dataset is None:
        st.info("Please upload or enter data to continue.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
        return

    dataset.columns = dataset.columns.str.strip()
    required = ['العمر Age', 'الجنسية Nationality', 'الجنس Gender']
    if not all(c in dataset.columns for c in required):
        st.error("❌ Required columns not found.")
        return

    gender_map = {'أنثى': 'Female', 'ذكر': 'Male'}
    dataset['Gender_English'] = dataset['الجنس Gender'].map(gender_map)

    # ------------- FILTERS -------------
    st.markdown("### Filter Data")
    c1, c2, c3 = st.columns(3)
    with c1:
        g_opts = dataset['الجنس Gender'].dropna().unique()
        sel_g = st.multiselect("Gender", g_opts, default=list(g_opts))
    with c2:
        n_opts = dataset['الجنسية Nationality'].dropna().unique()
        sel_n = st.multiselect("Nationality", n_opts, default=list(n_opts))
    with c3:
        date_cols = [c for c in dataset.columns if 'date' in c.lower()]
        date_col = date_cols[0] if date_cols else None
        if date_col:
            dataset[date_col] = pd.to_datetime(dataset[date_col], errors='coerce')
            min_d, max_d = dataset[date_col].min(), dataset[date_col].max()
            date_rng = st.date_input("Date Range", [min_d, max_d])
        else:
            st.write("No date field")

    fdf = dataset[
        dataset['الجنس Gender'].isin(sel_g) &
        dataset['الجنسية Nationality'].isin(sel_n)
    ]
    if date_col:
        start, end = pd.to_datetime(date_rng[0]), pd.to_datetime(date_rng[1])
        fdf = fdf[(fdf[date_col] >= start) & (fdf[date_col] <= end)]

    if fdf.empty:
        st.warning("No data after filters.")
        return

    # ---------------- KPI SUMMARY CARDS ----------------
    total_resp = len(fdf)
    distinct_nat = fdf['الجنسية Nationality'].nunique()
    gender_counts = fdf['Gender_English'].value_counts()
    ratio = f"{gender_counts.get('Male',0)} : {gender_counts.get('Female',0)} (M:F)"

    st.markdown("### 📊 Key Metrics Summary")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Respondents", total_resp)
    k2.metric("Distinct Nationalities", distinct_nat)
    k3.metric("Gender Ratio (M:F)", ratio)
    st.markdown("---")

    # ---------------- AGE DISTRIBUTION ----------------
    df_age = fdf["العمر Age"].value_counts().reset_index().rename(columns={'index':'العمر Age','العمر Age':'count'}).sort_values('العمر Age')
    df_rep = df_age['العمر Age'].repeat(df_age['count']).astype(float)
    stats = {
        "mean": df_rep.mean(), "median": df_rep.median(), "mode": df_rep.mode().iloc[0],
        "min": df_rep.min(), "max": df_rep.max()
    }

    st.markdown("### 📈 Age Distribution with Statistical Markers")
    fig_age = px.histogram(fdf, x='العمر Age', nbins=20, title="Age Distribution")
    fig_age.add_vline(x=stats["mean"], line_dash="dot", line_color="red", annotation_text=f"Mean: {stats['mean']:.2f}")
    fig_age.add_vline(x=stats["median"], line_dash="dash", line_color="orange", annotation_text=f"Median: {stats['median']:.2f}")
    fig_age.add_vline(x=stats["mode"], line_dash="dashdot", line_color="purple", annotation_text=f"Mode: {stats['mode']:.2f}")
    st.plotly_chart(fig_age, use_container_width=True)

    # ---------------- FREQUENCY TABLES WITH DOWNLOADS ----------------
    st.subheader("🧮 Frequency Distribution Analysis (Filtered Data)")

    # --- Nationality ---
    freq_nat = fdf["الجنسية Nationality"].value_counts().reset_index()
    freq_nat.columns = ["Nationality", "Frequency"]
    freq_nat["Percentage"] = (freq_nat["Frequency"] / freq_nat["Frequency"].sum() * 100).round(2)
    st.markdown("#### Nationality Frequency Table")
    st.dataframe(freq_nat)
    st.download_button("⬇️ Download Nationality CSV", freq_nat.to_csv(index=False).encode('utf-8'),
                       "nationality_freq.csv", "text/csv")

    # --- Gender ---
    freq_g = fdf["Gender_English"].value_counts().reset_index()
    freq_g.columns = ["Gender", "Frequency"]
    freq_g["Percentage"] = (freq_g["Frequency"] / freq_g["Frequency"].sum() * 100).round(2)
    st.markdown("#### Gender Frequency Table")
    st.dataframe(freq_g)
    st.download_button("⬇️ Download Gender CSV", freq_g.to_csv(index=False).encode('utf-8'),
                       "gender_freq.csv", "text/csv")

    # --- Age ---
    freq_a = fdf['العمر Age'].value_counts().reset_index()
    freq_a.columns = ["Age", "Frequency"]
    freq_a["Percentage"] = (freq_a["Frequency"] / freq_a["Frequency"].sum() * 100).round(2)
    st.markdown("#### Age Frequency Table")
    st.dataframe(freq_a)
    st.download_button("⬇️ Download Age CSV", freq_a.to_csv(index=False).encode('utf-8'),
                       "age_freq.csv", "text/csv")

    st.markdown("---")

    if st.button("Back to Home"):
        st.session_state.page = "home"


# -------------------------
# ANALYZE COMMENTS PAGE
# -------------------------
def analyze():
    """Analyze comments: translation, department classification, sentiment."""
    add_bg_from_local("background.png")
    st.title("Sentiment Classification with Primary Model")

    if st.button("Back to Home"):
        st.session_state.page = "home"
        return

    themes_topics = {"Customer Service": ["service","support"],"Product Quality":["quality"],"Delivery":["delivery"],"Billing":["bill"]}
    primary_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    cache = {}

    def translator_dual(t): return t, GoogleTranslator(source="auto", target="en").translate(t)
    def classify_department(c): 
        for k,v in themes_topics.items():
            if any(x in c.lower() for x in v): return k
        return "General"
    def analyze_primary(c): r = primary_pipeline(c)[0]; return r["label"], round(r["score"],2)

    upl = st.file_uploader("📄 Upload CSV, Excel, or TXT", type=["csv","xlsx","txt"])
    man = st.text_area("Enter comments manually (one per line):", height=200)

    if upl:
        df = pd.read_csv(upl) if upl.name.endswith(".csv") else pd.read_excel(upl)
        df["Translated"] = df["Comments"].apply(lambda x: translator_dual(str(x))[1])
        df["Department"] = df["Translated"].apply(classify_department)
        df[["Sentiment","Confidence"]] = df["Translated"].apply(lambda x: pd.Series(analyze_primary(x)))
        st.dataframe(df)
        st.download_button("⬇️ Download Results", df.to_csv(index=False).encode('utf-8'), "sentiment_results.csv", "text/csv")

    elif man.strip():
        lines = [l.strip() for l in man.split("\n") if l.strip()]
        df = pd.DataFrame({"Comments":lines})
        df["Translated"] = df["Comments"].apply(lambda x: translator_dual(str(x))[1])
        df["Department"] = df["Translated"].apply(classify_department)
        df[["Sentiment","Confidence"]] = df["Translated"].apply(lambda x: pd.Series(analyze_primary(x)))
        st.dataframe(df)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode('utf-8'), "manual_results.csv", "text/csv")
    else:
        st.info("📂 Upload a file or enter comments above.")


# -------------------------
# BACKGROUND HELPER
# -------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        enc = base64.b64encode(img.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{enc}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
    """, unsafe_allow_html=True)


# -------------------------
# MAIN ROUTING
# -------------------------
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
