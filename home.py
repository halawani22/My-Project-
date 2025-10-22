# home.py
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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PILGRIMAGE DEMOGRAPHICS DASHBOARD", layout="wide")

# ---------------- BACKGROUND FUNCTION ----------------
def get_base64(fp):
    with open(fp, "rb") as f:
        return base64.b64encode(f.read()).decode()

def add_bg_from_local(image_file):
    """Adds a static background image for the analyze page."""
    try:
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
    except Exception:
        pass

# ---------------- HOME PAGE ----------------
def home():
    """Landing page with background and navigation."""
    try:
        img_b64 = get_base64("pilgrimage.png")
    except Exception:
        img_b64 = ""

    st.markdown(
        f"""
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
        </style>

        <div class="overlay">
          <h1>PILGRIMAGEAI</h1>
          <h2>Voice of the Pilgrims</h2>
          <p>PILGRIMAGEAI is an AI-powered platform that automatically analyzes and categorizes large-scale pilgrim feedback data.</p>
          <ul>
            <li>Automatic categorization across key service areas</li>
            <li>Sentiment analysis for satisfaction insights</li>
            <li>Actionable intelligence for service improvement</li>
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

# ---------------- DASHBOARD ----------------
def dashboard():
    """Real-Time Demographic Dashboard (filters, KPIs, charts, frequency tables)."""
    st.title("Real-Time Demographic Dashboard")
    st_autorefresh(interval=10 * 1000, limit=None, key="datarefresh")

    data_source = st.radio("Select Data Source", ['Upload CSV', 'Enter API URL', 'Paste Raw CSV Text'])
    dataset = None

    if data_source == 'Upload CSV':
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls', 'ods'])
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].lower()
            try:
                if file_type in ['csv', 'txt']:
                    dataset = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
                elif file_type in ['xls', 'xlsx', 'ods']:
                    dataset = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading file: {e}")

    elif data_source == 'Enter API URL':
        api_url = st.text_input("Enter API URL returning CSV data")
        if api_url:
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                dataset = pd.read_csv(StringIO(response.text))
            except Exception as e:
                st.error(f"Failed to fetch data from API: {e}")

    elif data_source == 'Paste Raw CSV Text':
        raw_csv = st.text_area("Paste your CSV data here")
        if raw_csv:
            try:
                dataset = pd.read_csv(StringIO(raw_csv))
            except Exception as e:
                st.error(f"Failed to parse CSV text: {e}")

    if dataset is None:
        st.info("Please upload or enter data to continue.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
        return

    dataset.columns = dataset.columns.str.strip()
    required_cols = ['العمر Age', 'الجنسية Nationality', 'الجنس Gender']
    if not all(col in dataset.columns for col in required_cols):
        st.error("Required columns not found in uploaded data.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
        return

    gender_map = {'أنثى': 'Female', 'ذكر': 'Male'}
    dataset['Gender_English'] = dataset['الجنس Gender'].map(gender_map)

    st.markdown("### Filter Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender_options = dataset['الجنس Gender'].dropna().unique()
        selected_genders = st.multiselect("Filter by Gender", options=gender_options, default=list(gender_options))
    with col2:
        nationality_options = dataset['الجنسية Nationality'].dropna().unique()
        selected_nationalities = st.multiselect("Filter by Nationality", options=nationality_options, default=list(nationality_options))
    with col3:
        st.write("Date filters not available")

    filtered_df = dataset[
        dataset['الجنس Gender'].isin(selected_genders) &
        dataset['الجنسية Nationality'].isin(selected_nationalities)
    ]

    if filtered_df.empty:
        st.warning("No data after applying filters.")
        return

    st.markdown("### Summary Statistics (Filtered)")
    total_respondents = len(filtered_df)
    distinct_nations = filtered_df['الجنسية Nationality'].nunique()
    male_count = (filtered_df['Gender_English'] == 'Male').sum()
    female_count = (filtered_df['Gender_English'] == 'Female').sum()
    gender_ratio = f"{male_count}:{female_count} (M:F)"
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Respondents", f"{total_respondents:,}")
    k2.metric("Distinct Nationalities", distinct_nations)
    k3.metric("Gender Ratio (M:F)", gender_ratio)

    st.markdown("---")

    st.subheader("Age Distribution with Statistical Markers")
    df_age = filtered_df['العمر Age'].value_counts().reset_index()
    df_age.columns = ['Age', 'Count']
    df_age['Age'] = pd.to_numeric(df_age['Age'], errors='coerce')
    df_age = df_age.dropna(subset=['Age']).sort_values('Age')
    df_repeated = df_age['Age'].repeat(df_age['Count']).astype(float)
    stats = {
        "mean": df_repeated.mean(),
        "median": df_repeated.median(),
        "mode": df_repeated.mode().iloc[0],
        "skewness": df_repeated.skew(),
        "kurtosis": df_repeated.kurt()
    }
    fig_age_dist = px.histogram(filtered_df, x='العمر Age', nbins=20, title="Age Distribution")
    fig_age_dist.add_vline(x=stats["mean"], line_dash="dot", line_color="red", annotation_text=f"Mean: {stats['mean']:.2f}")
    fig_age_dist.add_vline(x=stats["median"], line_dash="dash", line_color="orange", annotation_text=f"Median: {stats['median']:.2f}")
    fig_age_dist.add_vline(x=stats["mode"], line_dash="dashdot", line_color="purple", annotation_text=f"Mode: {stats['mode']:.2f}")
    st.plotly_chart(fig_age_dist, use_container_width=True)
    st.info("Interpretation: Displays how ages are distributed. Mean, median, and mode indicate center; skewness shows bias direction.")

    st.markdown("---")
    fig_nat_gender = px.histogram(filtered_df, x='الجنسية Nationality', color='Gender_English', barmode='group',
                                  title="Nationality by Gender")
    st.plotly_chart(fig_nat_gender, use_container_width=True)
    st.info("Interpretation: Compares genders across nationalities. Taller bars = more participants.")

    st.markdown("---")
    fig_demo = px.histogram(filtered_df, x='العمر Age', color='الجنسية Nationality', facet_col='Gender_English',
                            barmode='overlay', title="Demographic Characteristics: Age, Gender, Nationality")
    st.plotly_chart(fig_demo, use_container_width=True)
    st.info("Interpretation: Each facet shows age spread by nationality for each gender.")

    st.markdown("---")
    if 'اللغة Language' in filtered_df.columns:
        language_translation = {
            'Bahasa Indonesia': 'Indonesian',
            'Français': 'French',
            'Türkçe': 'Turkish',
            'বাংলা (Bengali)': 'Bengali',
            'اردو': 'Urdu',
            'English': 'English',
            'فارسی': 'Persian (Farsi)',
            'العربية': 'Arabic'
        }
        language_gender_ct = pd.crosstab(filtered_df['Gender_English'], filtered_df['اللغة Language'])
        bars = []
        for language in language_gender_ct.columns:
            label_language = language_translation.get(language, language)
            bars.append(go.Bar(
                name=label_language,
                x=language_gender_ct.index.tolist(),
                y=language_gender_ct[language].tolist()
            ))
        fig_lang = go.Figure(data=bars)
        fig_lang.update_layout(barmode='stack', title='Distribution of Gender and Language',
                               xaxis_title='Gender', yaxis_title='Count', template='plotly_white')
        st.plotly_chart(fig_lang, use_container_width=True)
        st.info("Interpretation: Shows how languages are distributed by gender.")

    st.markdown("---")
    if st.button("Back to Home"):
        st.session_state.page = "home"

# ---------------- ANALYZE COMMENTS (UPDATED) ----------------
def analyze():
    add_bg_from_local("background.png")
    st.title("Sentiment Classification with Model Toggle")

    if st.button("Back to Home"):
        st.session_state.page = "home"
        return

    use_bert = st.toggle("Use Fine-Tuned BERT Model (instead of VADER)")

    if use_bert:
        try:
            sentiment_model = pipeline("sentiment-analysis", model="./saved_model_bert", framework="pt")
            st.success("✅ Loaded Fine-Tuned BERT Model")
        except Exception as e:
            st.warning(f"Could not load BERT model: {e}")
            sentiment_model = None
    else:
        sentiment_model = SentimentIntensityAnalyzer()
        st.success("✅ Using VADER Sentiment Analyzer")

    def analyze_sentiment(text):
        if not isinstance(text, str) or not text.strip():
            return "N/A", 0.0
        if use_bert and sentiment_model:
            res = sentiment_model(text)[0]
            return res["label"], round(res["score"], 3)
        else:
            scores = sentiment_model.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                return "POSITIVE", compound
            elif compound <= -0.05:
                return "NEGATIVE", compound
            else:
                return "NEUTRAL", compound

    uploaded = st.file_uploader("Upload CSV/Excel/TXT with Comments", type=["csv", "xlsx", "txt"])
    manual = st.text_area("Enter Comments (one per line):", height=200)

    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        if "Comments" not in df.columns:
            st.error("Column 'Comments' not found.")
            return
        df[["Sentiment", "Score"]] = df["Comments"].apply(lambda x: pd.Series(analyze_sentiment(x)))
        st.dataframe(df.head(100))
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Sentiment Results", csv, "sentiment_results.csv", "text/csv")

    elif manual.strip():
        df = pd.DataFrame({"Comments": [line for line in manual.split("\n") if line.strip()]})
        df[["Sentiment", "Score"]] = df["Comments"].apply(lambda x: pd.Series(analyze_sentiment(x)))
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Sentiment Results", csv, "manual_sentiment_results.csv", "text/csv")
    else:
        st.info("📂 Upload a file or type comments to analyze.")

# ---------------- MAIN ROUTING ----------------
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
