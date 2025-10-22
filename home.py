
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
import pdfplumber
import documentation
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PILGRIMAGE DEMOGRAPHICS DASHBOARD", layout="wide")


# ---------------- BACKGROUND FUNCTION ----------------
def get_base64(fp):
    with open(fp, "rb") as f:
        return base64.b64encode(f.read()).decode()


def add_bg_from_local(image_file):
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
    try:
        img_b64 = get_base64("pilgrimage.png")
    except Exception:
        img_b64 = ""

    st.markdown(
        f"""
        <style>
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
          <p>PILGRIMAGEAI automatically analyzes and categorizes large-scale pilgrim feedback data.</p>
          <ul>
            <li>Automatic categorization across service areas</li>
            <li>Sentiment analysis for satisfaction insights</li>
            <li>Actionable intelligence for improvement</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("View Dashboard"):
        st.session_state.page = "dashboard"

    if st.button("Analyze Comments"):
        st.session_state.page = "analyze"

    if st.button("Documentation"):
        st.session_state.page = "documentation"


# ---------------- DASHBOARD ----------------
def dashboard():
    st.title("Real-Time Demographic Dashboard")

    st_autorefresh(interval=10 * 1000, limit=None, key="datarefresh")

    # --- Data Input ---
    data_source = st.radio("Select Data Source", ['Upload CSV', 'Enter API URL', 'Paste Raw CSV Text'])
    dataset = None

    if data_source == 'Upload CSV':
        uploaded = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
        if uploaded is not None:
            dataset = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)

    elif data_source == 'Enter API URL':
        api_url = st.text_input("Enter API URL returning CSV data")
        if api_url:
            dataset = pd.read_csv(StringIO(requests.get(api_url).text))

    elif data_source == 'Paste Raw CSV Text':
        raw_csv = st.text_area("Paste CSV data")
        if raw_csv:
            dataset = pd.read_csv(StringIO(raw_csv))

    if dataset is None:
        st.info("Upload or enter data to continue.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
        return

    # --- Validate Columns ---
    dataset.columns = dataset.columns.str.strip()
    required_cols = ['العمر Age', 'الجنسية Nationality', 'الجنس Gender']
    if not all(col in dataset.columns for col in required_cols):
        st.error("Missing required columns.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
        return

    # --- Gender Translation ---
    gender_map = {'أنثى': 'Female', 'ذكر': 'Male'}
    dataset['Gender_English'] = dataset['الجنس Gender'].map(gender_map)

    # --- Filters ---
    st.markdown("### Filter Data")
    col1, col2 = st.columns(2)
    with col1:
        genders = dataset['الجنس Gender'].dropna().unique()
        sel_genders = st.multiselect("Gender", genders, default=list(genders))
    with col2:
        nations = dataset['الجنسية Nationality'].dropna().unique()
        sel_nations = st.multiselect("Nationality", nations, default=list(nations))
    fdf = dataset[(dataset['الجنس Gender'].isin(sel_genders)) & (dataset['الجنسية Nationality'].isin(sel_nations))]

    if fdf.empty:
        st.warning("No data after filters.")
        return

    # --- KPIs ---
    st.markdown("### Summary")
    total = len(fdf)
    uniq_nat = fdf['الجنسية Nationality'].nunique()
    males = (fdf['Gender_English'] == 'Male').sum()
    females = (fdf['Gender_English'] == 'Female').sum()
    ratio = f"{males}:{females}" if males + females > 0 else "N/A"
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Respondents", total)
    c2.metric("Distinct Nationalities", uniq_nat)
    c3.metric("Gender Ratio (M:F)", ratio)

    # --- Age Distribution ---
    st.subheader("Age Distribution")
    df_age = fdf['العمر Age'].value_counts().reset_index()
    df_age.columns = ['Age', 'Count']
    df_age['Age'] = pd.to_numeric(df_age['Age'], errors='coerce')
    df_age = df_age.dropna().sort_values('Age')
    fig_age = px.histogram(fdf, x='العمر Age', nbins=20, title="Age Distribution", color_discrete_sequence=['#3498db'])
    st.plotly_chart(fig_age, use_container_width=True)

    # --- Frequency Tables + Downloads ---
    st.subheader(" Frequency Tables")

    freq_nat = fdf['الجنسية Nationality'].value_counts().reset_index()
    freq_nat.columns = ['Nationality', 'Count']
    st.dataframe(freq_nat)
    st.download_button("⬇️ Download Nationality CSV", freq_nat.to_csv(index=False).encode(), "nationality.csv")

    freq_gender = fdf['Gender_English'].value_counts().reset_index()
    freq_gender.columns = ['Gender', 'Count']
    st.dataframe(freq_gender)
    st.download_button("⬇️ Download Gender CSV", freq_gender.to_csv(index=False).encode(), "gender.csv")

    freq_age = df_age
    st.dataframe(freq_age)
    st.download_button("⬇️ Download Age CSV", freq_age.to_csv(index=False).encode(), "age.csv")

    if st.button("Back to Home"):
        st.session_state.page = "home"


# ---------------- ANALYZE COMMENTS ----------------
def analyze():
    add_bg_from_local("background.png")
    st.title("💬 Analyze Pilgrim Comments")

    if st.button("Back to Home"):
        st.session_state.page = "home"
        return

    st.markdown("### Choose Sentiment Model")
    model_choice = st.radio("Select Model", ["VADER (Lexicon-Based)", "Fine-tuned BERT (Transformer-Based)"])

    # --- Load Model ---
    if model_choice == "VADER (Lexicon-Based)":
        nltk.download("vader_lexicon")
        sid = SentimentIntensityAnalyzer()
        st.success("✅ VADER loaded.")

        def analyze_sentiment(text):
            s = sid.polarity_scores(text)
            comp = s["compound"]
            if comp >= 0.05:
                return "Positive", comp
            elif comp <= -0.05:
                return "Negative", comp
            return "Neutral", comp
    else:
        try:
            model_path = "./saved_model_bert"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            st.success("✅ BERT model loaded.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            model, tokenizer = None, None

        def analyze_sentiment(text):
            if not model or not tokenizer:
                return "Error", 0.0
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                out = model(**inputs)
                probs = torch.nn.functional.softmax(out.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                conf = probs[0][pred].item()
            label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            return label_map.get(pred, "Unknown"), round(conf, 3)

    # --- Helper Functions ---
    def classify_department(comment):
        topics = {
            "Customer Service": ["service", "support", "help", "rude", "friendly"],
            "Product Quality": ["defective", "quality", "broken", "excellent"],
            "Delivery": ["late", "delivery", "shipping", "on time"],
            "Billing": ["invoice", "bill", "charged", "refund"]
        }
        if not isinstance(comment, str):
            return "General"
        t = comment.lower()
        for theme, words in topics.items():
            if any(w in t for w in words):
                return theme
        return "General"

    cache = {}

    def translator_dual(text, src="auto", dest="en"):
        if pd.isnull(text):
            return None, None
        text = str(text).strip()
        if text not in cache:
            try:
                cache[text] = GoogleTranslator(source=src, target=dest).translate(text)
            except Exception as e:
                cache[text] = f"Error: {e}"
        return text, cache[text]

    # --- File/Manual Input ---
    uploaded = st.file_uploader("Upload CSV, Excel, PDF, TXT, or JSON", type=["csv", "xlsx", "pdf", "txt", "json"])
    manual = st.text_area("Or paste comments manually:", height=200)

    def process(df):
        df[["Original", "Translated"]] = df["Comments"].apply(lambda x: pd.Series(translator_dual(x)))
        df["Department"] = df["Translated"].apply(classify_department)
        df[["Sentiment", "Confidence"]] = df["Translated"].apply(lambda x: pd.Series(analyze_sentiment(x)))
        return df

    if uploaded:
        ext = uploaded.name.lower()
        if ext.endswith(".pdf"):
            with pdfplumber.open(uploaded) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            df = pd.DataFrame({"Comments": [l.strip() for l in text.split("\n") if l.strip()]})
        elif ext.endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif ext.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded)
        elif ext.endswith(".txt"):
            text = uploaded.read().decode("utf-8")
            df = pd.DataFrame({"Comments": [l.strip() for l in text.split("\n") if l.strip()]})
        elif ext.endswith(".json"):
            df = pd.read_json(uploaded)
        else:
            st.error("Unsupported format.")
            return
        if "Comments" not in df.columns:
            st.error("Column 'Comments' not found.")
            return

        with st.spinner("Analyzing uploaded file..."):
            res = process(df)
        st.success("✅ Completed.")
        st.dataframe(res)
        st.download_button("⬇️ Download CSV", res.to_csv(index=False).encode(), "sentiment_results.csv")
    elif manual.strip():
        lines = [x.strip() for x in manual.split("\n") if x.strip()]
        df = pd.DataFrame({"Comments": lines})
        with st.spinner("Analyzing manual comments..."):
            res = process(df)
        st.success("✅ Done.")
        st.dataframe(res)
        st.download_button("⬇️ Download CSV", res.to_csv(index=False).encode(), "manual_results.csv")
    else:
        st.info("Upload a file or type comments to analyze.")


# ---------------- ROUTING ----------------
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
