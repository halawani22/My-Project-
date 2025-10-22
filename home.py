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
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PILGRIMAGE DEMOGRAPHICS DASHBOARD", layout="wide")

# ---------------- BACKGROUND FUNCTION ----------------
def get_base64(fp):
    with open(fp, "rb") as f:
        return base64.b64encode(f.read()).decode()

def add_bg_from_local(image_file):
    """Adds static background for pages."""
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
    """Landing Page"""
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
            background-repeat: no-repeat;
        }}
        .overlay {{
            background-color: rgba(255,255,255,0.85);
            padding: 2rem;
            border-radius: 1rem;
            max-width: 650px;
            margin: 8vh auto;
            text-align: center;
        }}
        .overlay h1, .overlay h2 {{
            color: #DAA520;
        }}
        </style>
        <div class="overlay">
            <h1>PILGRIMAGEAI</h1>
            <h2>Voice of the Pilgrims</h2>
            <p>AI-powered analysis of pilgrim feedback and demographics.</p>
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
    """Real-Time Demographic Dashboard"""
    st.title("Real-Time Demographic Dashboard")
    st_autorefresh(interval=10 * 1000, limit=None, key="datarefresh")

    # === Data Input ===
    data_source = st.radio("Select Data Source", ['Upload CSV', 'Enter API URL', 'Paste Raw CSV Text'])
    dataset = None

    if data_source == 'Upload CSV':
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                dataset = pd.read_csv(uploaded_file)
            else:
                dataset = pd.read_excel(uploaded_file)

    elif data_source == 'Enter API URL':
        api_url = st.text_input("Enter API URL returning CSV data")
        if api_url:
            response = requests.get(api_url)
            dataset = pd.read_csv(StringIO(response.text))

    elif data_source == 'Paste Raw CSV Text':
        raw_csv = st.text_area("Paste your CSV data here")
        if raw_csv:
            dataset = pd.read_csv(StringIO(raw_csv))

    if dataset is None:
        st.info("Please upload or enter data to continue.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
        return

    dataset.columns = dataset.columns.str.strip()
    required = ['العمر Age', 'الجنسية Nationality', 'الجنس Gender']
    if not all(col in dataset.columns for col in required):
        st.error("Missing columns: 'العمر Age', 'الجنسية Nationality', 'الجنس Gender'")
        return

    # === Filters ===
    gender_map = {'أنثى': 'Female', 'ذكر': 'Male'}
    dataset['Gender_English'] = dataset['الجنس Gender'].map(gender_map)
    st.markdown("### Filters")
    col1, col2 = st.columns(2)
    with col1:
        genders = st.multiselect("Gender", dataset['الجنس Gender'].unique(), default=list(dataset['الجنس Gender'].unique()))
    with col2:
        nations = st.multiselect("Nationality", dataset['الجنسية Nationality'].unique(), default=list(dataset['الجنسية Nationality'].unique()))

    filtered_df = dataset[dataset['الجنس Gender'].isin(genders) & dataset['الجنسية Nationality'].isin(nations)]
    if filtered_df.empty:
        st.warning("No data after filtering.")
        return

    # === KPI SUMMARY ===
    total = len(filtered_df)
    nat_unique = filtered_df['الجنسية Nationality'].nunique()
    male = (filtered_df['Gender_English'] == 'Male').sum()
    female = (filtered_df['Gender_English'] == 'Female').sum()
    ratio = f"{male}:{female} (M:F)" if male + female > 0 else "N/A"
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Respondents", total)
    c2.metric("Distinct Nationalities", nat_unique)
    c3.metric("Gender Ratio", ratio)

    st.markdown("---")

    # === AGE DISTRIBUTION ===
    st.subheader("Age Distribution with Statistics")
    df_age = filtered_df['العمر Age'].value_counts().reset_index()
    df_age.columns = ['Age', 'Count']
    df_age['Age'] = pd.to_numeric(df_age['Age'], errors='coerce')
    df_age = df_age.dropna().sort_values('Age')
    df_rep = df_age['Age'].repeat(df_age['Count']).astype(float)

    stats = {
        "mean": df_rep.mean(),
        "median": df_rep.median(),
        "mode": df_rep.mode().iloc[0],
        "skew": df_rep.skew(),
        "kurt": df_rep.kurt(),
    }

    fig = px.histogram(filtered_df, x='العمر Age', nbins=20, title="Age Distribution")
    fig.add_vline(x=stats["mean"], line_color='red', line_dash='dot', annotation_text=f"Mean: {stats['mean']:.1f}")
    fig.add_vline(x=stats["median"], line_color='orange', line_dash='dash', annotation_text=f"Median: {stats['median']:.1f}")
    fig.add_vline(x=stats["mode"], line_color='purple', line_dash='dashdot', annotation_text=f"Mode: {stats['mode']:.1f}")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Interpretation: Compare mean, median, and mode to identify skewness. Right-skew → older majority, left-skew → younger majority.")

    # === Frequency Tables ===
    st.markdown("### Frequency Tables")
    for colname, label in [('الجنسية Nationality', 'Nationality'), ('Gender_English', 'Gender'), ('العمر Age', 'Age')]:
        freq = filtered_df[colname].value_counts().reset_index()
        freq.columns = [label, "Count"]
        freq["Percent"] = (freq["Count"] / freq["Count"].sum() * 100).round(2)
        st.markdown(f"#### {label} Frequency Table")
        st.dataframe(freq)
        st.download_button(f"⬇️ Download {label} CSV", freq.to_csv(index=False).encode('utf-8'), f"{label.lower()}_freq.csv")

    st.markdown("---")

    if st.button("Back to Home"):
        st.session_state.page = "home"

# ---------------- ANALYZE COMMENTS ----------------
def analyze():
    add_bg_from_local("background.png")
    st.title("Sentiment Analysis of Comments")

    if st.button("Back to Home"):
        st.session_state.page = "home"
        return

    st.markdown("### Choose Sentiment Model")
    use_bert = st.toggle("Use Fine-Tuned BERT Model (instead of VADER)")

    # Initialize VADER
    nltk.download("vader_lexicon")
    vader = SentimentIntensityAnalyzer()

    # Load BERT model only if toggle is active
    bert_model = None
    bert_tokenizer = None
    if use_bert:
        try:
            save_path = "./saved_model_bert"
            bert_tokenizer = AutoTokenizer.from_pretrained(save_path)
            bert_model = AutoModelForSequenceClassification.from_pretrained(save_path)
            st.success("✅ Loaded fine-tuned BERT model successfully!")
        except Exception as e:
            st.error(f"⚠️ Failed to load BERT model: {e}")
            use_bert = False

    uploaded = st.file_uploader("Upload CSV/Excel with a 'Comments' column", type=["csv", "xlsx"])
    text_input = st.text_area("Or enter comments manually (one per line):")

    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    elif text_input.strip():
        df = pd.DataFrame({"Comments": [x.strip() for x in text_input.split("\n") if x.strip()]})
    else:
        st.info("Upload a file or enter text to start.")
        return

    if "Comments" not in df.columns:
        st.error("No 'Comments' column found.")
        return

    st.info("Translating and analyzing sentiment... please wait.")
    df["Translated"] = df["Comments"].apply(lambda x: GoogleTranslator(source="auto", target="en").translate(x))

    results = []
    for text in df["Translated"]:
        if use_bert and bert_model is not None:
            inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = bert_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                label_id = torch.argmax(probs).item()
                label = ["negative", "neutral", "positive"][label_id]
                score = probs[0][label_id].item()
        else:
            scores = vader.polarity_scores(text)
            comp = scores["compound"]
            label = "positive" if comp >= 0.05 else "negative" if comp <= -0.05 else "neutral"
            score = comp
        results.append({"Comment": text, "Sentiment": label, "Score": round(float(score), 3)})

    res_df = pd.DataFrame(results)
    st.success("✅ Sentiment analysis complete!")
    st.dataframe(res_df)
    st.download_button("⬇️ Download Results CSV", res_df.to_csv(index=False).encode("utf-8"), "sentiment_results.csv")

# ---------------- MAIN ----------------
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
