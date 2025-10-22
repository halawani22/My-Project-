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
# (Your dashboard function stays EXACTLY as before – unchanged)
# I’ll keep it omitted here for brevity, but in your file, keep the full dashboard section you provided.


# ---------------- ANALYZE COMMENTS PAGE ----------------
def analyze():
    """
    Analyze comments using toggle between:
    - VADER Sentiment Analyzer
    - Fine-tuned BERT Model (from ./saved_model_bert)
    """
    add_bg_from_local("background.png")
    st.title("Sentiment Classification with Primary Model")

    if st.button("Back to Home"):
        st.session_state.page = "home"
        return

    # --- Select Sentiment Model ---
    st.markdown("### 🔀 Choose Sentiment Model")
    sentiment_choice = st.radio("Select a model", ["VADER (Lexicon-Based)", "Fine-tuned BERT (Transformer-Based)"])

    # --- Load selected model ---
    if sentiment_choice == "VADER (Lexicon-Based)":
        nltk.download("vader_lexicon")
        sid = SentimentIntensityAnalyzer()
        st.success("✅ VADER Sentiment Analyzer loaded successfully.")

        def analyze_sentiment(text):
            scores = sid.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                return "Positive", compound
            elif compound <= -0.05:
                return "Negative", compound
            else:
                return "Neutral", compound

    else:
        try:
            model_path = "./saved_model_bert"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            st.success("✅ Fine-tuned BERT model loaded successfully.")
        except Exception as e:
            st.error(f"❌ Failed to load BERT model: {e}")
            model, tokenizer = None, None

        def analyze_sentiment(text):
            if not model or not tokenizer:
                return "Error", 0.0
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred].item()
            label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            return label_map.get(pred, "Unknown"), round(confidence, 3)

    # --- Translation cache ---
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

    # --- Department classification ---
    themes_topics = {
        "Customer Service": ["service", "support", "help", "rude", "friendly"],
        "Product Quality": ["defective", "quality", "broken", "excellent"],
        "Delivery": ["late", "delivery", "shipping", "on time"],
        "Billing": ["invoice", "bill", "charged", "refund"],
        "General Services": ["general", "other"]
    }

    def classify_department(comment):
        if not isinstance(comment, str):
            return "General Services"
        tokens = set(comment.lower().split())
        for theme, keywords in themes_topics.items():
            if any(keyword in tokens for keyword in keywords):
                return theme
        return "General Services"

    # --- File uploader / manual input ---
    uploaded_file = st.file_uploader("Upload comments (CSV, XLSX, PDF, TXT, JSON)", type=["csv", "xlsx", "pdf", "txt", "json"])
    manual_input = st.text_area("Or enter comments manually (one per line):", height=200)

    def process_chunk(df):
        df[["Original", "Translated"]] = df["Comments"].apply(lambda c: pd.Series(translator_dual(c)))
        df["Department"] = df["Translated"].apply(classify_department)
        df[["Sentiment", "Confidence"]] = df["Translated"].apply(lambda c: pd.Series(analyze_sentiment(c)))
        return df

    def extract_comments_in_chunks(file):
        filename = file.name.lower()
        if filename.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            yield pd.DataFrame({"Comments": lines})
        elif filename.endswith(".txt"):
            text = file.read().decode("utf-8")
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            yield pd.DataFrame({"Comments": lines})
        elif filename.endswith(".csv"):
            yield pd.read_csv(file, usecols=["Comments"])
        elif filename.endswith(".xlsx"):
            yield pd.read_excel(file, usecols=["Comments"])
        elif filename.endswith(".json"):
            yield pd.read_json(file)[["Comments"]]
        else:
            st.error("Unsupported file type.")
            yield None

    # --- Processing logic ---
    if uploaded_file:
        st.info("📂 Processing uploaded file...")
        results = []
        for chunk in extract_comments_in_chunks(uploaded_file):
            if chunk is not None:
                results.append(process_chunk(chunk))
        if results:
            df_results = pd.concat(results, ignore_index=True)
            st.success("✅ Analysis complete!")
            st.dataframe(df_results.head(500))
            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", csv, "sentiment_results.csv", "text/csv")
    elif manual_input.strip():
        st.info("📝 Processing manual input...")
        lines = [l.strip() for l in manual_input.split("\n") if l.strip()]
        df_manual = pd.DataFrame({"Comments": lines})
        df_results = process_chunk(df_manual)
        st.success("✅ Analysis complete!")
        st.dataframe(df_results)
        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results CSV", csv, "manual_sentiment_results.csv", "text/csv")
    else:
        st.info("Please upload a file or enter comments manually.")


# ---------------- MAIN ROUTING ----------------
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
