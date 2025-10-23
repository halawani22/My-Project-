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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pdfplumber
import documentation


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
          <p>PILGRIMAGEAI automatically analyzes and categorizes large-scale pilgrim feedback data.</p>
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
# (Keep your existing dashboard code exactly as it is — not modified)


# ---------------- ANALYZE COMMENTS PAGE ----------------
def analyze():
    """
    Restored full comment analyzer:
    - Handles CSV, Excel, PDF, TXT, JSON
    - Processes data in chunks
    - Translates to English
    - Classifies department
    - Analyzes sentiment using BERT or fallback to VADER
    - Returns confidence scores
    """
    add_bg_from_local("background.png")
    st.title("💬 Comprehensive Comment Analysis — Sentiment & Department Classification")

    if st.button("Back to Home"):
        st.session_state.page = "home"
        return

    st.info("Upload or paste comments for automatic translation, categorization, and sentiment analysis.")

    # --- Load sentiment models ---
    try:
        bert_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        st.success("✅ Loaded multilingual BERT sentiment model.")
    except Exception as e:
        bert_pipeline = None
        st.warning(f"⚠️ Could not load BERT, fallback to VADER. ({e})")

    vader_analyzer = SentimentIntensityAnalyzer()

    # --- Department keywords ---
    department_keywords = {
        "Customer Service": ["service", "support", "help", "rude", "staff", "friendly"],
        "Accommodation": ["hotel", "room", "stay", "clean", "toilet", "bed"],
        "Transport": ["bus", "driver", "transport", "delay", "car"],
        "Food & Catering": ["food", "meal", "restaurant", "breakfast", "catering"],
        "Religious Guidance": ["imam", "sermon", "guidance", "religious"],
        "General Services": ["general", "other"]
    }

    translation_cache = {}

    def translate_text(text):
        """Translate non-English text to English with caching."""
        if not isinstance(text, str) or not text.strip():
            return ""
        if text in translation_cache:
            return translation_cache[text]
        try:
            translated = GoogleTranslator(source="auto", target="en").translate(text)
            translation_cache[text] = translated
            return translated
        except Exception:
            return text

    def classify_department(comment):
        text = comment.lower()
        for dept, words in department_keywords.items():
            if any(w in text for w in words):
                return dept
        return "General Services"

    def analyze_sentiment(text):
        """Use BERT or fallback VADER for sentiment and confidence."""
        if not text.strip():
            return "Neutral", 0.0
        if bert_pipeline:
            try:
                result = bert_pipeline(text[:512])[0]
                label = result["label"]
                score = result["score"]
                if "1" in label or "2" in label:
                    return "Negative", score
                elif "3" in label:
                    return "Neutral", score
                else:
                    return "Positive", score
            except Exception:
                pass

        score = vader_analyzer.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "Positive", score
        elif score <= -0.05:
            return "Negative", score
        else:
            return "Neutral", score

    def extract_comments_in_chunks(file, chunksize=10000):
        """Yield data chunks for large file support."""
        fname = file.name.lower()
        if fname.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            yield pd.DataFrame({"Comments": lines})

        elif fname.endswith(".txt"):
            text = file.read().decode("utf-8", errors="ignore")
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            yield pd.DataFrame({"Comments": lines})

        elif fname.endswith(".csv"):
            for chunk in pd.read_csv(file, chunksize=chunksize):
                chunk.columns = [c.strip() for c in chunk.columns]
                if "Comments" in chunk.columns:
                    yield chunk[["Comments"]]

        elif fname.endswith(".xlsx") or fname.endswith(".xls"):
            df = pd.read_excel(file)
            if "Comments" in df.columns:
                yield df[["Comments"]]

        elif fname.endswith(".json"):
            df = pd.read_json(file)
            if "Comments" in df.columns:
                yield df[["Comments"]]

        else:
            st.error("Unsupported file type.")
            yield None

    def process_chunk(chunk):
        """Translate, classify, and analyze sentiment for each comment."""
        chunk["Original_Comment"] = chunk["Comments"]
        chunk["Translated_Comment"] = chunk["Comments"].apply(translate_text)
        chunk["Department"] = chunk["Translated_Comment"].apply(classify_department)
        sentiment_results = chunk["Translated_Comment"].apply(analyze_sentiment)
        chunk["Sentiment"] = sentiment_results.apply(lambda x: x[0])
        chunk["Confidence"] = sentiment_results.apply(lambda x: round(x[1], 3))
        return chunk[["Original_Comment", "Translated_Comment", "Department", "Sentiment", "Confidence"]]

    uploaded_file = st.file_uploader("Upload comments file (CSV, Excel, PDF, TXT, JSON)", type=["csv", "xlsx", "pdf", "txt", "json"])
    manual_input = st.text_area("Or paste comments manually (one per line):", height=200)

    results = []

    if uploaded_file:
        total_rows = 0
        st.info("Processing uploaded file — large files handled in chunks for efficiency.")
        progress = st.progress(0)
        for chunk in extract_comments_in_chunks(uploaded_file):
            if chunk is None:
                continue
            processed = process_chunk(chunk)
            results.append(processed)
            total_rows += len(processed)
            progress.progress(min(total_rows / 1000000, 1.0))
        if results:
            df_final = pd.concat(results, ignore_index=True)
            st.success(f"✅ Analysis complete! Processed {len(df_final)} comments.")
            st.dataframe(df_final.head(500))
            csv = df_final.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Full Results", csv, "comment_analysis_results.csv", "text/csv")

    elif manual_input.strip():
        lines = [l.strip() for l in manual_input.split("\n") if l.strip()]
        df_manual = pd.DataFrame({"Comments": lines})
        with st.spinner("Analyzing manual comments..."):
            df_final = process_chunk(df_manual)
        st.success("✅ Manual analysis complete!")
        st.dataframe(df_final)
        csv = df_final.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "manual_comment_results.csv", "text/csv")

    else:
        st.info("📂 Upload a file or enter comments to start analysis.")


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
