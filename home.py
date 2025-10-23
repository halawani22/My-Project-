# home.py
import streamlit as st
import matplotlib.pyplot as plt
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
def dashboard():
    """Real-Time Demographic Dashboard (filters, KPIs, charts, frequency tables)."""
    st.title("Real-Time Demographic Dashboard")

    # Auto-refresh every 10 seconds to simulate real-time updates when data source changes
    st_autorefresh(interval=10 * 1000, limit=None, key="datarefresh")

    # ----------- DATA INPUT -----------
    data_source = st.radio("Select Data Source", ['Upload CSV', 'Enter API URL', 'Paste Raw CSV Text'])
    dataset = None

    if data_source == 'Upload CSV':
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls', 'ods', 'txt', 'pdf'])
        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1].lower()
            try:
                if file_type in ['csv', 'txt']:
                    dataset = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
                elif file_type in ['xls', 'xlsx', 'ods']:
                    dataset = pd.read_excel(uploaded_file)
                elif file_type == 'pdf':
                    st.error("PDF uploads are supported on the analyze page only. Please use CSV/Excel here.")
                else:
                    st.error(f"Unsupported file type: {file_type}")
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

    # ----------- COLUMN CHECK -----------
    dataset.columns = dataset.columns.str.strip()
    required_cols = ['العمر Age', 'الجنسية Nationality', 'الجنس Gender']
    if not all(col in dataset.columns for col in required_cols):
        st.error("❌ Required columns not found in uploaded data. Required: 'العمر Age', 'الجنسية Nationality', 'الجنس Gender'")
        if st.button("Back to Home"):
            st.session_state.page = "home"
        return

    # ----------- GENDER TRANSLATION & PREP -----------
    gender_map = {'أنثى': 'Female', 'ذكر': 'Male'}
    dataset['Gender_English'] = dataset['الجنس Gender'].map(gender_map)

    # ----------- FILTERS -----------
    st.markdown("###  Filter Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender_options = dataset['الجنس Gender'].dropna().unique()
        selected_genders = st.multiselect("Filter by Gender", options=gender_options, default=list(gender_options))

    with col2:
        nationality_options = dataset['الجنسية Nationality'].dropna().unique()
        selected_nationalities = st.multiselect("Filter by Nationality", options=nationality_options, default=list(nationality_options))

    # Optional date detection & filter
    date_cols = [c for c in dataset.columns if 'date' in c.lower()]
    date_column = None
    date_range = None
    if date_cols:
        date_column = date_cols[0]
        dataset[date_column] = pd.to_datetime(dataset[date_column], errors='coerce')
        min_date = dataset[date_column].min()
        max_date = dataset[date_column].max()
        with col3:
            date_range = st.date_input("Filter by Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    else:
        with col3:
            st.write("No date column found for filtering")

    # Apply filters to dataset
    filtered_df = dataset[
        dataset['الجنس Gender'].isin(selected_genders) &
        dataset['الجنسية Nationality'].isin(selected_nationalities)
    ]

    if date_column and date_range and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (filtered_df[date_column] >= start_date) &
            (filtered_df[date_column] <= end_date)
        ]

    if filtered_df.empty:
        st.warning("No data after applying filters.")
        return

    # ---------------- KPI SUMMARY CARDS ----------------
    st.markdown("### Summary Statistics (Filtered)")
    total_respondents = len(filtered_df)
    distinct_nations = filtered_df['الجنسية Nationality'].nunique()
    male_count = (filtered_df['Gender_English'] == 'Male').sum()
    female_count = (filtered_df['Gender_English'] == 'Female').sum()
    # safe ratio formatting (avoid divide by zero)
    if male_count + female_count > 0:
        gender_ratio = f"{male_count} : {female_count} (M:F)"
    else:
        gender_ratio = "N/A"

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Respondents", f"{total_respondents:,}")
    k2.metric("Distinct Nationalities", distinct_nations)
    k3.metric("Gender Ratio (M:F)", gender_ratio)

    st.markdown("---")

    # ---------------- AGE DISTRIBUTION & STATS ----------------
    st.subheader("Age Distribution with Statistical Markers (Filtered Data)")

    # Build age frequency table robustly and ensure numeric sorting
    df_age = filtered_df['العمر Age'].value_counts().reset_index()
    # rename dynamically to avoid KeyError
    df_age.columns = ['Age', 'Count']
    df_age['Age'] = pd.to_numeric(df_age['Age'], errors='coerce')
    df_age = df_age.dropna(subset=['Age']).sort_values('Age')

    # Expand ages for correct statistics
    df_repeated = df_age['Age'].repeat(df_age['Count']).astype(float)

    # compute stats safely
    stats = {
        "mean": df_repeated.mean() if not df_repeated.empty else None,
        "median": df_repeated.median() if not df_repeated.empty else None,
        "mode": (df_repeated.mode().iloc[0] if not df_repeated.mode().empty else None),
        "min": df_repeated.min() if not df_repeated.empty else None,
        "max": df_repeated.max() if not df_repeated.empty else None,
        "skewness": df_repeated.skew() if not df_repeated.empty else None,
        "kurtosis": df_repeated.kurt() if not df_repeated.empty else None
    }

    fig_age_dist = px.histogram(filtered_df, x='العمر Age', nbins=20, title="Age Distribution (Filtered Data)")
    if stats["mean"] is not None:
        fig_age_dist.add_vline(x=stats["mean"], line_dash="dot", line_color="red", annotation_text=f"Mean: {stats['mean']:.2f}")
    if stats["median"] is not None:
        fig_age_dist.add_vline(x=stats["median"], line_dash="dash", line_color="orange", annotation_text=f"Median: {stats['median']:.2f}")
    if stats["mode"] is not None:
        fig_age_dist.add_vline(x=stats["mode"], line_dash="dashdot", line_color="purple", annotation_text=f"Mode: {stats['mode']:.2f}")

    st.plotly_chart(fig_age_dist, use_container_width=True)

    st.info(
        f"""
        **Interpretation (Age Distribution):**  
        - **Mean** (avg): {stats['mean']:.2f} if available.  
        - **Median**: {stats['median']:.2f} if available.  
        - **Mode**: {stats['mode']:.2f} if available.  
        - **Skewness**: {stats['skewness']:.2f} (positive → right-skew; negative → left-skew).  
        - **Kurtosis**: {stats['kurtosis']:.2f} (higher → heavier tails / more outliers).
        """
    )

    st.markdown("---")

    # ---------------- AGE COUNT LINE CHART ----------------
    df_age_counts = df_age.copy()
    fig_age_line = px.line(df_age_counts, x='Age', y='Count', markers=True, title="Age Counts by Age")
    st.plotly_chart(fig_age_line, use_container_width=True)
    st.info("Interpretation: line shows counts for each age value. Peaks indicate common ages.")

    st.markdown("---")

    # ---------------- NATIONALITY BY GENDER ----------------
    fig_nat_gender = px.histogram(filtered_df, x='الجنسية Nationality', color='Gender_English', barmode='group',
                                  title="Nationality by Gender (Filtered Data)")
    st.plotly_chart(fig_nat_gender, use_container_width=True)
    st.info("Interpretation: Compare bars per nationality to see relative gender representation per country.")

    st.markdown("---")

    # ---------------- BUBBLE CHART (Nationality & Gender by Mean Age) ----------------
    agg = filtered_df.groupby(['الجنسية Nationality', 'Gender_English']).agg(
        count=('العمر Age', 'count'),
        avg_age=('العمر Age', 'mean')
    ).reset_index()
    if not agg.empty:
        agg['avg_age'] = pd.to_numeric(agg['avg_age'], errors='coerce').round(1)
        fig_bubble = px.scatter(agg, x='الجنسية Nationality', y='count', size='avg_age', color='الجنسية Nationality',
                                facet_col='Gender_English', title="Nationality and Gender by Mean Age", size_max=40)
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.info("Interpretation: bubble size = average age; y-axis = count. Larger bubble = higher mean age for that group.")

    st.markdown("---")

    # ---------------- FULL DEMOGRAPHICS FACETED HISTOGRAM ----------------
    fig_demo = px.histogram(filtered_df, x='العمر Age', color='الجنسية Nationality', facet_col='Gender_English',
                            barmode='overlay', title="Demographic Characteristics: Age, Gender, Nationality")
    st.plotly_chart(fig_demo, use_container_width=True)
    st.info("Interpretation: Facets allow comparison of age distributions across genders and nationalities.")

    st.markdown("---")

    # ---------------- GENDER-LANGUAGE STACKED BAR (optional) ----------------
    if 'اللغة Language' in filtered_df.columns:
        st.subheader(" Gender–Language Interaction Analysis (Filtered Data)")

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
        # Ensure Gender_English column exists in filtered_df
        if 'Gender_English' not in filtered_df.columns:
            filtered_df['Gender_English'] = filtered_df['الجنس Gender'].map(gender_map)

        language_gender_ct = pd.crosstab(filtered_df['Gender_English'], filtered_df['اللغة Language'])
        bars = []
        for language in language_gender_ct.columns:
            label_language = language_translation.get(language, language)
            bars.append(go.Bar(
                name=label_language,
                x=language_gender_ct.index.tolist(),
                y=language_gender_ct[language].tolist(),
                hovertext=[f"Gender: {g}<br>Language: {label_language}<br>Count: {c}" for g, c in zip(language_gender_ct.index.tolist(), language_gender_ct[language].tolist())],
                hoverinfo='text'
            ))
        fig_lang = go.Figure(data=bars)
        fig_lang.update_layout(barmode='stack', title='Distribution of Gender and Language', xaxis_title='Gender', yaxis_title='Count', template='plotly_white')
        st.plotly_chart(fig_lang, use_container_width=True)
        st.info("Interpretation: stacked bars show language composition within each gender.")

    else:
        st.info("Language column ('اللغة Language') not found — Gender–Language analysis is optional.")

    st.markdown("---")

    # ---------------- FREQUENCY TABLES + CSV DOWNLOADS ----------------
    st.subheader(" Frequency Distribution Tables (Filtered Data)")

    # Nationality table + download
    freq_nat = filtered_df['الجنسية Nationality'].value_counts().reset_index()
    freq_nat.columns = ['Nationality', 'Frequency']
    freq_nat['Percentage'] = (freq_nat['Frequency'] / freq_nat['Frequency'].sum() * 100).round(2)
    st.markdown("#### Nationality Frequency Table")
    st.dataframe(freq_nat)
    st.download_button("⬇️ Download Nationality CSV", freq_nat.to_csv(index=False).encode('utf-8'), "nationality_freq.csv", "text/csv")
    st.info("Interpretation: percentages show relative share; frequency shows counts.")

    # Gender table + download
    freq_gender = filtered_df['Gender_English'].value_counts().reset_index()
    freq_gender.columns = ['Gender', 'Frequency']
    freq_gender['Percentage'] = (freq_gender['Frequency'] / freq_gender['Frequency'].sum() * 100).round(2)
    st.markdown("#### Gender Frequency Table")
    st.dataframe(freq_gender)
    st.download_button("⬇️ Download Gender CSV", freq_gender.to_csv(index=False).encode('utf-8'), "gender_freq.csv", "text/csv")
    st.info("Interpretation: quick look at gender proportions in filtered data.")

    # Age table + download
    freq_age = filtered_df['العمر Age'].value_counts().reset_index()
    freq_age.columns = ['Age', 'Frequency']
    freq_age = freq_age.sort_values('Age')
    freq_age['Percentage'] = (freq_age['Frequency'] / freq_age['Frequency'].sum() * 100).round(2)
    st.markdown("#### Age Frequency Table")
    st.dataframe(freq_age)
    st.download_button("⬇️ Download Age CSV", freq_age.to_csv(index=False).encode('utf-8'), "age_freq.csv", "text/csv")
    st.info("Interpretation: which ages dominate? use mean/median/skew to interpret distribution.")

    st.markdown("---")

    # ---------------- STATISTICAL DISTRIBUTION VISUALIZATIONS ----------------
    st.subheader(" Statistical Distribution Visualizations")
    st.markdown("---")

    # --- Nationality Frequency Bar Plot ---
    freq_table = filtered_df['الجنسية Nationality'].value_counts().reset_index()
    freq_table.columns = ['Nationality', 'Frequency']

    # Filter out 'Total' row if it exists
    freq_tableplt = freq_table[freq_table["Nationality"] != "Total"]

    fig, ax = plt.subplots(figsize=(6, 3))  # compact figure size
    freq_tableplt.plot(
        kind="bar",
        x="Nationality",
        y="Frequency",
        legend=False,
        color="steelblue",
        ax=ax
    )
    plt.title("Nationality: Frequency Distribution")
    plt.xlabel("Nationality")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    for i, v in enumerate(freq_tableplt["Frequency"]):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.info("Interpretation: Bar heights represent the number of respondents per nationality. Peaks indicate the most common nationalities.")

    # --- Gender Proportional Pie Chart ---
    freq_table_gender = filtered_df['Gender_English'].value_counts().reset_index()
    freq_table_gender.columns = ['Gender', 'Frequency']

    # Filter out 'Total' row if it exists
    freq_tableplt = freq_table_gender[freq_table_gender["Gender"] != "Total"].set_index("Gender")

    fig, ax = plt.subplots(figsize=(3, 3))  # square figure for pie chart
    freq_tableplt["Frequency"].plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90,
        ax=ax
    )
    plt.title("Gender: Proportionality Distribution")
    plt.ylabel("")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.info("Interpretation: Pie slices show proportion of each gender in the filtered data.")

    st.markdown("---")

    if st.button("Back to Home"):
        st.session_state.page = "home"


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











