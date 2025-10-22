import streamlit as st
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import StringIO
from streamlit_autorefresh import st_autorefresh
import documentation


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PILGRIMAGE DEMOGRAPHICS DASHBOARD", layout="wide")


# ---------------- BACKGROUND FUNCTION ----------------
def get_base64(fp):
    with open(fp, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---------------- HOME PAGE ----------------
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
    """, unsafe_allow_html=True)

    if st.button("View Dashboard"):
        st.session_state.page = "dashboard"

    if st.button("Documentation"):
        st.session_state.page = "documentation"


# ---------------- DASHBOARD ----------------
def dashboard():
    st.title("📊 Real-Time Demographic Dashboard")

    # Auto-refresh every 10 seconds
    st_autorefresh(interval=10 * 1000, limit=None, key="datarefresh")

    # ----------- DATA INPUT -----------
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
        raw_csv = st.text_area("Paste CSV data")
        if raw_csv:
            dataset = pd.read_csv(StringIO(raw_csv))

    if dataset is None:
        st.info("Please upload or enter data to continue.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
        return

    # ----------- COLUMN CHECK -----------
    dataset.columns = dataset.columns.str.strip()
    required_cols = ['العمر Age', 'الجنسية Nationality', 'الجنس Gender']
    if not all(col in dataset.columns for col in required_cols):
        st.error("❌ Required columns not found.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
        return

    # ----------- GENDER TRANSLATION -----------
    gender_map = {'أنثى': 'Female', 'ذكر': 'Male'}
    dataset['Gender_English'] = dataset['الجنس Gender'].map(gender_map)

    # ----------- FILTERS -----------
    st.markdown("### 🔍 Filter Data")
    col1, col2 = st.columns(2)

    with col1:
        genders = dataset['الجنس Gender'].dropna().unique()
        sel_genders = st.multiselect("Gender", genders, default=list(genders))
    with col2:
        nationalities = dataset['الجنسية Nationality'].dropna().unique()
        sel_nationalities = st.multiselect("Nationality", nationalities, default=list(nationalities))

    fdf = dataset[
        dataset['الجنس Gender'].isin(sel_genders) &
        dataset['الجنسية Nationality'].isin(sel_nationalities)
    ]

    if fdf.empty:
        st.warning("No data after filters.")
        return

    # ----------- KPI SUMMARY -----------
    st.markdown("### 📌 Summary Statistics")
    total_pilgrims = len(fdf)
    unique_nations = fdf['الجنسية Nationality'].nunique()
    male_count = (fdf['Gender_English'] == 'Male').sum()
    female_count = (fdf['Gender_English'] == 'Female').sum()
    gender_ratio = f"{(male_count / (male_count + female_count) * 100):.1f}% Male / {(female_count / (male_count + female_count) * 100):.1f}% Female"

    colA, colB, colC = st.columns(3)
    colA.metric("🧍 Total Respondents", total_pilgrims)
    colB.metric("🌍 Distinct Nationalities", unique_nations)
    colC.metric("⚧ Gender Ratio", gender_ratio)

    # ---------------- AGE DISTRIBUTION ----------------
    st.subheader("📈 Age Distribution with Statistical Markers")

    df_age = fdf["العمر Age"].value_counts().reset_index()
    df_age.columns = ["Age", "Count"]
    df_age["Age"] = pd.to_numeric(df_age["Age"], errors="coerce")
    df_age = df_age.dropna().sort_values("Age")

    df_rep = df_age["Age"].repeat(df_age["Count"]).astype(float)
    stats = {
        "mean": df_rep.mean(),
        "median": df_rep.median(),
        "mode": df_rep.mode().iloc[0] if not df_rep.mode().empty else None,
        "min": df_rep.min(),
        "max": df_rep.max(),
        "skewness": df_rep.skew(),
        "kurtosis": df_rep.kurt(),
    }

    fig_age = px.histogram(fdf, x="العمر Age", nbins=20, title="Age Distribution", color_discrete_sequence=["#3498db"])
    for stat, color in [("mean", "red"), ("median", "orange"), ("mode", "purple")]:
        if stats[stat]:
            fig_age.add_vline(x=stats[stat], line_dash="dot", line_color=color, annotation_text=stat.capitalize())
    st.plotly_chart(fig_age, use_container_width=True)
    st.info(
        f"""
        **Interpretation:**  
        - **Mean ({stats['mean']:.2f})** shows the average age.  
        - **Median ({stats['median']:.2f})** splits the age group evenly.  
        - **Mode ({stats['mode']:.2f})** is the most common age.  
        - **Skewness ({stats['skewness']:.2f})**: positive = younger-heavy.  
        - **Kurtosis ({stats['kurtosis']:.2f})**: higher = ages concentrated around mean.
        """
    )

    # ---------------- FREQUENCY DISTRIBUTIONS ----------------
    st.subheader("📊 Frequency Distribution Analysis")

    # --- Nationality ---
    freq_nat = fdf["الجنسية Nationality"].value_counts().reset_index()
    freq_nat.columns = ["Nationality", "Frequency"]
    freq_nat["Percentage"] = (freq_nat["Frequency"] / freq_nat["Frequency"].sum() * 100).round(2)
    st.dataframe(freq_nat)
    st.download_button("⬇️ Download Nationality Table", freq_nat.to_csv(index=False).encode("utf-8"), "nationality_distribution.csv", "text/csv")

    fig_nat = px.bar(freq_nat, x="Nationality", y="Frequency", title="Nationality Frequency Distribution", color="Frequency", color_continuous_scale="Blues")
    st.plotly_chart(fig_nat, use_container_width=True)
    st.info("**Interpretation:** The tallest bars represent the nationalities most frequently represented among pilgrims.")

    # --- Gender ---
    freq_gen = fdf["Gender_English"].value_counts().reset_index()
    freq_gen.columns = ["Gender", "Frequency"]
    freq_gen["Percentage"] = (freq_gen["Frequency"] / freq_gen["Frequency"].sum() * 100).round(2)
    st.dataframe(freq_gen)
    st.download_button("⬇️ Download Gender Table", freq_gen.to_csv(index=False).encode("utf-8"), "gender_distribution.csv", "text/csv")

    fig_gen = px.pie(freq_gen, names="Gender", values="Frequency", title="Gender Proportional Distribution", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_gen, use_container_width=True)
    st.info("**Interpretation:** Each slice represents the proportion of male vs female pilgrims in the dataset.")

    # --- Age ---
    freq_age = fdf["العمر Age"].value_counts().reset_index()
    freq_age.columns = ["Age", "Frequency"]
    freq_age["Percentage"] = (freq_age["Frequency"] / freq_age["Frequency"].sum() * 100).round(2)
    st.dataframe(freq_age)
    st.download_button("⬇️ Download Age Table", freq_age.to_csv(index=False).encode("utf-8"), "age_distribution.csv", "text/csv")

    fig_agefreq = px.bar(freq_age.sort_values("Age"), x="Age", y="Frequency", title="Age Frequency Distribution", color="Frequency", color_continuous_scale="Viridis")
    st.plotly_chart(fig_agefreq, use_container_width=True)
    st.info("**Interpretation:** This chart shows the number of pilgrims at each age interval, revealing concentration and range of age distribution.")

    # ---------------- LANGUAGE & GENDER ----------------
    if "اللغة Language" in fdf.columns:
        st.subheader("🌍 Gender–Language Interaction Analysis")

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

        gender_translation = {'أنثى': 'Female', 'ذكر': 'Male'}
        fdf['Gender_English'] = fdf['الجنس Gender'].map(gender_translation)

        lang_gender_ct = pd.crosstab(fdf['Gender_English'], fdf['اللغة Language'])
        bars = []
        for language in lang_gender_ct.columns:
            label = language_translation.get(language, language)
            bars.append(go.Bar(name=label, x=lang_gender_ct.index, y=lang_gender_ct[language]))

        fig_lang = go.Figure(data=bars)
        fig_lang.update_layout(barmode='stack', title='Distribution of Gender by Language', xaxis_title='Gender', yaxis_title='Count', template='plotly_white')
        st.plotly_chart(fig_lang, use_container_width=True)
        st.info("**Interpretation:** Stacked bars show how language preferences vary between male and female pilgrims.")

    # ---------------- NAVIGATION ----------------
    if st.button("Back to Home"):
        st.session_state.page = "home"


# ---------------- ROUTING ----------------
def main():
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    if st.session_state.page == "home":
        home()
    elif st.session_state.page == "dashboard":
        dashboard()
    elif st.session_state.page == "documentation":
        documentation.show()


if __name__ == "__main__":
    main()
