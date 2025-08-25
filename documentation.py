import streamlit as st

def show():
    #st.set_page_config(page_title="Documentation", layout="wide")

    # Navigation button to go back to Home Page
    if st.button("Go to Home Page"):
        st.session_state["page"] = "home"
        st.rerun()  # modern replacement for st.experimental_rerun()

    # Custom CSS styling
    css = """
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 18px; line-height: 1.6; color: #333; background-color: #f9f9f9; }
        h1, h2, h3 { color: #2c3e50; }
        blockquote { background-color: #fff8c4; padding: 10px; border-left: 5px solid #f1c40f; margin: 15px 0; }
        code { background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; }
        ul { margin-left: 20px; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Documentation content in Markdown
    documentation_md = """
# Introduction

## About the Documentation
This documentation provides insights about the application. It highlights valuable information about the application, purpose and audience, user manual, modules, libraries, and technology used, limitations of the app, and future improvements. It also details where to find codes for debugging and improvement.

# Content
## About the Application: Purpose
The application is designed to provide analysis of the sentiments, demographics and cross-demographic analysis of pilgrims visiting the Umrah and Hajj. The models work by classifying comments either positive or negative in their respective key service areas. In reality, Hajj and Umrah experience more than 30 million visitors who leave comments and feedback in their native languages; more than 27 languages across the globe. Analyzing the comments and feedback is a real challenge considering the characteristics of the big data: volume, Velocity, Variety, Veracity, and Value.  Accordingly, this AI-driven application remains valuable to authorities helping them to leverage on the comments towards addressing and sorting recurring concerns systematically. The application is significant in understanding the demographic nature of visitors significantly important for preparation, planning, management, and effecting hosting of Umrah and Hajj pilgrims.

## User Manuals
The user of the application being administrators and authorities in Hajj and Umrah, can use the application to either analyze sentiments, demographics, or cross-demographic characteristics of visitors. The following sections highlights how to use the features for analysis and text classification.

## A. Demographics and Cross-demographics Analysis
This feature can be accessed from the homepage (https://h7ain98dkm8yqrwiu5xqhr.streamlit.app/). Once on the home page, a user should scroll down to the Cross-Demographic and Demographic Analysis and Sentimental and Text Classification analysis section on the bottom left of the home page.

Double click on the Cross-Demographic and Demographic Analysis button to navigate to demographic and cross demographic analysis section. In this section, the user can get comprehensive and interactive insights into the demographics of Hajj and Umrah pilgrims. The key characteristics analyzed here include:
- Age Distribution: Interactive visualizations illustrating the range and concentration of pilgrims’ ages.
- Statistical Overview: Key metrics including minimum, maximum, mean, quartiles, and mode of pilgrim ages.
- Nationality & Gender Breakdown: Detailed analysis of visitor nationalities segmented by gender.
- Cross-Demographic Insights: Integrated visualizations combining age, gender, and nationality to highlight deeper demographic trends.

To leverage on this feature, scroll down on the page to a section where you can input data either as a file, URL, API, or textual data. Select your appropriate source and load data. A user should ensure that the data loaded has at least the following columns:
- 'الجنسية Nationality'
- 'الجنس Gender'
- 'العمر Age'

**Note:** Without these columns, the analysis will throw an error. Once data is loaded, the system analyzes the data and displays the visuals of the analysis. Users can filter data accordingly.

Expected visuals include:
- Line graphs showing descriptive and dispersion statistical distribution of age characteristics.
- An interactive Plotly linear age distribution.
- An interactive comparative bar graph of nationality by gender.
- An interactive bubble plot of mean ages of nationality and gender.
- An interactive histogram visualizing demographics by nationality, gender, and age.

There are no limits to the data a user can input since the system is designed to work with big data.

Once done and needing to access the sentimental and text classification feature, double click on the tab **Back Home** found at the left bottom of the dashboard.

## B. Sentimental and Text Classification Analysis
Once on the main page, scroll down to the bottom left of the page. Click on the button **Sentimental and Text Classification**.

This feature accepts data inputs as files or text comments. Future improvements will include URL and API support for real-time sentimental and text classification analysis. For sentiment and text classification analysis, a user can upload a file, type, or copy and paste comments appropriately. Once loaded, the data is analyzed and output displayed as a dataframe which can be downloaded for further analysis or documentation.

Once done, a user can input more data or navigate to the main page using the **Back Home** tab at the left top of the page.

# Tech Stack
PilgrimageAI is built using a modern and modular tech stack, enabling real-time data visualization, multilingual NLP processing, and a rich interactive user experience.

**Backend & App Framework:**
- Python: Core programming language for logic, data processing, and NLP.
- Streamlit: Rapid web app framework used to build interactive dashboards and forms.
- Streamlit Autorefresh: Enables real-time dashboard updates.

**Data Processing & Visualization:**
- Pandas
- Plotly Express
- Seaborn & Matplotlib

**Natural Language Processing (NLP):**
- Transformers (Hugging Face)
- GoogleTranslator (Deep Translator)
- Custom Text Classification Pipeline

**File & API Handling:**
- pdfplumber
- Requests
- StringIO

**UI/UX Enhancements:**
- Custom HTML & CSS
- Responsive Layouts

**Runtime Environment:**
- Streamlit Watchdog Disabled for Linux inotify issues
- OpenAI & ChatGPT for code generation and debugging

# Code Structure
For full code structure and debugging, visit the notebook and app.py on the public GitHub repo.

# Limitations and Future Improvements
- Real-time comment and feedback sentimental and text classification feature.
- Reduce run time and computational cost of analyzing big data.
"""

    st.markdown(documentation_md)

if __name__ == '__main__':
    show()





