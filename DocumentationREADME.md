# Introduction 

## About the Documentation 

This documentation provides insights about the application. It highlights valuable information about the application, purpose and audience, user manual, modules, libraries, and technology used, limitations of the app, and future improvements. It also details where to find codes for debugging and improvement. 

# Content 
## About the Application: Purpose 

The application is designed to provide analysis of the sentiments, demographics and cross-demographic analysis of pilgrims visiting the Umrah and Hajj. The models work by classifying comments either positive or negative in their respective key service areas. In reality, Hajj and Umrah experience more than 30 million visitors who leave comments and feedback in their native languages; more than 27 languages across the globe. Analyzing the comments and feedback is a real challenge considering the characteristics of the big data: volume, Velocity, Variety, Veracity, and Value.  Accordingly, this AI-driven application remains valuable to authorities helping them to leverage on the comments towards addressing and sorting recurring concerns systematically. The application is significant in understanding the demographic nature of visitors significantly important for preparation, planning, management, and effecting hosting of Umrah and Hajj pilgrims. 

## User Manuals 

The user of the application being administrators and authorities in Hajj and Umrah, can use the application to either analyze sentiments, demographics, or cross-demographic characteristics of visitors. The following sections highlights how to use the features for analysis and text classification. 

## A. Demographics and Cross-demographics Analysis 

This feature can be accessed from the homepage (https://e6i65vsyfxwc78pybgzg7a.streamlit.app/). Once on the home page, a user should scroll down to the Cross-Demographic and Demographic Analysis and Sentimental and Text Classification analysis (See the figure below) section on the bottom left of the home page.

<img width="494" height="154" alt="backhome" src="https://github.com/user-attachments/assets/28306de5-3e02-4374-b1c0-dc3ac930e95e" />


Double click on the Cross-Demographic and Demographic Analysis button to navigate to demographic and cross demographic analysis section.      
In this section, the user can get comprehensive and interactive insights into the demographics of Hajj and Umrah pilgrims. The key characteristics analyzed here in include: 

        •	Age Distribution: Interactive visualizations illustrating the range and concentration of pilgrims’ ages.    
        •	Statistical Overview: Key metrics including minimum, maximum, mean, quartiles, and mode of pilgrim ages.    
        •	Nationality & Gender Breakdown: Detailed analysis of visitor nationalities segmented by gender.    
        •	Cross-Demographic Insights: Integrated visualizations combining age, gender, and nationality to highlight deeper demographic trends.  
        
To leverage on this feature, scroll down on the page to a section where can input data either as a file, url, API, or textual data. The section can be found on the middle left part of the page. A user should come across the following section:       

Load Data Appropriately by Select the right Data Source     
 
Upload File
 
Enter API URL
 
Paste Raw Text

<img width="334" height="99" alt="load" src="https://github.com/user-attachments/assets/4ed69594-8195-4cde-acca-58f23c837962" />


Select your appropriate source and load data. A user should ensure that the data loaded has the following at least the following columns:      

a.	'الجنسية Nationality',     
b.	'الجنس Gender',     
c.	 'العمر Age'      

Note: Without these columns, the analysis will throw an error.      
 Once data is loaded, the system analyzes the data and displays the visuals of the analysis.     
Once data is loaded, a user has the ability to filter data accordingly.     

<img width="1504" height="335" alt="filterpic" src="https://github.com/user-attachments/assets/92eeee5d-1e95-4edc-bee4-5087a36de981" />


Users should expect to see:  

a.	A line graphs showing descriptive and dispersion statistical distribution of age characteristics      
b.	An interactive plotly linear age distribution      
c.	An interactive Comparative bar graph of nationality by gender      
d.	An interactive bubble plot of mean ages of nationality and gender     
e.	An interactive histogram visualizing demographics by nationality, gender, and age     

There are no limits to the data a user can input since the system is designed to work with Big data     
Once done and need to access the sentimental and text classification feature, double click on the tab Back Home found at the left bottom of the dashboard board.     

<img width="494" height="154" alt="backhome" src="https://github.com/user-attachments/assets/9fca9364-3147-482b-bda9-33a60c3675ff" />



## A. Sentimental and Text Classification Analysis    

Once on the main page, https://sentimentalanalysispilgrim-mwaaysgfzdssubzst7dba2.streamlit.app/, scroll down to the bottom left of the page. Click on the button sentimental and text classification.    
  




<img width="653" height="145" alt="sentpic" src="https://github.com/user-attachments/assets/b426ca94-6685-4c8b-b98e-d113c9698408" />


  
This feature accepts data inputs as files or text comments. Future improvements will include url and API to allow for real-time sentimental and text classification analysis. For sentiment and text classification analysis, a user can upload a file, type, or copy and paste comments appropriately. Once loaded the data is analyzed and output displayed as dataframe which can be downloaded for further analysis or documentation.  

The expected output is:     

<img width="1735" height="669" alt="outputpic" src="https://github.com/user-attachments/assets/dde66590-2318-437c-8fdf-7431ba684b43" />


Once done, a user can input more data or navigate to the main page using the Back Home tab at the left top of the page.     


# Tech Stack

PilgrimageAI is built using a modern and modular tech stack, enabling real-time data visualization, multilingual NLP processing, and a rich interactive user experience. Here's a breakdown of the technologies used:     
Backend & App Framework     
•	Python: Core programming language for logic, data processing, and NLP.     
•	Streamlit: Rapid web app framework used to build interactive dashboards and forms.      
•	Streamlit Autorefresh: Enables real-time dashboard updates by auto-refreshing at set intervals.     

# Data Processing & Visualization    
•	Pandas: Efficient data handling and analysis of tabular feedback data.    
•	Plotly Express: Used for building interactive visualizations (e.g., line charts, histograms, scatter plots).    
•	Seaborn & Matplotlib: Statistical plotting libraries used for advanced visual analytics like age distribution with statistical markers.   

#### Natural Language Processing (NLP)
•	Transformers (Hugging Face): For sentiment analysis using pretrained models like distilbert-base-uncased-finetuned-sst-2-english.      
•	GoogleTranslator (Deep Translator): Translates multilingual user feedback into English before processing.     
•	Custom Text Classification Pipeline: Uses keyword token matching to categorize feedback into themes (e.g., transport, accommodation, staff behavior).  

#### File & API Handling    
•	pdfplumber: Extracts text from PDF feedback files.    
•	Requests: Fetches data from external APIs.    
•	StringIO: Converts text and raw CSV input into readable data frames.    

#### UI/UX Enhancements    
•	Custom HTML & CSS: Embedded styles add rich visual design (e.g., background images, overlays, styled buttons).    
•	Responsive Layouts: Uses st.columns, container sizing, and adaptive rendering to ensure the app works well across devices.     

### Runtime Environment
•	Streamlit Watchdog Disabled: To avoid inotify limit errors on Linux systems, file system watchers are turned off with:      
         os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
         os.environ["STREAMLIT_WATCHDOG_MODE"] = "none"
OpenAI and Chatgpt:  to improve generated codes and debugging the app end-to-end      

#### Code Structure 

For Full code structure and debugging visit the notebook and app.py on github public repo     
 
# Limitations and Future Improvements     

-	A real-time comment and feedback sentimental and text classification feature    
-	Reduce run time and computational cost of analyzing big data    



