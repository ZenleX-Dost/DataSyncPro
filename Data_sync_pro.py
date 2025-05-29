import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64
import os
import json
import pyarrow.parquet as pq
import time
from io import StringIO, BytesIO
from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import KNNImputer
from bs4 import BeautifulSoup
import requests
import asyncio
from playwright.async_api import async_playwright
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re
import extruct
import zipfile
import validators
import bleach
import kaleido
import mimetypes

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from collections import Counter
import string

import logging
logging.basicConfig(level=logging.INFO)

# --- Set Page Config with Dark Theme ---
st.set_page_config(
    page_title="DataSync Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        return True
    except:
        return False

# Initialize NLTK data
download_nltk_data()

# --- Initialize Session State ---
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'scraped_dfs' not in st.session_state:
    st.session_state['scraped_dfs'] = []
if 'color_palette' not in st.session_state:
    st.session_state['color_palette'] = 'Viridis'
if 'categorical_palette' not in st.session_state:
    st.session_state['categorical_palette'] = 'Set2'
if 'active_mode' not in st.session_state:
    st.session_state['active_mode'] = 'Data Upload & Merge'
if 'nlp_results' not in st.session_state:
    st.session_state['nlp_results'] = {}



# --- Dark Theme CSS Override ---
st.markdown("""
    <style>
    /* Base dark theme */
    .main {
        background-color: #1e1e1e;
        color: #e0e0e0;
        font-family: 'Arial', sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #1e1e1e;
    }
    [data-testid="stSidebar"] {
        background-color: #2a2a2a;
        color: #e0e0e0;
    }
    /* Header styles */
    h1, h2, h3, h4 {
        color: #ffffff;
        font-weight: 600;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: #ffffff;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb, #1e40af);
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        transform: translateY(-2px);
    }
    /* Radio buttons in sidebar */
    .stRadio [data-baseweb="radio"] {
        background-color: #3a3a3a;
        color: #e0e0e0;
        border-radius: 6px;
        padding: 10px;
        transition: all 0.3s ease;
        font-weight: 500;
        border: 1px solid #4a4a4a;
    }
    .stRadio [data-baseweb="radio"] div[aria-checked="true"] {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: #ffffff;
        border: none;
    }
    /* Selectboxes and Multiselects */
    .stSelectbox div[data-baseweb="select"]>div:focus, .stMultiSelect div[data-baseweb="select"]>div:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 4px rgba(59,130,246,0.5);
    }
    /* Sidebar Elements */
    .sidebar .stSelectbox, .sidebar .stMultiSelect, .sidebar .stTextInput {
        margin-bottom: 15px;
    }
    .sidebar .stButton>button {
        width: 100%;
    }
    /* Expanders */
    .stExpander {
        background-color: #2a2a2a;
        border-radius: 8px;
        border: 1px solid #4a4a4a;
    }
    .stExpander summary {
        background-color: #3a3a3a;
        color: #e0e0e0;
        font-weight: 500;
    }
    .stExpander summary:hover {
        background-color: #3b82f6;
        color: #ffffff;
    }
    /* Links */
    a {
        color: #3b82f6;
    }
    a:hover {
        color: #2563eb;
    }
    /* Responsive Design */
    @media (max-width: 600px) {
        .stRadio {
            flex-direction: column;
            gap: 8px;
        }
        .stDataFrame {
            font-size: 0.9rem;
        }
        .stButton>button {
            padding: 8px 12px;
            font-size: 12px;
        }
    }
    /* Override Streamlit Default Light Theme Elements */
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }
    [data-testid="stTable"] {
        background-color: #2a2a2a;
        color: #e0e0e0;
    }
    /* Ensure dark theme for all Streamlit components */
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        background-color: #3a3a3a;
        color: #e0e0e0;
        border: 1px solid #4a4a4a;
    }
    </style>
""", unsafe_allow_html=True)

# --- Utility Functions ---
# (All utility functions remain unchanged: fix_column_names, read_file, validate_scraped_data, 
# scrape_with_playwright, scrape_with_selenium, parse_html, clean_data, feature_engineering, 
# generate_statistics, create_interactive_statistics, create_visualizations, get_base64_download_link)
# [Assuming these functions are included here as in the original code for brevity]
def create_interactive_statistics(df):
    """Generate interactive statistical visualizations."""
    if df is None or df.empty:
        return {}
    
    stat_visualizations = {}
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    color_palette = st.session_state.get('color_palette', 'Viridis')
    plotly_colors = px.colors.sequential.__dict__.get(color_palette, px.colors.sequential.Viridis)

    # Missing Values
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        fig = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            title="Missing Values by Column",
            labels={'x': 'Columns', 'y': 'Missing Count'},
            color=missing_data.values,
            color_continuous_scale=plotly_colors
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False, color='white'),
            yaxis=dict(showgrid=False, color='white'),
            xaxis_tickangle=45
        )
        stat_visualizations['Missing Values'] = fig

    # Data Types
    dtype_counts = df.dtypes.astype(str).value_counts()
    if not dtype_counts.empty:
        fig = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index,
            title="Data Types Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        stat_visualizations['Data Types'] = fig

    # Numeric Statistics
    if not numeric_cols.empty:
        stats_data = []
        for col in numeric_cols[:8]:
            if df[col].nunique() > 1:
                stats_data.append({
                    'Column': col,
                    'Mean': df[col].mean(),
                    'Median': df[col].median(),
                    'Std': df[col].std(),
                    'Skewness': df[col].skew(),
                    'Kurtosis': df[col].kurtosis()
                })
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            fig = px.bar(
                stats_df,
                x='Column',
                y=['Mean', 'Median'],
                title="Numeric Columns: Central Tendency",
                barmode='group',
                color_discrete_sequence=['#3b82f6', '#ef4444']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, color='white'),
                yaxis=dict(showgrid=False, color='white'),
                xaxis_tickangle=45
            )
            stat_visualizations['Central Tendency'] = fig

            fig2 = px.scatter(
                stats_df,
                x='Skewness',
                y='Kurtosis',
                text='Column',
                title="Skewness vs Kurtosis",
                color='Std',
                size='Std',
                color_continuous_scale=plotly_colors
            )
            fig2.update_traces(textposition='top center')
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, color='white'),
                yaxis=dict(showgrid=False, color='white')
            )
            stat_visualizations['Distribution Shape'] = fig2

    # Categorical Overview
    if not categorical_cols.empty:
        cat_stats = []
        for col in categorical_cols[:6]:
            if df[col].nunique() > 0:
                cat_stats.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Most Frequent': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                    'Missing %': df[col].isnull().mean() * 100
                })
        if cat_stats:
            cat_df = pd.DataFrame(cat_stats)
            fig = px.bar(
                cat_df,
                x='Column',
                y='Unique Values',
                title="Categorical Columns: Unique Value Counts",
                color='Missing %',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, color='white'),
                yaxis=dict(showgrid=False, color='white'),
                xaxis_tickangle=45
            )
            stat_visualizations['Categorical Overview'] = fig

    # Dataset Overview
    summary_data = {
        'Metric': ['Total Rows', 'Total Columns', 'Numeric Columns', 'Categorical Columns', 'Missing Values', 'Memory Usage (MB)'],
        'Value': [
            df.shape[0],
            df.shape[1],
            len(numeric_cols),
            len(categorical_cols),
            df.isnull().sum().sum(),
            round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        ]
    }
    fig = px.bar(
        summary_data,
        x='Metric',
        y='Value',
        title="Dataset Overview",
        color='Value',
        color_continuous_scale=plotly_colors
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, color='white'),
        yaxis=dict(showgrid=False, color='white'),
        xaxis_tickangle=45
    )
    stat_visualizations['Dataset Overview'] = fig

    return stat_visualizations
def fix_column_names(df):
    """Fix unnamed and problematic column names"""
    if df is None or df.empty:
        return df
    
    df = df.copy()
    new_columns = []
    
    for i, col in enumerate(df.columns):
        col_str = str(col).strip()
        
        # Handle unnamed columns
        if (col_str.lower().startswith('unnamed') or 
            col_str == '' or 
            col_str.isdigit() or 
            pd.isna(col)):
            
            # Try to infer column type and create meaningful name
            sample_data = df.iloc[:, i].dropna().head(10)
            if not sample_data.empty:
                if pd.api.types.is_numeric_dtype(sample_data):
                    new_columns.append(f'numeric_column_{i+1}')
                elif pd.api.types.is_datetime64_any_dtype(sample_data):
                    new_columns.append(f'date_column_{i+1}')
                else:
                    new_columns.append(f'text_column_{i+1}')
            else:
                new_columns.append(f'column_{i+1}')
        else:
            # Clean existing column names
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', col_str.lower())
            clean_name = re.sub(r'_+', '_', clean_name).strip('_')
            if not clean_name:
                clean_name = f'column_{i+1}'
            new_columns.append(clean_name)
    
    df.columns = new_columns
    return df

@st.cache_data(show_spinner=False)
def read_file(file):
    file_ext = os.path.splitext(file.name)[1].lower()
    allowed_types = {
        '.csv': 'text/csv',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.xls': 'application/vnd.ms-excel',
        '.json': 'application/json',
        '.parquet': 'application/octet-stream',
        '.txt': 'text/plain',
        '.avif': 'image/avif'
    }
    mime_type, _ = mimetypes.guess_type(file.name)
    if file_ext not in allowed_types or (mime_type and mime_type not in allowed_types.values()):
        st.error(f"Unsupported file type: {file_ext}. Supported types: {', '.join(allowed_types.keys())}")
        return None
    try:
        file.seek(0)
        if file_ext == '.csv':
            encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252', 'utf-16']
            for encoding in encodings:
                try:
                    file.seek(0)
                    content = file.read().decode(encoding)
                    delimiters = [',', ';', '\t', '|']
                    counts = {d: content.count(d) for d in delimiters}
                    best_delimiter = max(counts, key=counts.get, default=',')
                    df = pd.read_csv(StringIO(content), sep=best_delimiter, engine='python', on_bad_lines='skip')
                    if df.shape[1] > 0:
                        df = fix_column_names(df)  # Fix column names
                        st.success(f"Successfully read CSV with {encoding} encoding and '{best_delimiter}' delimiter.")
                        return df
                    st.warning(f"CSV file parsed with {encoding} but no valid columns detected.")
                except Exception as e:
                    st.warning(f"Failed to read CSV with {encoding}: {str(e)}")
            st.error("Failed to parse CSV file. Tried multiple encodings and delimiters.")
            return None
        elif file_ext in ['.xlsx', '.xls']:
            try:
                df = pd.read_excel(file, engine='openpyxl' if file_ext == '.xlsx' else 'xlrd')
                if df.shape[1] > 0:
                    df = fix_column_names(df)  # Fix column names
                    st.success("Successfully read Excel file.")
                    return df
                st.error("Excel file is empty or invalid.")
                return None
            except Exception as e:
                st.error(f"Excel parsing error: {str(e)}")
                return None
        elif file_ext == '.json':
            try:
                df = pd.read_json(file, orient='records', lines=True)
                if df.shape[1] > 0:
                    df = fix_column_names(df)  # Fix column names
                    st.success("Successfully read JSON file (records orient).")
                    return df
                st.warning("JSON file is empty or invalid with records orient. Trying alternative parsing...")
                file.seek(0)
                try:
                    df = pd.read_json(file, orient='table')
                    if df.shape[1] > 0:
                        df = fix_column_names(df)
                        st.success("Successfully read JSON file (table orient).")
                        return df
                except ValueError:
                    file.seek(0)
                    df = pd.json_normalize(json.load(file))
                    if df.shape[1] > 0:
                        df = fix_column_names(df)
                        st.success("Successfully read JSON file (normalized).")
                        return df
                st.error("JSON file is empty or invalid after trying multiple parsing methods.")
                return None
            except Exception as e:
                st.error(f"JSON parsing error: {str(e)}")
                return None
        elif file_ext == '.parquet':
            try:
                df = pq.read_table(file).to_pandas()
                if df.shape[1] > 0:
                    df = fix_column_names(df)  # Fix column names
                    st.success("Successfully read Parquet file.")
                    return df
                st.error("Parquet file is empty or invalid.")
                return None
            except Exception as e:
                st.error(f"Parquet parsing error: {str(e)}")
                return None
        elif file_ext == '.txt':
            for delimiter in ['\t', ',', ';', '|', ' ']:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, sep=delimiter, engine='python', on_bad_lines='skip')
                    if df.shape[1] > 1:
                        df = fix_column_names(df)  # Fix column names
                        st.success(f"Successfully read TXT with '{delimiter}' delimiter.")
                        return df
                except:
                    continue
            file.seek(0)
            content = file.read().decode('utf-8', errors='replace')
            df = pd.DataFrame({"text_content": content.splitlines()})
            if df.shape[0] > 0:
                st.success("Successfully read TXT as single column text.")
                return df
            st.error("Text file is empty or invalid.")
            return None
        elif file_ext == '.avif':
            try:
                file.seek(0)
                img = Image.open(file)
                metadata = img.info
                if metadata:
                    df = pd.DataFrame([metadata])
                    st.success("Successfully extracted metadata from AVIF file.")
                    return df
                st.warning("No metadata extracted from AVIF file.")
                return pd.DataFrame()
            except Exception as e:
                st.error(f"Could not extract metadata from AVIF file: {str(e)}")
                return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# --- Web Scraping Functions ---
def validate_scraped_data(data_frames, url):
    validation_results = []
    for i, df in enumerate(data_frames):
        result = {'index': i+1, 'is_valid': True, 'issues': [], 'confidence': 0.9}
        if df.empty or df.shape[0] < 2:
            result['is_valid'] = False
            result['issues'].append("DataFrame is empty or has too few rows.")
            result['confidence'] = 0.2
        else:
            total_cells = df.shape[0] * df.shape[1]
            if total_cells > 0:
                missing_ratio = df.isnull().sum().sum() / total_cells
                if missing_ratio > 0.5:
                    result['issues'].append(f"High missing value ratio: {missing_ratio:.2%}")
                    result['confidence'] = max(0.1, result['confidence'] - 0.3)
            unnamed_cols = [col for col in df.columns if 'unnamed' in str(col).lower()]
            if unnamed_cols:
                result['issues'].append(f"Contains 'Unnamed' columns: {', '.join(map(str, unnamed_cols))}")
                result['confidence'] = max(0.1, result['confidence'] - 0.1)
            special_chars = df.apply(lambda x: x.astype(str).str.contains(r'[^\w\s]', regex=True).sum() if x.dtype == "object" else 0).sum()
            if special_chars > df.shape[0] * 0.5:
                result['issues'].append("High proportion of special characters in text data.")
                result['confidence'] = max(0.1, result['confidence'] - 0.2)
        if result['issues']:
            result['is_valid'] = False
            result['suggestions'] = [
                "Try different HTML tags (e.g., div, span, a, table).",
                "Specify more relevant class names or IDs.",
                "Check if JavaScript rendering is required.",
                "Consider refining the URL or element selectors.",
                "Verify if the site uses JSON-LD, microdata, or OpenGraph."
            ]
        validation_results.append(result)
    return validation_results

async def scrape_with_playwright(url, tags=None, classes=None, ids=None):
    data_frames = []
    messages = []
    html_content = None
    retries = 3
    for attempt in range(retries):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, wait_until='networkidle', timeout=120000)  # Increased timeout
                html_content = await page.content()
                messages.append(f"Attempt {attempt+1}: Successfully loaded webpage content using Playwright.")
                await browser.close()
                break
        except Exception as e:
            messages.append(f"Attempt {attempt+1} failed with Playwright: {str(e)}")
            if attempt == retries - 1:
                messages.append("Falling back to requests...")
                try:
                    time.sleep(1)  # Rate limiting delay
                    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
                    response.raise_for_status()
                    html_content = response.text
                    messages.append("Retrieved static content with requests.")
                except Exception as e_req:
                    return [], f"Failed to retrieve content after {retries} attempts: {str(e_req)}"
    if not html_content:
        return [], "Failed to retrieve any content from the URL."
    return parse_html(html_content, url, tags, classes, ids, messages)

def scrape_with_selenium(url, tags=None, classes=None, ids=None):
    data_frames = []
    messages = []
    html_content = None
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0")
    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        time.sleep(1)  # Rate limiting delay
        driver.get(url)
        time.sleep(5)
        html_content = driver.page_source
        messages.append("Successfully loaded webpage with Selenium.")
    except Exception as e:
        messages.append(f"Error with Selenium: {str(e)}. Falling back to requests...")
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
            response.raise_for_status()
            html_content = response.text
            messages.append("Retrieved static content with requests.")
        except Exception as e_req:
            return [], f"Failed to retrieve content: {str(e_req)}"
    finally:
        if driver:
            driver.quit()
    if not html_content:
        return [], "Failed to retrieve any content from the URL."
    return parse_html(html_content, url, tags, classes, ids, messages)

def parse_html(html_content, url, tags, classes, ids, messages):
    data_frames = []
    try:
        soup = BeautifulSoup(html_content, 'lxml')
    except ImportError:
        soup = BeautifulSoup(html_content, 'html.parser')
        messages.append("lxml not installed; using html.parser")
    
    try:
        tables = pd.read_html(StringIO(html_content), flavor='lxml', header=0)
        valid_tables = [table.copy() for table in tables if table.shape[0] > 1 and table.shape[1] > 1]
        if valid_tables:
            data_frames.extend(valid_tables)
            messages.append(f"Extracted {len(valid_tables)} HTML tables.")
    except Exception as e:
        messages.append(f"No tables found or error extracting tables: {str(e)}")
    
    lists = soup.find_all(['ul', 'ol'])
    for i, lst in enumerate(lists):
        items = [li.get_text(strip=True) for li in lst.find_all('li') if li.get_text(strip=True)]
        if items:
            df = pd.DataFrame({'item': items, 'list_type': lst.name, 'list_index': i+1})
            data_frames.append(df.copy())
            messages.append(f"Extracted list {i+1} with {len(items)} items.")
    
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4'])
    header_data = []
    for header in headers:
        text = header.get_text(strip=True)
        if text:
            content = []
            for sibling in header.find_next_siblings():
                if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                    break
                if sibling.get_text(strip=True):
                    content.append(sibling.get_text(strip=True))
            header_data.append({
                'header': text,
                'level': header.name,
                'content_snippet': ' '.join(content)[:500].strip()
            })
    if header_data:
        df = pd.DataFrame(header_data)
        data_frames.append(df.copy())
        messages.append(f"Extracted {len(header_data)} headers with content.")
    
    paragraphs = soup.find_all('p')
    para_data = [
        {'content': bleach.clean(p.get_text(strip=True), tags=[], strip=True), 'type': 'paragraph'}
        for p in paragraphs if len(p.get_text(strip=True).strip()) > 15
    ]
    if para_data:
        df = pd.DataFrame(para_data)
        data_frames.append(df.copy())
        messages.append(f"Extracted {len(para_data)} paragraphs.")
    
    search_tags = tags if tags else ['div', 'span', 'a', 'article', 'section', 'table']
    for tag in soup.find_all(search_tags):
        if classes and any(cls in tag.get('class', []) for cls in classes):
            content = bleach.clean(tag.get_text(strip=True).strip().replace('\n', ' ')[:500], tags=[], strip=True)
            if content:
                data_frames.append(pd.DataFrame([{
                    'content': content,
                    'tag': tag.name,
                    'classes': ' '.join(tag.get('class', []))
                }]))
                messages.append(f"Extracted element with class: {tag.get('class', [])}")
        elif ids and tag.get('id') in ids:
            content = bleach.clean(tag.get_text(strip=True).strip().replace('\n', ' ')[:500], tags=[], strip=True)
            if content:
                data_frames.append(pd.DataFrame([{
                    'content': content,
                    'tag': tag.name,
                    'id': tag.get('id')
                }]))
                messages.append(f"Extracted element with ID: {tag.get('id')}")
        elif not classes and not ids:
            content = bleach.clean(tag.get_text(strip=True).strip().replace('\n', ' ')[:500], tags=[], strip=True)
            if content:
                data_frames.append(pd.DataFrame([{
                    'content': content,
                    'tag': tag.name,
                    'classes': ' '.join(tag.get('class', []))
                }]))
                messages.append(f"Extracted element with tag: {tag.name}")
    
    try:
        extracted = extruct.extract(html_content, base_url=url, syntaxes=['json-ld', 'microdata', 'opengraph'])
        jsonld_data = extracted.get('json-ld', [])
        microdata = extracted.get('microdata', [])
        opengraph = extracted.get('opengraph', [])
        if jsonld_data:
            df = pd.json_normalize(jsonld_data)
            data_frames.append(df.copy())
            messages.append(f"Extracted {len(jsonld_data)} JSON-LD items.")
        if microdata:
            df = pd.json_normalize(microdata)
            data_frames.append(df.copy())
            messages.append(f"Extracted {len(microdata)} microdata items.")
        if opengraph:
            df = pd.json_normalize(opengraph)
            data_frames.append(df.copy())
            messages.append(f"Extracted {len(opengraph)} OpenGraph items.")
    except Exception as e:
        messages.append(f"Failed to extract structured data: {str(e)}")
    
    validation_results = validate_scraped_data(data_frames, url)
    for i, result in enumerate(validation_results):
        if not result['is_valid']:
            messages.append(f"DataFrame {result['index']} issues: {', '.join(result['issues'])}")
            if result.get('suggestions'):
                messages.append(f"Suggestions: {', '.join(result['suggestions'])}")
        messages.append(f"DataFrame {result['index']} confidence: {result['confidence']:.2%}")
    
    if not data_frames:
        df = pd.DataFrame([{'content_snippet': bleach.clean(html_content[:500].strip(), tags=[], strip=True), 'type': 'raw_html_sample'}])
        data_frames.append(df.copy())
        messages.append("No structured content found; returning raw HTML sample.")
    
    # Fix column names for all scraped dataframes
    for i, df_item in enumerate(data_frames):
        df_item = fix_column_names(df_item)
        data_frames[i] = df_item
    
    return data_frames, " | ".join(messages)

# --- Data Cleaning ---
@st.cache_data(show_spinner=False)
def clean_data(df_input, missing_method='median', outlier_method='keep', normalize_method=None, custom_fill=None):
    if df_input is None or df_input.empty:
        return None, ["No valid data to clean."]
    
    # Fix column names first
    df = fix_column_names(df_input.copy())
    cleaning_steps = ["✅ Standardized column names for better processing"]
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if not date_cols.empty:
        with st.spinner("Extracting date/time features..."):
            for col in date_cols:
                try:
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                    df[f'{col}_quarter'] = df[col].dt.quarter
                    df[f'{col}_weekofyear'] = df[col].dt.isocalendar().week.astype('int64')
                    cleaning_steps.append(f"Extracted year, month, day, dayofweek, quarter, weekofyear from '{col}'.")
                except Exception as e:
                    cleaning_steps.append(f"Failed to extract date features from '{col}': {e}")
    if df_input is not None and 'date' in df_input.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        cleaning_steps.append("✅ Converted 'date' column to datetime format")
    
    with st.spinner("Handling missing values..."):
        missing_before = df.isnull().sum().sum()
        if not numeric_cols.empty:
            if missing_method == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif missing_method == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif missing_method == 'knn':
                try:
                    imputer = KNNImputer(n_neighbors=5)
                    df[numeric_cols] = pd.DataFrame(imputer.fit_transform(df[numeric_cols]),
                                                    columns=numeric_cols, index=df.index)
                    cleaning_steps.append(f"Imputed numeric missing values using KNN (n=5).")
                except Exception as e:
                    cleaning_steps.append(f"Failed KNN imputer: {e}. Falling back to median.")
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif missing_method == 'interpolate':
                df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
                cleaning_steps.append(f"Interpolated numeric missing values linearly.")
            elif missing_method == 'custom' and custom_fill is not None:
                df[numeric_cols] = df[numeric_cols].fillna(custom_fill)
                cleaning_steps.append(f"Filled numeric missing values with custom value: {custom_fill}.")
        if not categorical_cols.empty:
            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
                    cleaning_steps.append(f"Filled categorical missing values in '{col}' with mode/default ('{mode_val}').")
        if missing_method == 'drop':
            rows_before_drop = df.shape[0]
            df = df.dropna(how='any').reset_index(drop=True)
            if rows_before_drop > df.shape[0]:
                cleaning_steps.append(f"Dropped {rows_before_drop - df.shape[0]} rows with missing values.")
        missing_after = df.isnull().sum().sum()
        if missing_before > missing_after:
            cleaning_steps.append(f"Reduced missing values from {missing_before} to {missing_after}.")
    if df.empty:
        return df, cleaning_steps + ["DataFrame empty after missing value handling."]
    with st.spinner("Removing duplicates..."):
        duplicates_before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        if duplicates_before > len(df):
            cleaning_steps.append(f"Removed {duplicates_before - len(df)} duplicate rows.")
    with st.spinner("Standardizing column names..."):
        original_cols = list(df.columns)
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col).strip().lower()) for col in df.columns]
        if original_cols != list(df.columns):
            cleaning_steps.append("Standardized column names (e.g., spaces to underscores, lowercase).")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    with st.spinner(f"Handling outliers using {outlier_method}..."):
        if outlier_method != 'keep' and not numeric_cols.empty:
            for col in numeric_cols:
                if df[col].isnull().all() or df[col].nunique() < 2:
                    continue
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                if outlier_method == 'cap':
                    outliers_capped = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outliers_capped > 0:
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                        cleaning_steps.append(f"Capped {outliers_capped} outliers in '{col}'.")
                elif outlier_method == 'remove':
                    rows_before_outlier_removal = df.shape[0]
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)].reset_index(drop=True)
                    if rows_before_outlier_removal > df.shape[0]:
                        cleaning_steps.append(f"Removed {rows_before_outlier_removal - df.shape[0]} rows due to outliers in '{col}'.")
            if outlier_method != 'keep' and "Outlier handling" not in "".join(cleaning_steps):
                cleaning_steps.append(f"Outlier handling using {outlier_method} completed (no changes needed).")
    if df.empty:
        return df, cleaning_steps + ["DataFrame empty after outlier handling."]
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if normalize_method and not numeric_cols.empty:
        with st.spinner(f"Normalizing numeric columns using {normalize_method}..."):
            try:
                valid_cols = [col for col in numeric_cols if df[col].nunique() > 1]
                if not valid_cols:
                    cleaning_steps.append("Skipped normalization: No numeric columns with variance.")
                else:
                    scaler = {'minmax': MinMaxScaler(), 'standard': StandardScaler(), 'robust': RobustScaler()}[normalize_method]
                    df[valid_cols] = pd.DataFrame(scaler.fit_transform(df[valid_cols]),
                                                columns=valid_cols, index=df.index)
                    cleaning_steps.append(f"Normalized {len(valid_cols)} numeric columns using {normalize_method} scaler.")
            except Exception as e:
                cleaning_steps.append(f"Failed to normalize numeric columns: {e}.")
    return df, cleaning_steps

# --- Feature Engineering ---
@st.cache_data(show_spinner=False)
def feature_engineering(df_input, poly_degree=2, bin_cols=None, encode_cols=None, pca_components=0):
    if df_input is None or df_input.empty:
        return None, ["No valid data for feature engineering."]
    df = df_input.copy()
    steps = []
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if not date_cols.empty:
        with st.spinner("Extracting date/time features..."):
            for col in date_cols:
                try:
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                    df[f'{col}_quarter'] = df[col].dt.quarter
                    df[f'{col}_weekofyear'] = df[col].dt.isocalendar().week.astype('int64')
                    steps.append(f"Extracted year, month, day, dayofweek, quarter, weekofyear from '{col}'.")
                except Exception as e:
                    steps.append(f"Failed to extract date features from '{col}': {e}")
    if bin_cols:
        with st.spinner("Applying binning..."):
            for col in bin_cols:
                if col in numeric_cols:
                    try:
                        df[f'{col}_binned'] = pd.qcut(df[col], q=4, labels=[f'{col}_Q1', f'{col}_Q2', f'{col}_Q3', f'{col}_Q4'], duplicates='drop')
                        steps.append(f"Binned '{col}' into 4 quantiles.")
                    except Exception as e:
                        steps.append(f"Failed to bin '{col}': {e}.")
                else:
                    steps.append(f"'{col}' is not numeric; skipping binning.")
    if encode_cols:
        with st.spinner("Applying encoding..."):
            encoding_type = st.session_state.get('encoding_type_radio', 'label')
            for col in encode_cols:
                if col in categorical_cols:
                    try:
                        if encoding_type == 'onehot':
                            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                            encoded_features = encoder.fit_transform(df[[col]])
                            encoded_col_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                            df_encoded = pd.DataFrame(encoded_features, columns=encoded_col_names, index=df.index)
                            df = pd.concat([df, df_encoded], axis=1)
                            df = df.drop(columns=[col])
                            steps.append(f"One-hot encoded '{col}'.")
                        else:
                            le = LabelEncoder()
                            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                            steps.append(f"Label encoded '{col}'.")
                    except Exception as e:
                        steps.append(f"Failed to encode '{col}': {e}")
                else:
                    steps.append(f"'{col}' is not categorical; skipping encoding.")
    if poly_degree > 1 and not numeric_cols.empty:
        with st.spinner("Adding polynomial features..."):
            try:
                cols_for_poly = [col for col in numeric_cols if df[col].nunique() > 1]
                if cols_for_poly:
                    # Limit polynomial features to prevent memory explosion
                    max_poly_features = 1000
                    from math import comb
                    est_features = sum(comb(len(cols_for_poly) + k - 1, k) for k in range(1, poly_degree + 1))
                    if est_features > max_poly_features:
                        steps.append(f"Skipped polynomial features: Estimated {est_features} features exceed limit of {max_poly_features}.")
                    else:
                        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                        poly_features = poly.fit_transform(df[cols_for_poly])
                        poly_cols = poly.get_feature_names_out(cols_for_poly)
                        df_poly = pd.DataFrame(poly_features, columns=poly_cols, index=df.index)
                        new_poly_columns = [col for col in poly_cols if col not in df.columns]
                        df = pd.concat([df, df_poly[new_poly_columns]], axis=1)
                        steps.append(f"Added polynomial features (degree={poly_degree}) for {len(cols_for_poly)} numeric columns.")
                else:
                    steps.append("Skipped polynomial features: No numeric columns with unique values.")
            except Exception as e:
                steps.append(f"Failed to add polynomial features: {e}")
    if pca_components > 0 and not numeric_cols.empty:
        with st.spinner("Applying PCA..."):
            try:
                data_for_pca = df[numeric_cols].fillna(0)
                if data_for_pca.shape[1] >= pca_components:
                    pca = PCA(n_components=pca_components)
                    pca_features = pca.fit_transform(data_for_pca)
                    pca_col_names = [f'pca_component_{i+1}' for i in range(pca_components)]
                    df[pca_col_names] = pd.DataFrame(pca_features, index=df.index)
                    steps.append(f"Added {pca_components} PCA components.")
                else:
                    steps.append(f"Skipped PCA: Not enough numeric columns ({data_for_pca.shape[1]}) for {pca_components} components.")
            except Exception as e:
                steps.append(f"Failed to apply PCA: {e}")
    return df, steps

# --- Statistics Generation ---
def generate_statistics(df):
    if df is None or df.empty:
        return pd.DataFrame()
    stats = df.describe(include='all').T
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if not numeric_cols.empty:
        stats.loc[numeric_cols, 'skewness'] = df[numeric_cols].skew()
        stats.loc[numeric_cols, 'kurtosis'] = df[numeric_cols].kurtosis()
        stats.loc[numeric_cols, 'variance'] = df[numeric_cols].var()
        stats.loc[numeric_cols, 'median'] = df[numeric_cols].median()
        stats.loc[numeric_cols, 'mode'] = df[numeric_cols].mode().iloc[0]
        stats.loc[numeric_cols, 'range'] = df[numeric_cols].max() - df[numeric_cols].min()
        stats.loc[numeric_cols, 'iqr'] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
        stats.loc[numeric_cols, 'missing_percent'] = df[numeric_cols].isnull().mean() * 100
    if not categorical_cols.empty:
        stats.loc[categorical_cols, 'unique_values'] = df[categorical_cols].nunique()
        stats.loc[categorical_cols, 'top_value'] = df[categorical_cols].mode().iloc[0]
        stats.loc[categorical_cols, 'missing_percent'] = df[categorical_cols].isnull().mean() * 100
    stats['dtype'] = df.dtypes
    return stats.fillna('-')

def create_interactive_statistics(df):
    """Create interactive statistical visualizations"""
    if df is None or df.empty:
        return {}
    
    stat_visualizations = {}
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    color_palette = st.session_state.get('color_palette', 'Viridis')
    plotly_colors = px.colors.sequential.__dict__.get(color_palette, px.colors.sequential.Viridis)
    
    # 1. Missing Values Heatmap
    try:
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            fig = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title="Missing Values by Column",
                labels={'x': 'Columns', 'y': 'Missing Count'},
                color=missing_data.values,
                color_continuous_scale=plotly_colors
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, color='white'),
                yaxis=dict(showgrid=False, color='white'),
                xaxis_tickangle=45
            )
            stat_visualizations['Missing Values'] = fig
    except Exception as e:
        st.warning(f"Could not generate missing values chart: {e}")
    
    # 2. Data Types Distribution
    try:
        dtype_counts = df.dtypes.astype(str).value_counts()
        fig = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index,
            title="Data Types Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        stat_visualizations['Data Types'] = fig
    except Exception as e:
        st.warning(f"Could not generate data types chart: {e}")
    
    # 3. Numeric Columns Statistics
    if not numeric_cols.empty:
        try:
            stats_data = []
            for col in numeric_cols[:8]:  # Limit to 8 columns for readability
                stats_data.append({
                    'Column': col,
                    'Mean': df[col].mean(),
                    'Median': df[col].median(),
                    'Std': df[col].std(),
                    'Skewness': df[col].skew(),
                    'Kurtosis': df[col].kurtosis()
                })
            
            stats_df = pd.DataFrame(stats_data)
            
            # Create subplots for different statistics
            fig = px.bar(
                stats_df,
                x='Column',
                y=['Mean', 'Median'],
                title="Numeric Columns: Central Tendency",
                barmode='group',
                color_discrete_sequence=['#3b82f6', '#ef4444']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, color='white'),
                yaxis=dict(showgrid=False, color='white'),
                xaxis_tickangle=45
            )
            stat_visualizations['Central Tendency'] = fig
            
            # Skewness and Kurtosis
            fig2 = px.scatter(
                stats_df,
                x='Skewness',
                y='Kurtosis',
                text='Column',
                title="Skewness vs Kurtosis",
                color='Std',
                size='Std',
                color_continuous_scale=plotly_colors
            )
            fig2.update_traces(textposition='top center')
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, color='white'),
                yaxis=dict(showgrid=False, color='white')
            )
            stat_visualizations['Distribution Shape'] = fig2
            
        except Exception as e:
            st.warning(f"Could not generate numeric statistics charts: {e}")
    
    # 4. Categorical Columns Statistics
    if not categorical_cols.empty:
        try:
            cat_stats = []
            for col in categorical_cols[:6]:  # Limit to 6 columns
                cat_stats.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Most Frequent': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                    'Missing %': df[col].isnull().mean() * 100
                })
            
            cat_df = pd.DataFrame(cat_stats)
            
            fig = px.bar(
                cat_df,
                x='Column',
                y='Unique Values',
                title="Categorical Columns: Unique Value Counts",
                color='Missing %',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, color='white'),
                yaxis=dict(showgrid=False, color='white'),
                xaxis_tickangle=45
            )
            stat_visualizations['Categorical Overview'] = fig
            
        except Exception as e:
            st.warning(f"Could not generate categorical statistics charts: {e}")
    
    # 5. Overall Dataset Summary
    try:
        summary_data = {
            'Metric': ['Total Rows', 'Total Columns', 'Numeric Columns', 'Categorical Columns', 'Missing Values', 'Memory Usage (MB)'],
            'Value': [
                df.shape[0],
                df.shape[1],
                len(numeric_cols),
                len(categorical_cols),
                df.isnull().sum().sum(),
                round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            ]
        }
        
        fig = px.bar(
            summary_data,
            x='Metric',
            y='Value',
            title="Dataset Overview",
            color='Value',
            color_continuous_scale=plotly_colors
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False, color='white'),
            yaxis=dict(showgrid=False, color='white'),
            xaxis_tickangle=45
        )
        stat_visualizations['Dataset Overview'] = fig
        
    except Exception as e:
        st.warning(f"Could not generate dataset overview chart: {e}")
    
    return stat_visualizations

# --- NLP Processing Functions ---
@st.cache_data(show_spinner=False)
def preprocess_text(text, remove_stopwords=True, lemmatize=True, remove_punctuation=True, lowercase=True):
    """Preprocess text data with various cleaning options"""
    if pd.isna(text) or text == '':
        return ''
    
    processed_text = str(text)
    
    # Convert to lowercase
    if lowercase:
        processed_text = processed_text.lower()
    
    # Remove punctuation
    if remove_punctuation:
        processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(processed_text)
    
    # Remove stopwords
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        except:
            pass
    
    # Lemmatization
    if lemmatize:
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        except:
            pass
    
    return ' '.join(tokens)

@st.cache_data(show_spinner=False)
def perform_sentiment_analysis(texts):
    """Perform sentiment analysis using TextBlob"""
    results = []
    for text in texts:
        if pd.isna(text) or text == '':
            results.append({'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'})
            continue
        
        try:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            results.append({
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment
            })
        except:
            results.append({'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'})
    
    return pd.DataFrame(results)

@st.cache_data(show_spinner=False)
def extract_named_entities(texts):
    """Extract named entities from text"""
    all_entities = []
    for text in texts:
        if pd.isna(text) or text == '':
            continue
        
        try:
            tokens = word_tokenize(str(text))
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)
            
            entities = []
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity = ' '.join([token for token, pos in chunk.leaves()])
                    entities.append({'entity': entity, 'label': chunk.label()})
            
            all_entities.extend(entities)
        except:
            continue
    
    return pd.DataFrame(all_entities) if all_entities else pd.DataFrame(columns=['entity', 'label'])

@st.cache_data(show_spinner=False)
def perform_topic_modeling(texts, n_topics=5, method='lda'):
    """Perform topic modeling using LDA or clustering"""
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts if pd.notna(text) and text.strip()]
    
    if len(processed_texts) < 3:
        return None, None, "Not enough text data for topic modeling"
    
    try:
        if method == 'lda':
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english', min_df=2)
            doc_term_matrix = vectorizer.fit_transform(processed_texts)
            
            # LDA Topic Modeling
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
            lda_topics = lda.fit_transform(doc_term_matrix)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
                topics.append({
                    'topic_id': topic_idx,
                    'top_words': ', '.join(top_words),
                    'weight': topic.max()
                })
            
            return pd.DataFrame(topics), lda_topics, "LDA topic modeling completed successfully"
        
        else:  # clustering
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english', min_df=2)
            vectors = vectorizer.fit_transform(processed_texts)
            
            kmeans = KMeans(n_clusters=n_topics, random_state=42)
            clusters = kmeans.fit_predict(vectors)
            
            # Get top terms per cluster
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for i in range(n_topics):
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-10:][::-1]
                top_words = [feature_names[idx] for idx in top_indices]
                
                topics.append({
                    'topic_id': i,
                    'top_words': ', '.join(top_words),
                    'weight': cluster_center.max()
                })
            
            return pd.DataFrame(topics), clusters, "K-means clustering completed successfully"
    
    except Exception as e:
        return None, None, f"Topic modeling failed: {str(e)}"

@st.cache_data(show_spinner=False)
def generate_word_frequency(texts, top_n=20):
    """Generate word frequency analysis"""
    all_words = []
    for text in texts:
        if pd.notna(text) and text.strip():
            processed = preprocess_text(text)
            words = processed.split()
            all_words.extend(words)
    
    if not all_words:
        return pd.DataFrame(columns=['word', 'frequency'])
    
    word_freq = Counter(all_words)
    top_words = word_freq.most_common(top_n)
    
    return pd.DataFrame(top_words, columns=['word', 'frequency'])

def create_visualizations(df, numeric_cols=None, categorical_cols=None, plot_types=None):
    """Create various data visualizations"""
    if df is None or df.empty:
        return {}
    
    visualizations = {}
    color_palette = st.session_state.get('color_palette', 'Viridis')
    plotly_colors = px.colors.sequential.__dict__.get(color_palette, px.colors.sequential.Viridis)
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if plot_types is None:
        plot_types = ['heatmap', 'histogram', 'bar']
    
    # Correlation Heatmap
    if 'heatmap' in plot_types and len(numeric_cols) > 1:
        try:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap",
                color_continuous_scale=plotly_colors
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            visualizations['heatmap'] = fig
        except Exception as e:
            st.warning(f"Could not generate correlation heatmap: {e}")
    
    # Histograms for numeric columns
    if 'histogram' in plot_types and numeric_cols:
        try:
            hist_plots = {}
            for col in numeric_cols[:6]:  # Limit to 6 columns
                if df[col].nunique() > 1:
                    fig = px.histogram(
                        df,
                        x=col,
                        title=f"Distribution of {col}",
                        nbins=30,
                        color_discrete_sequence=['#3b82f6']
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(showgrid=False, color='white'),
                        yaxis=dict(showgrid=False, color='white')
                    )
                    hist_plots[col] = fig
            if hist_plots:
                visualizations['histogram'] = hist_plots
        except Exception as e:
            st.warning(f"Could not generate histograms: {e}")
    
    # Bar plots for categorical columns
    if 'bar' in plot_types and categorical_cols:
        try:
            bar_plots = {}
            for col in categorical_cols[:4]:  # Limit to 4 columns
                if df[col].nunique() > 1 and df[col].nunique() <= 20:  # Avoid too many categories
                    value_counts = df[col].value_counts().head(10)
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Top Values in {col}",
                        color=value_counts.values,
                        color_continuous_scale=plotly_colors
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(showgrid=False, color='white'),
                        yaxis=dict(showgrid=False, color='white'),
                        xaxis_tickangle=45
                    )
                    bar_plots[col] = fig
            if bar_plots:
                visualizations['bar'] = bar_plots
        except Exception as e:
            st.warning(f"Could not generate bar plots: {e}")
    
    # Violin plots
    if 'violin' in plot_types and numeric_cols:
        try:
            violin_plots = {}
            for col in numeric_cols[:4]:  # Limit to 4 columns
                if df[col].nunique() > 10:
                    fig = px.violin(
                        df,
                        y=col,
                        title=f"Violin Plot of {col}",
                        color_discrete_sequence=['#3b82f6']
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(showgrid=False, color='white'),
                        yaxis=dict(showgrid=False, color='white')
                    )
                    violin_plots[col] = fig
            if violin_plots:
                visualizations['violin'] = violin_plots
        except Exception as e:
            st.warning(f"Could not generate violin plots: {e}")
    
    # 3D Scatter plot
    if '3d_scatter' in plot_types and len(numeric_cols) >= 3:
        try:
            fig = px.scatter_3d(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                z=numeric_cols[2],
                title=f"3D Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]} vs {numeric_cols[2]}",
                color_discrete_sequence=['#3b82f6']
            )
            fig.update_layout(
                scene=dict(
                    xaxis=dict(color='white'),
                    yaxis=dict(color='white'),
                    zaxis=dict(color='white')
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            visualizations['3d_scatter'] = fig
        except Exception as e:
            st.warning(f"Could not generate 3D scatter plot: {e}")
    
    # Time series plot (if date column exists)
    if 'time_series' in plot_types:
        try:
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if not date_cols.empty and numeric_cols:
                date_col = date_cols[0]
                numeric_col = numeric_cols[0]
                fig = px.line(
                    df,
                    x=date_col,
                    y=numeric_col,
                    title=f"Time Series: {numeric_col} over {date_col}",
                    color_discrete_sequence=['#3b82f6']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=False, color='white'),
                    yaxis=dict(showgrid=False, color='white')
                )
                visualizations['time_series'] = fig
        except Exception as e:
            st.warning(f"Could not generate time series plot: {e}")
    
    return visualizations

def create_nlp_visualizations(df, text_column):
    """Create NLP-specific visualizations"""
    visualizations = {}
    
    if text_column not in df.columns:
        return visualizations
    
    texts = df[text_column].dropna().astype(str)
    
    # Word Cloud
    try:
        all_text = ' '.join([preprocess_text(text) for text in texts])
        if all_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='black', 
                                colormap='viridis', max_words=100).generate(all_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout()
            visualizations['Word Cloud'] = fig
    except:
        pass
    
    # Sentiment Distribution
    try:
        sentiment_results = perform_sentiment_analysis(texts)
        if not sentiment_results.empty:
            sentiment_counts = sentiment_results['sentiment'].value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            visualizations['Sentiment Distribution'] = fig
    except:
        pass
    
    # Word Frequency
    try:
        word_freq = generate_word_frequency(texts, top_n=15)
        if not word_freq.empty:
            fig = px.bar(
                word_freq,
                x='frequency',
                y='word',
                orientation='h',
                title="Top 15 Most Frequent Words",
                color='frequency',
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False, color='white'),
                yaxis=dict(showgrid=False, color='white')
            )
            visualizations['Word Frequency'] = fig
    except:
        pass
    
    # Text Length Distribution
    try:
        text_lengths = [len(str(text).split()) for text in texts]
        fig = px.histogram(
            x=text_lengths,
            title="Text Length Distribution (Words)",
            nbins=20,
            color_discrete_sequence=['#3b82f6']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False, color='white', title='Number of Words'),
            yaxis=dict(showgrid=False, color='white', title='Frequency')
        )
        visualizations['Text Length Distribution'] = fig
    except:
        pass
    
    return visualizations

# --- Sidebar for Mode Selection and Configurations ---
with st.sidebar:
    st.header("DataSync Pro")
    
    # Mode Selection
    st.session_state['active_mode'] = st.radio(
        "Select Operation Mode",
        ["Data Upload & Merge", "Data Processing", "Web Scraping", "NLP Analysis"],
        key='mode_selector'
    )
    
    # Conditional Configuration Display
    if st.session_state['active_mode'] == "Data Upload & Merge":
        st.subheader("Upload Settings")
        st.info("File Upload Configuration")
        
        # File format help
        with st.expander("Supported File Formats"):
            st.write("""
            - **CSV**: Comma-separated values
            - **Excel**: .xlsx, .xls files
            - **JSON**: JavaScript Object Notation
            - **Parquet**: Columnar storage format
            - **TXT**: Plain text files
            - **AVIF**: Image metadata extraction
            """)
        
        # Merge settings
        if st.session_state['df'] is not None:
            st.subheader("Merge Settings")
            st.write(f"Current dataset: {st.session_state['df'].shape[0]} rows × {st.session_state['df'].shape[1]} cols")
    
    elif st.session_state['active_mode'] == 'Data Processing':
        st.header("Data Processing Settings")
        
        if st.session_state['df'] is not None and not st.session_state['df'].empty:
            df = st.session_state['df'].copy()
            st.info(f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} cols")
            
            # Data Processing Options
            st.subheader("Processing Options")
            
            missing_method = st.selectbox(
                "Handle Missing Values",
                ['median', 'mean', 'knn', 'interpolate', 'drop', 'custom'],
                key='missing_method',
                help="Select method to handle missing values"
            )
            if missing_method == 'custom':
                custom_fill = st.number_input(
                    "Custom Fill Value",
                    value=0.0,
                    key='custom_fill'
                )
            
            outlier_method = st.selectbox(
                "Handle Outliers",
                ['keep', 'cap', 'remove'],
                key='outlier_method',
                help="Select method to handle outliers"
            )
            
            normalize_method = st.selectbox(
                "Normalization Method",
                ['none', 'minmax', 'standard', 'robust'],
                key='normalize_method',
                help="Select normalization method"
            )
            
            poly_degree = st.slider(
                "Polynomial Features Degree",
                1, 3, 2,
                key='poly_degree',
                help="Degree for polynomial features"
            )
            pca_components = st.slider(
                "PCA Components",
                0, 10, 0,
                key='pca_components',
                help="Number of PCA components"
            )
            
            # Column Selection for Processing
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            bin_cols = st.multiselect(
                "Columns to Bin",
                numeric_cols,
                key='bin_cols',
                help="Select numeric columns for binning"
            )
            encode_cols = st.multiselect(
                "Columns to Encode",
                categorical_cols,
                key='encode_cols',
                help="Select categorical columns for encoding"
            )
            
            # Visualization Options
            st.subheader("Visualization Options")
            
            selected_numeric_cols = st.multiselect(
                "Numeric Columns for Visualization",
                numeric_cols,
                default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols,
                key='selected_numeric_cols'
            )
            
            selected_categorical_cols = st.multiselect(
                "Categorical Columns for Visualization",
                categorical_cols,
                default=categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols,
                key='selected_categorical_cols'
            )
                
            plot_types = st.multiselect(
                "Plot Types",
                ['heatmap', 'histogram', 'violin', '3d_scatter', 'time_series', 'bar'],
                default=['heatmap', 'bar'],
                key='plot_types',
                help="Select visualization types"
            )
        else:
            st.warning("No data available. Please upload data first.")
            
    elif st.session_state['active_mode'] == "Web Scraping":
        st.header("Web Scraping Settings")
        
        # Initialize session state for non-widget values
        if 'web_scrape_url' not in st.session_state:
            st.session_state['web_scrape_url'] = "https://example.com/"
        
        st.text_input(
            "Website URL",
            value=st.session_state.get('web_scrape_url', 'https://example.com/'),
            key='scrape_url_input',
            help="Enter a valid URL to scrape"
        )

        st.selectbox(
            "Scraper Engine",
            ['Playwright', 'Selenium'],
            key='scraper_type_input',
            help="Choose the scraping engine"
        )

        st.subheader("Target Elements")
        common_tags = ['title', 'div', 'span', 'a', 'table', 'p', 'ul', 'article', 'section', 'ol', 'li', 'h1', 'h2', 'h3', 'h4']
        st.multiselect(
            "HTML Tags",
            common_tags,
            default=['table', 'div'],
            key='scrape_tags_input',
            help="Select HTML tags to scrape"
        )

        st.text_input(
            "Custom Tags",
            placeholder="Enter custom tags, separated by commas",
            key='custom_tags_input',
            help="Enter additional tags"
        )

        st.text_input(
            "Classes",
                       placeholder="Enter CSS classes, separated by commas",
            key='scrape_classes_input',
            help="Enter CSS classes to target"
        )

        st.text_input(
            "IDs",
            placeholder="Enter element IDs, separated by commas",
            key='scrape_ids_input',
            help="Enter element IDs to target"
        )

        # Scraping status
        if st.session_state.get('scraped_dfs'):
            st.success(f"{len(st.session_state['scraped_dfs'])} datasets scraped")
            
    elif st.session_state['active_mode'] == "NLP Analysis":
        st.header("NLP Analysis Settings")
        
        if st.session_state['df'] is not None and not st.session_state['df'].empty:
            df = st.session_state['df']
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            if text_columns:
                st.selectbox(
                    "Select Text Column",
                    text_columns,
                    key='nlp_text_column',
                    help="Choose the column containing text data"
                )
                
                st.subheader("Text Preprocessing")
                st.checkbox("Remove Stopwords", value=True, key='nlp_remove_stopwords')
                st.checkbox("Lemmatization", value=True, key='nlp_lemmatize')
                st.checkbox("Remove Punctuation", value=True, key='nlp_remove_punct')
                st.checkbox("Convert to Lowercase", value=True, key='nlp_lowercase')
                
                st.subheader("Analysis Options")
                st.checkbox("Sentiment Analysis", value=True, key='nlp_sentiment')
                st.checkbox("Named Entity Recognition", value=True, key='nlp_ner')
                st.checkbox("Topic Modeling", value=True, key='nlp_topics')
                st.checkbox("Word Frequency Analysis", value=True, key='nlp_word_freq')
                
                if st.session_state.get('nlp_topics', False):
                    st.slider("Number of Topics", 2, 10, 5, key='nlp_n_topics')
                    st.selectbox("Topic Modeling Method", ['lda', 'clustering'], key='nlp_topic_method')
            else:
                st.warning("No text columns found in the dataset")
        else:
            st.info("Upload data first to enable NLP analysis")

# --- Download Helper ---
def get_base64_download_link(data, filename, text, mime_type):
    b64 = base64.b64encode(data).decode('utf-8')
    sanitized_text = bleach.clean(text, tags=[], attributes={}, strip=True)
    return f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{sanitized_text}</a>'

# --- Main App Structure ---
st.title("DataSync Pro")
st.markdown("*A powerful tool for data import, preprocessing, visualization, web scraping, and NLP analysis.*")

# --- Main Content Based on Selected Mode ---
if st.session_state['active_mode'] == "Data Upload & Merge":
    st.header("Upload Your Data")
    uploaded_files = st.file_uploader(
        "Upload Files (CSV, Excel, JSON, Parquet, TXT, AVIF)",
        type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'txt', 'avif'],
        accept_multiple_files=True,
        help="Supported formats: CSV, Excel (.xlsx, .xls), JSON, Parquet, TXT, AVIF (metadata extraction)",
        key='file_uploader'
    )
    
    st.subheader("Or Use Sample Data")
    use_sample = st.checkbox("Load Sample Data", value=False, key='use_sample')
    datasets = []
    
    # Load sample data if selected
    if use_sample:
        data = {
            'date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D')),
            'value_a': np.random.randint(0, 101, 100),
            'value_b': np.random.uniform(10, 1000, 100),
            'category': np.random.choice(['Alpha', 'Beta', 'Gamma'], 100),
            'rating': np.random.uniform(1, 5, 100),
            'mix_col_str_num': np.random.choice([10, 'abc', 20, 'def'], 100)
        }
        sample_df = pd.DataFrame(data)
        sample_df.loc[np.random.choice(sample_df.index, 10), 'value_a'] = np.nan
        sample_df.loc[np.random.choice(sample_df.index, 5), 'value_b'] = np.nan
        sample_df.loc[np.random.choice(sample_df.index, 5), 'category'] = np.nan
        datasets.append(sample_df)
        st.success("Sample data loaded successfully!")
    
    # Process uploaded files
    for file in uploaded_files:
        with st.spinner(f"Reading {file.name}..."):
            df_uploaded = read_file(file)
            if df_uploaded is not None and not df_uploaded.empty:
                datasets.append(df_uploaded)
    
    # Handle datasets
    if datasets:
        if len(datasets) == 1:
            st.session_state['df'] = datasets[0]
            st.subheader("Current Dataset Preview")
            st.info(f"Dataset: {datasets[0].shape[0]} rows, {datasets[0].shape[1]} columns")
            st.dataframe(datasets[0].head(), use_container_width=True)
        else:
            st.subheader("Merge Datasets")
            for i, df_item in enumerate(datasets):
                st.write(f"**Dataset {i+1}** ({df_item.shape[0]} rows, {df_item.shape[1]} columns)")
                st.dataframe(df_item.head(3), use_container_width=True)
            
            # Merge options
            col_options = sorted(list(set(col for df in datasets for col in df.columns)))
            merge_key = st.selectbox(
                "Select Common Column for Merging",
                col_options,
                key='merge_key',
                help="Choose a column present in all datasets to merge on"
            )
            merge_type = st.selectbox(
                "Merge Type",
                ['inner', 'outer', 'left', 'right'],
                help="inner: keep matching rows; outer: keep all rows; left: keep all left rows; right: keep all right rows",
                key='merge_type'
            )
            
            if st.button("Merge Datasets", key='merge_button'):
                try:
                    with st.spinner("Merging datasets..."):
                        merged_df = datasets[0].copy()
                        for i, df_to_merge in enumerate(datasets[1:], 1):
                            if merge_key not in merged_df.columns or merge_key not in df_to_merge.columns:
                                raise ValueError(f"Merge key '{merge_key}' not found in dataset {i+1}.")
                            merged_df = merged_df.merge(
                                df_to_merge,
                                on=merge_key,
                                how=merge_type,
                                suffixes=('', f'_{i+1}_merged')
                            )
                    st.session_state['df'] = merged_df
                    st.success("Datasets merged successfully!")
                    st.info(f"Merged Dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
                    st.dataframe(merged_df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error merging datasets: {str(e)}")
    else:
        st.info("No data loaded. Please upload a file or select sample data.")

elif st.session_state['active_mode'] == "Data Processing":
    st.header("Data Preprocessing & Analysis")
    if st.session_state['df'] is not None and not st.session_state['df'].empty:
        df = st.session_state['df']
        st.subheader("Current Data Preview")
        st.info(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Process Data & Generate Insights", key='process_button_main'):
            try:
                progress = st.progress(0)
                
                # Get processing options from sidebar
                missing_method = st.session_state.get('missing_method', 'median')
                outlier_method = st.session_state.get('outlier_method', 'keep')
                normalize_method = st.session_state.get('normalize_method', None)
                custom_fill = st.session_state.get('custom_fill', None)
                poly_degree = st.session_state.get('poly_degree', 2)
                pca_components = st.session_state.get('pca_components', 0)
                bin_cols = st.session_state.get('bin_cols', [])
                encode_cols = st.session_state.get('encode_cols', [])
                selected_numeric_cols = st.session_state.get('selected_numeric_cols', [])
                selected_categorical_cols = st.session_state.get('selected_categorical_cols', [])
                plot_types = st.session_state.get('plot_types', ['heatmap', 'bar'])
                
                # Data Cleaning
                with st.spinner("Cleaning data..."):
                    df_processed, cleaning_steps = clean_data(
                        df, missing_method, outlier_method, normalize_method, custom_fill
                    )
                    progress.progress(0.33)
                
                if df_processed is None or df_processed.empty:
                    st.error("Data cleaning resulted in an empty dataset. Adjust cleaning options.")
                    st.stop()
                
                # Feature Engineering
                with st.spinner("Performing feature engineering..."):
                    df_processed, fe_steps = feature_engineering(
                        df_processed, poly_degree, bin_cols, encode_cols, pca_components
                    )
                    progress.progress(0.66)
                
                if df_processed is None or df_processed.empty:
                    st.error("Feature engineering resulted in an empty dataset.")
                    st.stop()
                
                # Update session state
                st.session_state['df'] = df_processed
                st.success("Data processing complete!")
                progress.progress(1.0)
                
                # Processing Summary
                st.subheader("Processing Summary")
                for step in cleaning_steps + fe_steps:
                    st.write(f"- {step}")
                
                # Processed Data Preview
                st.subheader("Processed Data Preview")
                st.info(f"Dataset: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")
                st.dataframe(df_processed.head(), use_container_width=True)
                
                # Interactive Statistical Summary
                st.subheader("Interactive Statistical Analysis")
                with st.spinner("Generating interactive statistics..."):
                    stat_visualizations = create_interactive_statistics(df_processed)
                    
                    if stat_visualizations:
                        # Create tabs for different statistical views
                        stat_tabs = st.tabs(list(stat_visualizations.keys()))
                        
                        for i, (stat_name, fig) in enumerate(stat_visualizations.items()):
                            with stat_tabs[i]:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Download option for each chart
                                img_buffer = BytesIO()
                                try:
                                    fig.write_image(img_buffer, format='png', width=1200, height=800)
                                    st.download_button(
                                        f"Download {stat_name} Chart (PNG)",
                                        data=img_buffer.getvalue(),
                                        file_name=f"{stat_name.lower().replace(' ', '_')}_statistics.png",
                                        mime='image/png',
                                        key=f'download_stat_{stat_name}'
                                    )
                                except Exception as e:
                                    st.warning(f"Could not generate PNG for {stat_name}: {e}")
                    else:
                        st.warning("No statistical visualizations could be generated.")
                
                # Traditional Statistical Summary (as fallback)
                with st.expander("Traditional Statistical Summary", expanded=False):
                    stats = generate_statistics(df_processed)
                    if stats.empty:
                        st.warning("No statistics available.")
                    else:
                        st.dataframe(stats, use_container_width=True)
                
                # Visualizations
                st.subheader("Data Visualizations")
                visualizations = create_visualizations(
                    df_processed, selected_numeric_cols, selected_categorical_cols, plot_types
                )
                if not visualizations:
                    st.warning("No visualizations generated.")
                else:
                    for vis_type, figs in visualizations.items():
                        with st.expander(f"{vis_type.replace('_', ' ').title()}", expanded=True):
                            if isinstance(figs, dict):
                                for key, fig in figs.items():
                                    st.write(f"**{key}**")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Download option for Plotly figures
                                    img_buffer = BytesIO()
                                    try:
                                        fig.write_image(img_buffer, format='png', width=1000, height=700)
                                        st.download_button(
                                            f"Download {key} (PNG)",
                                            data=img_buffer.getvalue(),
                                            file_name=f"{vis_type}_{key.lower().replace(' ', '_')}.png",
                                            mime='image/png',
                                            key=f'download_vis_{vis_type}_{key}'
                                        )
                                    except Exception as e:
                                        st.warning(f"Could not generate PNG for {key}: {e}")
                            else:
                                # Single figure
                                st.plotly_chart(figs, use_container_width=True)
                                
                                # Download option for single figure
                                img_buffer = BytesIO()
                                try:
                                    figs.write_image(img_buffer, format='png', width=1000, height=700)
                                    st.download_button(
                                        f"Download {vis_type} (PNG)",
                                        data=img_buffer.getvalue(),
                                        file_name=f"{vis_type.lower().replace(' ', '_')}.png",
                                        mime='image/png',
                                        key=f'download_vis_{vis_type}'
                                    )
                                except Exception as e:
                                    st.warning(f"Could not generate PNG for {vis_type}: {e}")
                
                # Download Options
                st.subheader("Download Options")
                csv_data = df_processed.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Processed Data (CSV)",
                    data=csv_data,
                    file_name="processed_data.csv",
                    mime="text/csv",
                    key='download_csv'
                )
                
                excel_buffer = BytesIO()
                df_processed.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_buffer.seek(0)
                st.download_button(
                    "Download Processed Data (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name="processed_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='download_excel'
                )
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
    else:
        st.info("No data available. Please upload or merge data in the 'Data Upload & Merge' mode.")

elif st.session_state['active_mode'] == "Web Scraping":
    st.header("Web Scraping")

    # Initialize session state for scraping
    if 'scraped_dfs' not in st.session_state:
        st.session_state['scraped_dfs'] = []
    if 'scrape_messages' not in st.session_state:
        st.session_state['scrape_messages'] = []

    # Get scraping parameters from sidebar
    url = st.session_state.get('scrape_url_input', 'https://example.com/')
    
    # Prepare scraping parameters
    tags = st.session_state.get('scrape_tags_input', ['table', 'div'])
    custom_tags = st.session_state.get('custom_tags_input', '')
    if custom_tags:
        tags.extend([tag.strip().lower() for tag in custom_tags.split(',') if tag.strip()])
    
    classes_list = None
    if st.session_state.get('scrape_classes_input'):
        classes_list = [cls.strip() for cls in st.session_state['scrape_classes_input'].split(',') if cls.strip()]
    
    ids_list = None
    if st.session_state.get('scrape_ids_input'):
        ids_list = [id.strip() for id in st.session_state['scrape_ids_input'].split(',') if id.strip()]

    # Scrape Button
    if st.button("Scrape Data", key='scrape_button'):
        if not validators.url(url):
            st.error("Invalid URL.")
        else:
            try:
                with st.spinner("Scraping data..."):
                    if st.session_state.get('scraper_type_input') == 'Playwright':
                        data_frames, messages = asyncio.run(scrape_with_playwright(url, tags, classes_list, ids_list))
                    else:
                        data_frames, messages = scrape_with_selenium(url, tags, classes_list, ids_list)

                    if not data_frames:
                        st.error("No data scraped.")
                    else:
                        st.session_state['scraped_dfs'].extend(data_frames)
                        st.session_state['scrape_messages'] = messages.split(" | ") if isinstance(messages, str) else messages
                        st.success(f"Scraped {len(data_frames)} datasets!")
            except Exception as e:
                st.error(f"Error scraping: {str(e)}")

    # Display Scraped Data
    if st.session_state['scraped_dfs']:
        st.subheader("Scraped Datasets")
        for i, df in enumerate(st.session_state['scraped_dfs']):
            st.write(f"**DataFrame {i+1}** ({df.shape[0]:,} rows, {df.shape[1]} cols)")
            st.dataframe(df.head(), use_container_width=True)
            st.download_button(
                f"Download DataFrame {i+1} (CSV)",
                data=df.to_csv(index=False),
                file_name=f"scraped_data_{i+1}.csv",
                mime='text/csv',
                key=f'download_scraped_{i}'
            )

        # Display Messages
        st.subheader("Scraping Messages")
        for msg in st.session_state['scrape_messages']:
            st.write(f"- {msg}")

        # Clear Scraped Data Option
        if st.button("Clear Scraped Data", key='clear_scraped_button'):
            st.session_state['scraped_dfs'] = []
            st.session_state['scrape_messages'] = []
            st.success("Cleared all scraped data.")

        # Merge Scraped Data
        if len(st.session_state['scraped_dfs']) > 1:
            st.subheader("Merge Scraped DataFrames")
            # Find common columns
            all_cols = [set(df.columns) for df in st.session_state['scraped_dfs']]
            common_cols = set.intersection(*all_cols)
            
            if common_cols:
                scraped_merge_key = st.selectbox(
                    "Select Common Column for Merging Scraped Data",
                    sorted(list(common_cols)),
                    key='scraped_merge_key'
                )
                scraped_merge_type = st.selectbox(
                    "Merge Type",
                    ['inner', 'outer', 'left', 'right'],
                    key='scraped_merge_type'
                )
                if st.button("Merge Scraped Data", key='merge_scraped_button'):
                    try:
                        with st.spinner("Merging scraped DataFrames..."):
                            merged_scraped = st.session_state['scraped_dfs'][0].copy()
                            for idx, df_to_merge in enumerate(st.session_state['scraped_dfs'][1:], 1):
                                if scraped_merge_key not in merged_scraped.columns or scraped_merge_key not in df_to_merge.columns:
                                    st.error(f"Merge key '{scraped_merge_key}' not found in DataFrame {idx+1}.")
                                    st.stop()
                                merged_scraped = merged_scraped.merge(
                                    df_to_merge,
                                    on=scraped_merge_key,
                                    how=scraped_merge_type,
                                    suffixes=('', f'_{idx}_scraped')
                                )
                        st.session_state['df'] = merged_scraped
                        st.session_state['scraped_dfs'] = [merged_scraped]
                        st.success("Scraped Data merged and set as current dataset!")
                        st.info(f"Merged Scraped Dataset: {merged_scraped.shape[0]} rows, {merged_scraped.shape[1]} columns")
                        st.dataframe(merged_scraped.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error merging scraped data: {str(e)}")
            else:
                st.info("No common columns found between scraped DataFrames. You can only concatenate (stack).")
                if st.button("Join Scraped Data (Stack Rows)", key="concat_scraped_button"):
                    try:
                        with st.spinner("Joining scraped DataFrames..."):
                            joined = pd.concat(st.session_state['scraped_dfs'], axis=0, ignore_index=True, sort=False)
                        st.session_state['df'] = joined
                        st.session_state['scraped_dfs'] = [joined]
                        st.success("Scraped Data joined and set as current dataset!")
                        st.info(f"Joined Scraped Dataset: {joined.shape[0]} rows, {joined.shape[1]} columns")
                        st.dataframe(joined.head(), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error joining scraped data: {str(e)}")

elif st.session_state['active_mode'] == "NLP Analysis":
    st.header("Natural Language Processing Analysis")
    
    if st.session_state['df'] is not None and not st.session_state['df'].empty:
        df = st.session_state['df']
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if not text_columns:
            st.warning("No text columns found in the dataset. Please ensure your data contains text columns.")
            st.stop()
        
        text_column = st.session_state.get('nlp_text_column', text_columns[0])
        
        st.subheader("Text Data Preview")
        st.info(f"Analyzing column: '{text_column}' | Total texts: {df[text_column].dropna().shape[0]:,}")
        
        # Show sample texts
        sample_texts = df[text_column].dropna().head(3)
        for i, text in enumerate(sample_texts):
            with st.expander(f"Sample Text {i+1}", expanded=False):
                st.write(str(text)[:500] + "..." if len(str(text)) > 500 else str(text))
        
        if st.button("Perform NLP Analysis", key='nlp_analyze_button'):
            try:
                progress = st.progress(0)
                texts = df[text_column].dropna().astype(str)
                
                if len(texts) == 0:
                    st.error("No valid text data found in the selected column.")
                    st.stop()
                
                results = {}
                
                # Text Preprocessing
                if st.session_state.get('nlp_remove_stopwords', True) or st.session_state.get('nlp_lemmatize', True):
                    with st.spinner("Preprocessing text data..."):
                        processed_texts = texts.apply(lambda x: preprocess_text(
                            x,
                            remove_stopwords=st.session_state.get('nlp_remove_stopwords', True),
                            lemmatize=st.session_state.get('nlp_lemmatize', True),
                            remove_punctuation=st.session_state.get('nlp_remove_punct', True),
                            lowercase=st.session_state.get('nlp_lowercase', True)
                        ))
                        results['processed_texts'] = processed_texts
                        progress.progress(0.2)
                
                # Sentiment Analysis
                if st.session_state.get('nlp_sentiment', True):
                    with st.spinner("Performing sentiment analysis..."):
                        sentiment_results = perform_sentiment_analysis(texts)
                        results['sentiment'] = sentiment_results
                        progress.progress(0.4)
                
                # Named Entity Recognition
                if st.session_state.get('nlp_ner', True):
                    with st.spinner("Extracting named entities..."):
                        entities = extract_named_entities(texts)
                        results['entities'] = entities
                        progress.progress(0.6)
                
                # Topic Modeling
                if st.session_state.get('nlp_topics', True):
                    with st.spinner("Performing topic modeling..."):
                        n_topics = st.session_state.get('nlp_n_topics', 5)
                        method = st.session_state.get('nlp_topic_method', 'lda')
                        topics_df, topic_assignments, topic_message = perform_topic_modeling(texts, n_topics, method)
                        results['topics'] = topics_df
                        results['topic_assignments'] = topic_assignments
                        results['topic_message'] = topic_message
                        progress.progress(0.8)
                
                # Word Frequency
                if st.session_state.get('nlp_word_freq', True):
                    with st.spinner("Analyzing word frequency..."):
                        word_freq = generate_word_frequency(texts, top_n=20)
                        results['word_frequency'] = word_freq
                        progress.progress(1.0)
                
                # Store results in session state
                st.session_state['nlp_results'] = results
                
                st.success("NLP analysis completed successfully!")
                
                # Display Results
                st.header("NLP Analysis Results")
                
                # Sentiment Analysis Results
                if 'sentiment' in results and not results['sentiment'].empty:
                    st.subheader("Sentiment Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    sentiment_counts = results['sentiment']['sentiment'].value_counts()
                    avg_polarity = results['sentiment']['polarity'].mean()
                    avg_subjectivity = results['sentiment']['subjectivity'].mean()
                    
                    with col1:
                        st.metric("Most Common Sentiment", sentiment_counts.index[0], 
                                f"{sentiment_counts.iloc[0]} texts")
                    with col2:
                        st.metric("Average Polarity", f"{avg_polarity:.3f}", 
                                "(-1=negative, +1=positive)")
                    with col3:
                        st.metric("Average Subjectivity", f"{avg_subjectivity:.3f}", 
                                "(0=objective, 1=subjective)")
                    
                    # Add sentiment columns to dataframe
                    df_with_sentiment = df.copy()
                    sentiment_data = results['sentiment'].reindex(df[text_column].dropna().index)
                    df_with_sentiment = df_with_sentiment.join(sentiment_data, rsuffix='_sentiment')
                    st.dataframe(df_with_sentiment[[text_column, 'sentiment', 'polarity', 'subjectivity']].head(), 
                               use_container_width=True)
                
                # Named Entity Recognition Results
                if 'entities' in results and not results['entities'].empty:
                    st.subheader("Named Entities")
                    entity_counts = results['entities']['label'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Entity Types:**")
                        st.dataframe(entity_counts.head(10), use_container_width=True)
                    
                    with col2:
                        st.write("**Sample Entities:**")
                        sample_entities = results['entities'].groupby('label').head(3)
                        st.dataframe(sample_entities, use_container_width=True)
                
                # Topic Modeling Results
                if 'topics' in results and results['topics'] is not None:
                    st.subheader("Topic Modeling")
                    st.info(results['topic_message'])
                    
                    st.write("**Discovered Topics:**")
                    st.dataframe(results['topics'], use_container_width=True)
                
                # Word Frequency Results
                if 'word_frequency' in results and not results['word_frequency'].empty:
                    st.subheader("Word Frequency Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Top 10 Words:**")
                        st.dataframe(results['word_frequency'].head(10), use_container_width=True)
                    
                    with col2:
                        total_words = results['word_frequency']['frequency'].sum()
                        unique_words = len(results['word_frequency'])
                        st.metric("Total Words", f"{total_words:,}")
                        st.metric("Unique Words", f"{unique_words:,}")
                
                # Visualizations
                st.header("NLP Visualizations")
                nlp_visualizations = create_nlp_visualizations(df, text_column)
                
                if nlp_visualizations:
                    for vis_name, fig in nlp_visualizations.items():
                        with st.expander(f"{vis_name}", expanded=True):
                            if isinstance(fig, plt.Figure):
                                st.pyplot(fig)
                                # Download option for matplotlib figures
                                img_buffer = BytesIO()
                                fig.savefig(img_buffer, format='png', bbox_inches='tight', 
                                          facecolor='black', dpi=150)
                                st.download_button(
                                    f"Download {vis_name} (PNG)",
                                    data=img_buffer.getvalue(),
                                    file_name=f"{vis_name.lower().replace(' ', '_')}_nlp.png",
                                    mime='image/png',
                                    key=f'download_nlp_{vis_name}'
                                )
                            else:
                                st.plotly_chart(fig, use_container_width=True)
                                # Download option for plotly figures
                                img_buffer = BytesIO()
                                try:
                                    fig.write_image(img_buffer, format='png', width=1000, height=700)
                                    st.download_button(
                                        f"Download {vis_name} (PNG)",
                                        data=img_buffer.getvalue(),
                                        file_name=f"{vis_name.lower().replace(' ', '_')}_nlp.png",
                                        mime='image/png',
                                        key=f'download_nlp_{vis_name}'
                                    )
                                except Exception as e:
                                    st.warning(f"Could not generate PNG for {vis_name}: {e}")
                
                # Download NLP Results
                st.header("Download NLP Results")
                
                # Prepare combined results dataframe
                if 'sentiment' in results:
                    nlp_results_df = df.copy()
                    sentiment_data = results['sentiment'].reindex(df.index, fill_value={'polarity': 0, 'subjectivity': 0, 'sentiment': 'unknown'})
                    for col in sentiment_data.columns:
                        nlp_results_df[f'nlp_{col}'] = sentiment_data[col]
                    
                    # Download options
                    csv_data = nlp_results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results with NLP Data (CSV)",
                        data=csv_data,
                        file_name='data_with_nlp_analysis.csv',
                        mime='text/csv',
                        key='download_nlp_csv'
                    )
                    
                    excel_buffer = BytesIO()
                    nlp_results_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_buffer.seek(0)
                    st.download_button(
                        "Download Results with NLP Data (Excel)",
                        data=excel_buffer.getvalue(),
                        file_name='data_with_nlp_analysis.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key='download_nlp_excel'
                    )
                
            except Exception as e:
                st.error(f"Error during NLP analysis: {str(e)}")
    else:
        st.info("No data available. Please upload data in 'Data Upload & Merge' mode first.")

# --- Footer ---
st.markdown(
    "<hr><p style='text-align: center; color: #e0e0e0;'>DataSync Pro - Developed by AMINE EL HEND & MOUAD BOULAID | Version 7.3 | May 2025</p>",
    unsafe_allow_html=True
)