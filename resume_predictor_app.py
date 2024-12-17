import streamlit as st
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from xgboost import XGBClassifier
from bs4 import BeautifulSoup
import joblib
import nltk
nltk.download('stopwords')
# Load necessary resources
@st.cache_resource
def load_model_and_vectorizer():
    model = XGBClassifier()
    model.load_model("best_xgb_model.json")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Save TF-IDF vectorizer earlier
    return model, vectorizer

# Text cleaning functions
def clean_html(html_content):
    if pd.isna(html_content):
        return ''
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=' ', strip=True)

def clean_text(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines
    text = re.sub(r'[^\w\s,.\-]', '', text)  # Remove unwanted characters
    return text.strip()

def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english')) | {"resume", "cv", "job", "role", "employment"}
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def add_length_features(text_series):
    return pd.DataFrame({
        'char_count': text_series.apply(len),
        'word_count': text_series.apply(lambda x: len(x.split())),
        'avg_word_length': text_series.apply(lambda x: np.mean([len(word) for word in x.split()]) if x else 0)
    })

# Set a custom title and description with markdown
st.set_page_config(page_title="Resume Category Predictor", layout="wide")
st.title("**Resume Category Predictor**")
st.markdown(
    """
    Welcome to the **Resume Category Predictor**! 
    Upload a resume file (TXT or PDF) and let us predict its most likely category based on the content.
    
    ### Instructions:
    1. Click on the file upload button below.
    2. Choose a resume file in TXT or PDF format.
    3. Wait for the processing and see the predicted category.
    """
)

# Sidebar for extra options or description
st.sidebar.header("About This App")
st.sidebar.markdown(
    """
    This application uses machine learning to classify resumes into various categories based on their content.
    
    It uses XGBoost and TF-IDF for text classification.
    
    #### Categories:
    HR, DESIGNER, INFORMATION-TECHNOLOGY, TEACHER, ADVOCATE, BUSINESS-DEVELOPMENT, HEALTHCARE, FITNESS, AGRICULTURE, BPO, SALES, CONSULTANT, DIGITAL-MEDIA, AUTOMOBILE, CHEF, FINANCE, APPAREL, ENGINEERING, ACCOUNTANT, CONSTRUCTION, PUBLIC-RELATIONS, BANKING, ARTS, AVIATION
    """
)

# Load model and vectorizer
xgb_model, tfidf_vectorizer = load_model_and_vectorizer()

# File upload
uploaded_file = st.file_uploader("Upload a resume file (TXT or PDF)", type=["txt", "pdf"])

if uploaded_file is not None:
    # Show a loading spinner while the file is being processed
    with st.spinner("Processing the file..."):
        # Process uploaded file
        if uploaded_file.type == "application/pdf":
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ''.join(page.extract_text() for page in pdf_reader.pages)
        else:
            text = uploaded_file.read().decode("utf-8")

        # Preprocess text
        cleaned_text = clean_text(text)
        preprocessed_text = preprocess_text(cleaned_text)
        text_series = pd.Series([preprocessed_text])

        # TF-IDF vectorization and feature extraction
        tfidf_features = tfidf_vectorizer.transform(text_series)
        length_features = add_length_features(text_series)
        combined_features = hstack([tfidf_features, length_features.values])

        # Make predictions
        prediction = xgb_model.predict(combined_features)[0]

        # Categories Mapping
        categories = [
            'HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE',
            'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE', 'BPO', 'SALES',
            'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE', 'CHEF', 'FINANCE', 'APPAREL',
            'ENGINEERING', 'ACCOUNTANT', 'CONSTRUCTION', 'PUBLIC-RELATIONS', 'BANKING',
            'ARTS', 'AVIATION'
        ]
        predicted_category = categories[prediction]

        # Display output with custom styling
        st.success(f"**Predicted Category**: **{predicted_category}**", icon="✅")


