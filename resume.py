import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from xgboost import XGBClassifier
from bs4 import BeautifulSoup

# Step 1: Load Dataset
data = pd.read_csv("Resume.csv")

# Step 2: Text Cleaning Functions
def clean_html(html_content):
    if pd.isna(html_content):
        return ''
    soup = BeautifulSoup(html_content, "html.parser")
    clean_text = soup.get_text(separator=' ', strip=True)
    return clean_text

def clean_text(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\w\s,.\-]', '', text)  # Remove special characters except basic punctuation
    return text.strip()

# Preprocessing Function for TF-IDF
def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english')) | {"resume", "cv", "job", "role", "employment"}  # Custom stopwords
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Lowercase
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Stemming + remove stopwords
    return ' '.join(words)

# Step 3: Apply Cleaning Process
# Clean 'Resume_html' and 'Resume_str' (if 'Resume_html' exists)
if 'Resume_html' in data.columns:
    data['Cleaned_Resume_html'] = data['Resume_html'].apply(clean_html)
    data['Resume_str'] = data['Resume_str'] + ' ' + data['Cleaned_Resume_html']  # Combine cleaned HTML with string

data['Cleaned_Resume_str'] = data['Resume_str'].apply(clean_text)  # General text cleaning
X = data['Cleaned_Resume_str'].apply(preprocess_text)  # Preprocess cleaned text
y = data['Category']

# Step 4: Encode Labels
y = pd.factorize(y)[0]

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Step 6: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), sublinear_tf=True)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 7: Length-Based Features
def add_length_features(text_series):
    return pd.DataFrame({
        'char_count': text_series.apply(len),
        'word_count': text_series.apply(lambda x: len(x.split())),
        'avg_word_length': text_series.apply(lambda x: np.mean([len(word) for word in x.split()]) if x else 0)
    })

X_train_len = add_length_features(X_train)
X_test_len = add_length_features(X_test)

# Combine TF-IDF and Length Features
from scipy.sparse import hstack
X_train_combined = hstack([X_train_tfidf, X_train_len.values])
X_test_combined = hstack([X_test_tfidf, X_test_len.values])

# Step 8: Handle Imbalanced Data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)

from xgboost import XGBClassifier
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test_combined)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))


# Save the model using XGBoost's native method
xgb_model.save_model("best_xgb_model.json")
print("Model saved as best_xgb_model.json")

import joblib


model_filename = 'tfidf_vectorizer.pkl'
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print(f"Model saved as {model_filename}")