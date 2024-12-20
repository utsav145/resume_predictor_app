
# Resume Category Predictor

## Overview
The **Resume Category Predictor** is a machine learning-based web application designed to automatically categorize resumes into predefined categories based on their content. This application uses **XGBoost** for classification and **TF-IDF** for feature extraction. It streamlines the resume screening process for HR professionals and recruiters by automating the categorization of uploaded resumes.

## Features
- **File Upload**: Users can upload resumes in **TXT** or **PDF** format.
- **Text Preprocessing**: Resumes undergo text cleaning, including:
  - HTML tag removal (for PDF files).
  - Removal of unwanted characters and extra whitespace.
  - Text normalization (lowercasing, stemming).
- **Feature Extraction**: Uses **TF-IDF** to convert text into numerical features. Additional length-based features (word count, character count, average word length) are also extracted.
- **Prediction**: The preprocessed features are passed through a trained **XGBoost** model to predict the resume's category.
- **Real-time Feedback**: Displays the predicted category of the uploaded resume.

## Technologies Used
- **Streamlit**: For creating the interactive web interface.
- **XGBoost**: For training the classification model.
- **TF-IDF Vectorizer**: For converting text data into numerical features.
- **BeautifulSoup**: For parsing and cleaning HTML from PDF files.
- **NLTK**: For text preprocessing (stemming and stopwords removal).
- **Joblib**: For saving and loading the trained model and vectorizer.

## Requirements
To run this project, you need the following Python libraries:
- `streamlit`
- `xgboost`
- `nltk`
- `joblib`
- `scipy`
- `scikit-learn`
- `pandas`
- `pyPDF2`
- `beautifulsoup4`

You can install the required libraries by running the following command:
```bash
pip install -r requirements.txt
```

## Setup
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/resume-category-predictor.git
   cd resume-category-predictor
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that the necessary files (`best_xgb_model.json` and `tfidf_vectorizer.pkl`) are in the project directory:
   - `best_xgb_model.json`: The trained XGBoost model.
   - `tfidf_vectorizer.pkl`: The TF-IDF vectorizer used for transforming resume text.

## How to Use
1. Run the Streamlit application:
   ```bash
   streamlit run resume_predictor_app.py
   ```

2. Open the web interface in your browser, where you can upload a resume in **TXT** or **PDF** format.

3. Once uploaded, the application will process the file, clean and transform the text, and predict the resume's category using the trained XGBoost model.

4. The predicted category will be displayed on the screen in real-time.

## Model Details
- **Model Type**: XGBoost Classifier (Gradient Boosting)
- **Features**: The model uses TF-IDF features combined with additional length-based features (character count, word count, and average word length).
- **Categories**: The resumes are classified into predefined categories based on your dataset. The categories could be roles such as "Software Engineer", "Data Scientist", "Product Manager", etc.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **XGBoost**: A scalable machine learning system for tree boosting.
- **TF-IDF**: A widely used method for text feature extraction.
- **BeautifulSoup**: A library for web scraping and parsing HTML content.

---

### `requirements.txt` (for convenience)
```plaintext
streamlit
xgboost
nltk
joblib
scipy
scikit-learn
pandas
pyPDF2
beautifulsoup4
```
