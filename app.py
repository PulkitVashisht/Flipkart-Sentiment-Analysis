from flask import Flask, request, render_template
from joblib import load
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = load(r"C:\Users\pulki\OneDrive\Desktop\flipkart sentiment analysis\webapp\Pickle\Random_forest_model.pkl")
tfidf_vectorizer = load(r"C:\Users\pulki\OneDrive\Desktop\flipkart sentiment analysis\webapp\Pickle\tfidf_vectorizer.pkl")

import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')
import nltk
nltk.download('wordnet')

# Preprocessing functions
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r'\W+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_words)

# Perform sentiment analysis
def predict_sentiment(input_text):
    preprocess_text = clean_text(input_text)
    preprocessed_text = lemmatize_text(preprocess_text)
    features = tfidf_vectorizer.transform([preprocessed_text])
    prediction = model.predict(features)[0]
    return "Positive" if prediction == 'positive' else "Negative"

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for handling form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = predict_sentiment(text)
        return render_template('output.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)