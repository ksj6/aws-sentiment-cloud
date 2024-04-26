from flask import Flask, render_template, request
import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import numpy as np

app = Flask(__name__)

# Download NLTK resources (only required once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Load the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Preprocess text function
def preprocess_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    
    # Convert words to lowercase
    tokens = [word.lower() for word in tokens]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Reconstruct the text from tokens
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Load the model
with open('sentiment_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    sia = model_data['model']
    preprocess = model_data['preprocess']

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_sentiment():
    text = request.form.get('text')
    print("Received input text:", text)  # Debug statement
    if text is None or text == '':
        return render_template('index.html', sentiment='Error: Empty input')
    cleaned_text = preprocess(text)
    sentiment_score = sia.polarity_scores(cleaned_text)
    
    if sentiment_score['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return render_template('index.html', sentiment=sentiment)


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0',port=8080)