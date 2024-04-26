import pandas as pd
import nltk
import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from flask import Flask, render_template, request

app = Flask(__name__)

# Download NLTK resources (only required once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Load your dataset with explicit encoding
dataset_path = '/home/ubuntu/finaldataset.csv'
df = pd.read_csv(dataset_path)

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

# Apply the preprocessing function to your DataFrame
df['clean_text'] = df['review'].apply(preprocess_text)

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Train the model
predictions = []
for i, row in df.iterrows():
    review = row['clean_text']
    
    sentiment_score = sia.polarity_scores(review)
    if sentiment_score['compound'] >= 0.05:
        predicted_sentiment = 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        predicted_sentiment = 'Negative'
    else:
        predicted_sentiment = 'Neutral'
    
    predictions.append(predicted_sentiment)

# Add predicted sentiment column to the DataFrame
df['predicted_sentiment'] = predictions

# Serialize the model, sentiment analyzer, and preprocessing function
model_data = {
    'model': sia,
    'preprocess': preprocess_text
}

# Save the model using joblib
joblib.dump(model_data, 'sentiment_model.pkl')

# Load the model
with open('sentiment_model.pkl', 'rb') as f:
    model_data = joblib.load(f)
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
    app.run(host='0.0.0.0', port=8080)
