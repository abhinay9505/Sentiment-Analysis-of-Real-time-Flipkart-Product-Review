from flask import Flask,request,render_template

import re
import sklearn
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import string
lemmatizer = WordNetLemmatizer()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def clean(doc): # doc is a string of text
    doc = doc.replace("READ MORE", "")
    
    # Remove punctuation and numbers.
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])

    # Converting to lower case
    doc = doc.lower()
    
    # Tokenization
    tokens = nltk.word_tokenize(doc)

    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Stop word removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
    
    # Join and return
    return " ".join(filtered_tokens)

@app.route('/prediction' ,methods=['POST'])

def prediction():
    text = [request.form.get("text")]
    text_clean = [clean(texts) for texts in text]

    model = joblib.load("best_models/navie_bayes.pkl")
    prediction = model.predict(text_clean)
    return render_template("prediction.html", prediction=prediction[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000, debug=True)