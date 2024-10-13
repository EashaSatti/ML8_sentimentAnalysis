import tensorflow as tf
from flask import Flask, render_template, request
import numpy as np
import re
import os
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

# Load the model globally
try:
    model = load_model('sentiment_analysis_model_new.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route('/sentiment_analysis_prediction', methods=['GET', 'POST'])
def sent_only_prediction():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = ''
        probability = None
        img_filename = None

        if model is not None:
            max_review_length = 200
            word_to_id = imdb.get_word_index()
            strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
            text = text.lower().replace("<br />", " ")
            text = re.sub(strip_special_chars, "", text.lower())

            words = text.split()
            x_test = [[word_to_id.get(word, 0) if word in word_to_id else 0 for word in words]]
            x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
            vector = np.array(x_test)

            try:
                prediction = model.predict(vector)
                probability = prediction[0][0]
                print(f"Raw prediction: {probability}")

                probability = round(probability * 100, 2)  # Convert to percentage
    
                 # Adjust threshold to interpret sentiment
                if probability < 45:  # Setting a new lower range for Negative
                    sentiment = 'Negative'
                    img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Sad.jpeg')
                elif probability >= 45 and probability <= 55:  # Tightening the Neutral range
                    sentiment = 'Neutral'
                    img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Neutral.jpeg')
                else:  # For anything above 55, it will be Positive
                    sentiment = 'Positive'
                    img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling.jpeg')
    
            except Exception as e:
                print(f"Error during prediction: {e}")
                sentiment = 'Error during prediction'
        else:
             sentiment = 'Model not loaded'

        return render_template('home.html', text=text, sentiment=sentiment, probability=probability, image=img_filename)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
