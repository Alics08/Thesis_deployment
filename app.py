import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS

import os
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model and data
try:
    model_path = os.path.join(os.path.dirname(__file__), 'chatbot_model.h5')
    model = load_model(model_path)
    logging.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logging.error(f"Error loading model: {e}")

try:
    intents_path = os.path.join(os.path.dirname(__file__), 'data.json')
    intents = json.loads(open(intents_path).read())
    logging.info(f"Intents loaded successfully from {intents_path}")
except Exception as e:
    logging.error(f"Error loading intents: {e}")

try:
    words_path = os.path.join(os.path.dirname(__file__), 'words.pkl')
    words = pickle.load(open(words_path, 'rb'))
    logging.info(f"Words loaded successfully from {words_path}")
except Exception as e:
    logging.error(f"Error loading words: {e}")

try:
    classes_path = os.path.join(os.path.dirname(__file__), 'classes.pkl')
    classes = pickle.load(open(classes_path, 'rb'))
    logging.info(f"Classes loaded successfully from {classes_path}")
except Exception as e:
    logging.error(f"Error loading classes: {e}")

lemmatizer = WordNetLemmatizer()

# Function to clean and process the user's input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert the user's sentence into a bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    logging.info(f"Found in bag: {w}")
    return np.array(bag)

# Function to predict the intent based on the user's input
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.30
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intents": classes[r[0]], "probability": str(r[1])})
    logging.info(f"Prediction results: {return_list}")
    return return_list

# Class to get responses from the chatbot
class YourChatbotClass:
    def __init__(self, threshold, intents):
        self.threshold = threshold
        self.intents = intents

    def get_response(self, predicted_intent, confidence, sentence):
        logging.info(f"Received message: {sentence}")
        logging.info(f"Predicted intent: {predicted_intent} with confidence: {confidence}")
        if confidence > self.threshold:
            for intent in self.intents['intents']:
                if intent['tag'] == predicted_intent:
                    responses = intent['responses']
                    selected_response = random.choice(responses)
                    logging.info(f"Selected response: {selected_response}")
                    return selected_response
            return "I'm sorry, that's beyond my trained knowledge. Can you please provide more information or rephrase your question?"
        else:
            sentence_words = clean_up_sentence(sentence)
            for word in sentence_words:
                for intent in self.intents['intents']:
                    if word in intent['patterns']:
                        return "I'm sorry, that's beyond my trained knowledge. Can you please provide more information or rephrase your question?"
            return "I'm sorry, that's beyond my trained knowledge. Could you please rephrase or be more specific?"

# Function to get the chatbot response
def chatbot_response(msg):
    try:
        ints = predict_class(msg, model)
        if ints:  # Check if predictions are available
            res = chatbot.get_response(ints[0]['intents'], float(ints[0]['probability']), msg)
            return res
        else:
            logging.warning(f"No intents predicted for message: {msg}")
            return "I'm sorry, I didn't understand that."
    except Exception as e:
        logging.error(f"Error during chatbot response generation: {e}")
        return "There was an error processing your request. Please try again."

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

chatbot = YourChatbotClass(0.97, intents)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)), debug=os.environ.get('FLASK_DEBUG', '0') == '1')