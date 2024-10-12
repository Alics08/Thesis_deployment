import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS
import os
import logging  # Import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model and data
model = load_model(os.path.join(os.path.dirname(__file__), 'chatbot_model.h5'))
intents = json.loads(open(os.path.join(os.path.dirname(__file__), 'data.json')).read())
words = pickle.load(open(os.path.join(os.path.dirname(__file__), 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(os.path.dirname(__file__), 'classes.pkl'), 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    logging.info("found in bag: %s" % w)  # Log found words
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.30
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intents": classes[r[0]], "probability": str(r[1])})
    logging.info(f"Prediction results: {return_list}")  # Log prediction results
    return return_list

class YourChatbotClass:
    def __init__(self, threshold, intents):
        self.threshold = threshold
        self.intents = intents

    def get_response(self, predicted_intent, confidence, sentence):
        logging.info(f"Received message: {sentence}")  # Log received message
        logging.info(f"Predicted intent: {predicted_intent} with confidence: {confidence}")  # Log intent and confidence
        if confidence > self.threshold:
            for intent in self.intents['intents']:
                if intent['tag'] == predicted_intent:
                    responses = intent['responses']
                    selected_response = random.choice(responses)
                    logging.info(f"Selected response: {selected_response}")  # Log selected response
                    return selected_response
            return "I'm sorry, that's beyond my trained knowledge. Can you please provide more information or rephrase your question?"
        else:
            sentence_words = clean_up_sentence(sentence)
            for word in sentence_words:
                for intent in self.intents['intents']:
                    if word in intent['patterns']:
                        return "I'm sorry, that's beyond my trained knowledge. Can you please provide more information or rephrase your question?"
            return "I'm sorry, that's beyond my trained knowledge. Could you please rephrase or be more specific?"

def chatbot_response(msg):
    ints = predict_class(msg, model)
    if ints:  # Check if predictions are available
        res = chatbot.get_response(ints[0]['intents'], float(ints[0]['probability']), msg)
        return res
    else:
        logging.warning(f"No intents predicted for message: {msg}")  # Log warning if no intent is found
        return "I'm sorry, I didn't understand that."

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

chatbot = YourChatbotClass(0.97, intents)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)), debug=os.environ.get('FLASK_DEBUG', '0') == '1')  # For testing
