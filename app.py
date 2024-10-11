import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request
from flask_cors import CORS  # Import CORS
import os

# Load the pre-trained model and data
model = load_model('chatbot_model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
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
    return return_list

class YourChatbotClass:
    def __init__(self, threshold, intents):
        self.threshold = threshold
        self.intents = intents

    def get_response(self, predicted_intent, confidence, sentence):
        if confidence > self.threshold:
            for intent in self.intents['intents']:
                if intent['tag'] == predicted_intent:
                    responses = intent['responses']
                    return random.choice(responses)
            # If predicted intent not found, return a default message
            return "I'm sorry, that's beyond my trained knowledge. Can you please provide more information or rephrase your question?"
        else:
            # Check if any keyword in the sentence matches any patterns
            sentence_words = clean_up_sentence(sentence)
            for word in sentence_words:
                for intent in self.intents['intents']:
                    if word in intent['patterns']:
                        return "I'm sorry, that's beyond my trained knowledge. Can you please provide more information or rephrase your question?"
            # If no keywords match, return a default message
            return "I'm sorry, that's beyond my trained knowledge. Could you please rephrase or be more specific?"

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = chatbot.get_response(ints[0]['intents'], float(ints[0]['probability']), msg)
    return res

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
 # Enable CORS for the entire app

app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

chatbot = YourChatbotClass(0.97, intents)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)), debug=True)  # For testing


