from flask import Flask, render_template, request, jsonify
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

with open("intents.json") as file:
    intents = json.load(file)

def clean_up(sentence):
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words

def predict_class(sentence):
    sentence_words = clean_up(sentence)
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern_words = clean_up(pattern)
            if all(word in sentence_words for word in pattern_words):
                return intent["tag"]
    return "unknown"

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I don't understand that."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.form["msg"]
    intent = predict_class(user_input)
    response = get_response(intent)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)
