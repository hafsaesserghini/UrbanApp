import nltk
import json
import random
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
import bcrypt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

nltk.download('punkt')
nltk.download('stopwords')

tokenizer = TreebankWordTokenizer()
client = MongoClient("mongodb://localhost:27017/")
db = client['database']
users_collection = db['users']

# Charger le dataset
with open('intents.json') as file:
    data = json.load(file)

intents = data['intents']
patterns = []
tags = []
responses_dict = {}

for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    responses_dict[intent['tag']] = intent['responses']

stop_words = set(stopwords.words('english'))
words = []
cleaned_patterns = []
for pattern in patterns:
    tokenized_words = tokenizer.tokenize(pattern)
    cleaned = [word.lower() for word in tokenized_words if word.isalnum() and word.lower() not in stop_words]
    cleaned_patterns.append(" ".join(cleaned))
    words.extend(cleaned)

words = sorted(list(set(words)))
vectorizer = TfidfVectorizer(vocabulary=words)
X = vectorizer.fit_transform(cleaned_patterns).toarray()

unique_tags = sorted(set(tags))
y = np.array([unique_tags.index(tag) for tag in tags])
y = tf.keras.utils.to_categorical(y, num_classes=len(unique_tags))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(unique_tags), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

def predict_intent(text):
    text_tokens = tokenizer.tokenize(text)
    text_tokens = [word.lower() for word in text_tokens if word.isalnum() and word.lower() not in stop_words]
    text_vector = vectorizer.transform([" ".join(text_tokens)]).toarray()
    prediction = model.predict(text_vector)
    tag_index = np.argmax(prediction)
    return unique_tags[tag_index]

def get_response(tag):
    return random.choice(responses_dict[tag])

app = Flask(__name__)
app.secret_key = "your_secret_key"

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if users_collection.find_one({"email": email}):
            return "Email déjà enregistré", 400
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        users_collection.insert_one({"email": email, "password": hashed_password})
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = users_collection.find_one({"email": email})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['user_id'] = str(user['_id'])
            session['email'] = user['email']
            return redirect(url_for('chat_page'))
        else:
            return jsonify({"error": "Invalid email or password"}), 401
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/chat")
def chat_page():
    if 'user_id' in session:
        return render_template('chat.html')
    return redirect(url_for('login'))

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = get_chat_response(msg)
    return response

def get_chat_response(user_input):
    predicted_tag = predict_intent(user_input)
    response = get_response(predicted_tag)
    return response

# --- Routes ajoutées depuis l'ancienne version ---

@app.route("/categories")
def categories():
    return render_template("categories.html")

@app.route("/flood")
def flood():
    return render_template("flood.html")

@app.route("/earthquake")
def earthquake():
    return render_template("earthquake.html")

@app.route("/medical-emergency")
def med():
    return render_template("medical-emergency.html")

@app.route("/power-outage")
def powero():
    return render_template("power-outage.html")

@app.route("/fire-incident")
def firei():
    return render_template("fire-incident.html")

@app.route("/road-accident")
def roada():
    return render_template("road-accident.html")

@app.route("/water-shortage")
def watershortage():
    return render_template("water-shortage.html")

@app.route("/transportation-disruption")
def transportationd():
    return render_template("transportation-disruption.html")

@app.route("/pollution-hazard")
def pollutionhazard():
    return render_template("pollution-hazard.html")

if __name__ == '__main__':
    app.run(debug=True)
