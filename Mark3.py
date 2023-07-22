# from imp import load_module
import re
import json
import random
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model




# --------------------------------------------Basic Preprocessing-------------------------------------------------
#Reading the json file
with open("D://Study Material//GLATHON//PyTorch_model//Pyrates_bot-main//FINAL//back//intents.json", "r", encoding="utf8") as file:
    data = json.load(file)

#Initializing the variables
intents = data["intents"]
lemmatizer = WordNetLemmatizer()
encoder = LabelEncoder()
tokenizer = Tokenizer(lower=True, split=' ')
max_length = 20

# Load the model and pickle files
Patterns = pickle.load(open('Patterns.pkl', 'rb'))
Tags = pickle.load(open('Tags.pkl', 'rb'))
model = load_model('Mark3.model')

# Tokenize words
tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(Patterns)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Encode the classes
encoder = LabelEncoder()
encoder.fit(Tags)
y = encoder.transform(Tags)

# Pad sequences
max_length = 20
X = pad_sequences(tokenizer.texts_to_sequences(Patterns), maxlen=max_length, padding="post")
# ------------------------------------------------End---------------------------------------------------------------




# -----------------------------------------Functions----------------------------------------------------------------
#Function for preprocessing the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokenized_words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokenized_words]
    processed_text = " ".join(lemmatized_words)
    return processed_text


# Function to get the response
def get_response():
    while True:
        message = input("Enter your message:")
        message = preprocess_text(message)
        if message == "quit" or message == "stop":
            break
        message = pad_sequences(tokenizer.texts_to_sequences([message]), maxlen=max_length, padding="post")
        prediction = model.predict(message)
        tag = encoder.inverse_transform([np.argmax(prediction)])
        for intent in intents:
            if intent['tag'] == tag:
                print(random.choice(intent['responses']))
# -------------------------------------------------End---------------------------------------------------------       
get_response()