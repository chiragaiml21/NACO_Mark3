#Importing all the Libraries
import re
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#Reading the intents.json file
with open("D://Study Material//GLATHON//PyTorch_model//Pyrates_bot-main//FINAL//back//intents.json", "r", encoding="utf8") as file:
    data = json.load(file)
intents = data["intents"]


Patterns = []
Tags = []
lemmatizer = WordNetLemmatizer()


#Function for preprocessing the text
def preprocess_text(text):
    # Normalize text (convert to lowercase and remove non-alphanumeric characters)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)   #Substitute the matched string in text with space.

    #Tokenize text
    tokenized_words = word_tokenize(text)

    # Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokenized_words]   #changing, Changed---->change

    processed_text = " ".join(lemmatized_words)
    return processed_text


#Separating the Tags and Patterns
for intent in intents:
    tag = intent['tag']
    patterns = intent['patterns']
    for pattern in patterns:
        processed_pattern = preprocess_text(pattern)
        Patterns.append(processed_pattern)
        Tags.append(tag)

# --------Saving Tags and Patterns in pickle files to use them in Mark3 file------
# Patterns = sorted(set(Patterns))
# Tags = sorted(set(Tags))

pickle.dump(Patterns, open('Patterns.pkl', 'wb'))
pickle.dump(Tags, open('Tags.pkl', 'wb'))
# ----------------------------------------------------------------------------------



# Tokenize the Patterns
tokenizer = Tokenizer(lower=True, split=' ')
yo = tokenizer.fit_on_texts(Patterns)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
print(yo)
# # Encode the tags
# encoder = LabelEncoder()
# encoder.fit(Tags)
# y = encoder.transform(Tags)

# # Pad sequences
# max_length = 20
# X = pad_sequences(tokenizer.texts_to_sequences(Patterns), maxlen=max_length, padding="post")



# # Split training and testing data
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# #Creating the model
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length, trainable=True))
# model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
# model.add(tf.keras.layers.GlobalMaxPooling1D())
# model.add(tf.keras.layers.Dense(32, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(len(np.unique(y)), activation="softmax"))
# model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# model.summary()


# #Training the model
# model_history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))


# #Plotting the accuracy and loss
# model_history.history.keys()
# accuracy = model_history.history['accuracy']
# print("The accuracy of model is : ",accuracy[-1]*100,"%")


# model.save("Mark3.model")