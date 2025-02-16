import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import tkinter
from tkinter import *

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load trained model and necessary files
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    # Tokenize and lemmatize the sentence
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words, show_details=True):
    # Create a bag of words
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {word}")
    return np.array(bag)

def predict_class(sentence):
    # Predict the intent of the user input
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    # Fetch the appropriate response for the predicted intent
    tag = ints[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I don't understand."

def send():
    # Send the user message and get a bot response
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12))

        ints = predict_class(msg)
        res = get_response(ints, intents)

        ChatBox.insert(END, "Bot: " + res + '\n\n')
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)

# Tkinter GUI Setup
root = Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=False, height=False)

# Create Chat Window
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial")
ChatBox.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

# Create Send Button
SendButton = Button(
    root, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
    bd=0, bg="#f9a602", activebackground="#3c9d9b", fg='#000000', command=send
)

# Create Entry Box
EntryBox = Text(root, bd=0, bg="white", width="29", height="5", font="Arial")

# Place Components on the Screen
scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

root.mainloop()
