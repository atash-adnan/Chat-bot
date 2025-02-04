# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import SGD
# import random
# import nltk 
# from nltk.stem import WordNetLemmatizer 
# lemmatizer = WordNetLemmatizer()
# import json
# import pickle 
# import os 

# words=[]
# classes = []
# documents = []
# ignore_letters = ['!', '?', ',', '.']
# intents_file = open('intents.json').read()
# intents = json.loads(intents_file) 

# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         # Add your logic here, e.g., print the pattern
#         print(pattern)  

#         #tokenize each word
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         word = nltk.word_tokenize(pattern)
#         words.extend(word)
#         documents.append((word, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define lists
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

# Load intents from the JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Download the required NLTK data for tokenization
nltk.download('punkt')

# Tokenize and process patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add the pattern to the documents list
        documents.append((word_list, intent['tag']))
        # Add the tag to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and sort the words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))

# Sort the classes
classes = sorted(set(classes))

print(f"{len(documents)} documents")  # patterns and intents
print(f"{len(classes)} classes: {classes}")  # intents
print(f"{len(words)} unique lemmatized words: {words}")  # vocabulary

# Save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    # Tokenize and lemmatize words in the pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    # Create a bag of words array
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # Create an output row
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle the training data and convert to numpy arrays
random.shuffle(training)
training = np.array(training, dtype=object)

# Separate features (X) and labels (Y)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))
print("Training data created")

# Create the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5', hist)
print("Model created and saved successfully")


