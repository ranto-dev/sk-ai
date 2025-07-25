from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import random
import nltk
import json
import os 

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file_path = 'dataset-Yitronix-acceuil.json'

# Vérifier si le fichier intents.json existe
if not os.path.exists(data_file_path):
    print(f"Erreur: Le fichier '{data_file_path}' est introuvable.")
    print("Veuillez vous assurer qu'il se trouve dans le même répertoire que le script.")
    exit()

data_file = open(data_file_path, encoding='utf-8').read() # Spécifier l'encodage
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent ['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))
        
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

# Sauvegarder words.pkl et classes.pkl
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

# --- Correction ici: Créer train_x et train_y séparément ---
train_x = []
train_y = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    train_x.append(bag)
    train_y.append(output_row)

# Shuffle our features and turn into np.array
# Il est préférable de mélanger X et Y ensemble pour maintenir la correspondance
combined = list(zip(train_x, train_y))
random.shuffle(combined)
train_x, train_y = zip(*combined) # Dézipper les listes

train_x = np.array(list(train_x)) # Convertir la liste de listes en np.array
train_y = np.array(list(train_y)) # Convertir la liste de listes en np.array

print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# Correction: 'lr' est déprécié, utiliser 'learning_rate'
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
# Correction: model.save() ne prend pas 'hist' comme deuxième argument
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chat-acceuil-model.h5') # Sauvegarder le modèle

print("model created")