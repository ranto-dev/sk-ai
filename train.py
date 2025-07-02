import json
import numpy as np
import nltk
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')

# 1. Charger le dataset
with open("dataset.json") as file:
    data = json.load(file)

# 2. Préparation des données
sentences = []
labels = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

# 3. Encodage des étiquettes
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Sauvegarde du label_encoder
with open("label_encoder.pickle", "wb") as f:
    pickle.dump(label_encoder, f)

# 4. Tokenisation
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# Sauvegarde du tokenizer
with open("tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f)

sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding="post")

# 5. Construction du modèle
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=padded_sequences.shape[1]),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dense(len(set(encoded_labels)), activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# 6. Entraînement
model.fit(padded_sequences, np.array(encoded_labels), epochs=1000)

# 7. Sauvegarde du modèle
model.save("my_model.h5")
print("✅ Modèle entraîné et sauvegardé dans model.h5")
