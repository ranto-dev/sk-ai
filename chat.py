import json
import random
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Chargement des fichiers nécessaires
model = load_model("model.h5")

with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

with open("intents.json") as f:
    data = json.load(f)

# 2. Fonction de réponse du chatbot
def chatbot_response(message):
    sequence = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequence, maxlen=model.input_shape[1], padding='post')
    prediction = model.predict(padded, verbose=0)
    tag = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# 3. Boucle de conversation
print("🤖 Chatbot en ligne (tapez 'quit' pour quitter)")
while True:
    msg = input("Vous: ")
    if msg.lower() in ["quit", "exit", "q"]:
        print("Bot: À bientôt !")
        break
    response = chatbot_response(msg)
    print("Bot:", response)
