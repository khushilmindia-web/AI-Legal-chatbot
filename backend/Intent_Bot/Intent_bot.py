import random
import json
import pickle

import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from backend.paths import INTENTS_PATH, MODEL_DIR

# from keras.models import load_model
import tensorflow as tf
# from tensorflow.keras.models import load_model

class IntentBot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(INTENTS_PATH.read_text(encoding="utf-8"))
        self.words = pickle.load(open(MODEL_DIR / "Intents" / "words.pkl", 'rb'))
        self.classes = pickle.load(open(MODEL_DIR / "Intents" / "classes.pkl", 'rb'))

        # self.model = load_model("Model/Intents/Legalchatbot.h5")
        self.model = tf.keras.models.load_model(MODEL_DIR / "Intents" / "Legalchatbot.h5")

    def _safe_tokenize(self, sentence):
        try:
            return nltk.word_tokenize(sentence)
        except LookupError:
            return sentence.split()

    def _safe_lemmatize(self, word):
        try:
            return self.lemmatizer.lemmatize(word.lower())
        except LookupError:
            return word.lower()
        
    def clean_up_sentence(self, sentence):
        sentence_words = self._safe_tokenize(sentence)
        sentence_words = [self._safe_lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0]*len(self.words)

        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]), verbose=0)[0]

        ERROR_THRESHOLD = 0.25
        results = [(i, float(r)) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []

        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': r[1]})
        return return_list

    def get_response(self, user_message):
        ints = self.predict_class(user_message)

        if len(ints) == 0:
            return "Sorry, I didn't understand that."
        tag = ints[0]['intent']

        for i in self.intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    
    def get_response_with_confidence(self, user_message):
        ints = self.predict_class(user_message)
        
        if len(ints) == 0:
            return "Sorry, I didn't understand that.", 0.0

        top_intent = ints[0]
        tag = top_intent['intent']
        confidence = top_intent['probability']

        for i in self.intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses']), confidence
        
        return "Sorry, I didn't understand that.", confidence
    print("Executed!")
    
