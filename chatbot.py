"""
chatbot.py — Core ChatNad logic: loads the trained model and generates replies.
"""

import json
import random
import string
import joblib
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

CONFIDENCE_THRESHOLD = 0.35  # below this, ChatNad admits it doesn't understand


def clean_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return " ".join(tokens)


class ChatNad:
    def __init__(self, model_dir="model"):
        self.model = joblib.load(f"{model_dir}/chatnad_model.pkl")
        self.vectorizer = joblib.load(f"{model_dir}/chatnad_vectorizer.pkl")
        with open(f"{model_dir}/intents_processed.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.responses = {
            intent["tag"]: intent["responses"] for intent in data["intents"]
        }

    def predict_intent(self, text: str):
        cleaned = clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(vec)[0]
        classes = self.model.classes_
        best_idx = probs.argmax()
        return classes[best_idx], probs[best_idx]

    def get_response(self, text: str) -> str:
        if not text or not text.strip():
            return "Say something and I'll do my best to respond!"

        tag, confidence = self.predict_intent(text)

        if confidence < CONFIDENCE_THRESHOLD or tag == "fallback":
            tag = "fallback"

        return random.choice(self.responses.get(tag, self.responses["fallback"]))


if __name__ == "__main__":
    bot = ChatNad()
    print("ChatNad is ready! Type 'quit' to exit.\n")
    while True:
        msg = input("You: ")
        if msg.lower() in ("quit", "exit"):
            print("ChatNad: Goodbye!")
            break
        reply = bot.get_response(msg)
        print(f"ChatNad: {reply}")
