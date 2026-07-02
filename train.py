"""
train.py — Trains ChatNad's intent classification model.

Approach: Classic ML (not deep learning)
  1. Load intents.json (patterns -> tag)
  2. Preprocess text: lowercase, tokenize, lemmatize
  3. Vectorize with TF-IDF
  4. Train a classifier to predict intent tag from user text
  5. Save vectorizer + model + intents to disk for use in the app
"""

import json
import string
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, tokenize, lemmatize."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return " ".join(tokens)


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(clean_text(pattern))
            labels.append(intent["tag"])
    return texts, labels, data


def main():
    texts, labels, raw_data = load_data("data/intents.json")
    print(f"Loaded {len(texts)} training examples across {len(set(labels))} intents.")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(texts)
    y = labels

    model = SVC(kernel="linear", probability=True, C=1.0)

    # Dataset is small, so a single train/test split is noisy.
    # Use stratified k-fold cross-validation for a more honest accuracy estimate.
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf)
    print("\n--- Cross-validation results ---")
    print(f"Fold accuracies: {[round(s, 2) for s in scores]}")
    print(f"Mean accuracy: {scores.mean():.2f}")

    # Fit final model on ALL data for deployment (standard practice for small,
    # curated intent datasets where every example matters)
    model.fit(X, y)

    joblib.dump(model, "model/chatnad_model.pkl")
    joblib.dump(vectorizer, "model/chatnad_vectorizer.pkl")
    with open("model/intents_processed.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    print("\nSaved model, vectorizer, and intents to /model")


if __name__ == "__main__":
    main()
