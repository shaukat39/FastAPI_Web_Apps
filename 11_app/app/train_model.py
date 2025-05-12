# app/train_model.py

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import re

def clean_text(text):
    text = re.sub(r"<.*?>", "", text.lower())
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

def train_and_save_model():
    df = pd.read_csv("data/IMDB Dataset.csv")
    df['review'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    X_train, _, y_train, _ = train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('nb', MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline.named_steps['tfidf'], 'saved_model/vectorizer.pkl')
    joblib.dump(pipeline.named_steps['nb'], 'saved_model/model.pkl')

if __name__ == "__main__":
    train_and_save_model()
