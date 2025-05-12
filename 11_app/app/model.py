# app/model.py

import joblib
from pathlib import Path

VEC_PATH = Path("saved_model/vectorizer.pkl")
MODEL_PATH = Path("saved_model/model.pkl")

vectorizer = joblib.load(VEC_PATH)
model = joblib.load(MODEL_PATH)
