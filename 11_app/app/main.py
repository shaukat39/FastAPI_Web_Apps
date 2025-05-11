# app/main.py

from fastapi import FastAPI
from app.schema import SentimentRequest, SentimentResponse
from app.model import model, vectorizer
from app.utils import preprocess
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    text = preprocess(request.text)
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    confidence = np.max(model.predict_proba(features))

    return SentimentResponse(
        sentiment="positive" if prediction == 1 else "negative",
        confidence=round(float(confidence), 2)
    )
@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running."}