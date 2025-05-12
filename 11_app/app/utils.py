# app/utils.py

import re

def preprocess(text: str) -> str:
    text = re.sub(r"<.*?>", "", text.lower())
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()
