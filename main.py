from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np
import re
import emoji

# -----------------------------
# App & Template Setup
# -----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Global Model Cache
# -----------------------------
current_model = None
current_tokenizer = None
current_labels = None
current_device = None
current_lang = None

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text, lang="english"):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = emoji.replace_emoji(text, replace="")
    if lang == "english":
        text = re.sub(r"[^A-Za-z\s]", "", text)
    else:
        text = re.sub(r"[^A-Za-z\u0900-\u097F\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Lazy Model Loader
# -----------------------------
def load_model(lang: str):
    global current_model, current_tokenizer, current_labels, current_device, current_lang

    if current_lang == lang and current_model is not None:
        # Use already loaded model
        return current_tokenizer, current_model, current_labels, current_device

    # Remove old model from memory
    if current_model is not None:
        del current_model
        torch.cuda.empty_cache()

    if lang == "english":
        model_path = "english_pt"
        model_file = "english_model.pt"
        labels = ["hatespeech", "normal", "offensive"]
    else:
        model_path = "marathi_pt"
        model_file = "marathi_model.pt"
        labels = ["hate", "not", "offn", "prfn"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_config(config)
    state_dict = torch.load(f"{model_path}/{model_file}", map_location="cpu")
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Update global cache
    current_model = model
    current_tokenizer = tokenizer
    current_labels = labels
    current_device = device
    current_lang = lang

    return tokenizer, model, labels, device

# -----------------------------
# Prediction Function
# -----------------------------
def predict_proba(texts, tokenizer, model, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
def analyze(request: Request, text: str = Form(...), language: str = Form(...)):
    stages = []

    # Stage 1: Receive text
    stages.append({"stage": "Received text", "value": text})

    # Stage 2: Load model
    stages.append({"stage": f"Loading {language} model...", "value": None})
    tokenizer, model, labels, device = load_model(language.lower())

    # Stage 3: Clean text
    cleaned_text = clean_text(text, lang=language.lower())
    stages.append({"stage": "Cleaned text", "value": cleaned_text})

    # Stage 4: Prediction
    probs = predict_proba([cleaned_text], tokenizer, model, device)
    pred = np.argmax(probs, axis=1)[0]
    stages.append({"stage": "Prediction", "value": labels[pred]})
    stages.append({"stage": "Confidence per class", "value": dict(zip(labels, probs[0]))})

    # Stage 5: LIME explanation
    explainer = LimeTextExplainer(class_names=labels)
    def predict_fn(texts):
        return predict_proba(texts, tokenizer, model, device)
    exp = explainer.explain_instance(cleaned_text, predict_fn, num_features=8, num_samples=100)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "stages": stages,
        "lime_html": exp.as_html(),
        "labels": labels,
        "pred_label": labels[pred],
        "probs": probs[0],
        "cleaned_text": cleaned_text,
        "original_text": text,
        "language": language
    })
