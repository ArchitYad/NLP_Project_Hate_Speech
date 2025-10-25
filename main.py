from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import numpy as np
import re
import emoji
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

app = FastAPI(title="Multilingual Hate Speech Detector")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# 1. Text Cleaning
# -----------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[^A-Za-z\u0900-\u097F\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# 2. Model Loader (cached)
# -----------------------------
_loaded_models = {}

def load_model(lang: str):
    if lang in _loaded_models:
        return _loaded_models[lang]

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

    _loaded_models[lang] = (tokenizer, model, labels, device)
    return _loaded_models[lang]


# -----------------------------
# 3. Prediction
# -----------------------------
def predict_proba(texts, tokenizer, model, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()


# -----------------------------
# 4. Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, text: str = Form(...), lang: str = Form(...)):
    cleaned_text = clean_text(text)
    tokenizer, model, labels, device = load_model(lang)

    probs = predict_proba([cleaned_text], tokenizer, model, device)
    pred_idx = np.argmax(probs, axis=1)[0]
    prediction = labels[pred_idx]
    confidence = probs[0][pred_idx] * 100

    explainer = LimeTextExplainer(class_names=labels)
    explanation = explainer.explain_instance(
        text_instance=cleaned_text,
        classifier_fn=lambda x: predict_proba(x, tokenizer, model, device),
        num_features=8,
        num_samples=100
    ).as_html()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "text": text,
            "cleaned_text": cleaned_text,
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%",
            "labels": labels,
            "probs": list(zip(labels, probs[0])),
            "explanation": explanation,
            "lang": lang,
        },
    )
