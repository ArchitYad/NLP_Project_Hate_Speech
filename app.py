import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import re
import emoji

# -----------------------------
# 1. Text Cleaning
# -----------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# 2. Load English Model
# -----------------------------
@st.cache_resource
def load_english_model():
    model_path = "english_pt"  # folder with tokenizer + config + .pt
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load config and initialize empty model
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_config(config)

    # Load .pt weights manually
    state_dict = torch.load(f"{model_path}/english_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_english_model()
labels = ["hatespeech", "normal", "offensive"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 3. Prediction Function
# -----------------------------
def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.set_page_config(page_title="English Hate Speech Detector", layout="wide")

st.title("🧠 English Hate Speech Detector with LIME")
st.write("Detects hate speech in English text using a fine-tuned BERT model and explains results with LIME.")

text = st.text_area("Enter text to analyze:", height=150)

if st.button("Analyze"):
    if not text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned_text = clean_text(text)
        st.markdown("### 🧹 Cleaned Text")
        st.write(cleaned_text)

        probs = predict_proba([cleaned_text])
        pred = np.argmax(probs, axis=1)[0]

        st.subheader(f"🧩 Prediction: **{labels[pred]}**")
        st.write(f"Confidence: {probs[0][pred]*100:.2f}%")

        st.markdown("### 🔢 Confidence per class")
        st.bar_chart(dict(zip(labels, probs[0])))

        st.markdown("### 🧠 LIME Explanation")
        explainer = LimeTextExplainer(class_names=labels)

        def predict_fn(texts):
            return predict_proba(texts)

        exp = explainer.explain_instance(
            text_instance=cleaned_text,
            classifier_fn=predict_fn,
            num_features=8,
            num_samples=100
        )

        st.components.v1.html(exp.as_html(), height=500, scrolling=True)

st.markdown("---")
st.caption("Built using BERT model fine-tuned for English hate speech detection.")
