import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import re
import emoji

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[^A-Za-z\u0900-\u097F\s]", "", text)  # includes Marathi
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Load Marathi Model
# -----------------------------
@st.cache_resource
def load_model():
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
    return tokenizer, model, labels, device

tokenizer, model, labels, device = load_model()

# -----------------------------
# Prediction
# -----------------------------
def predict_proba(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Marathi Hate Speech Detector", layout="wide")
st.title("🧠 Marathi Hate Speech Detector with LIME")

text = st.text_area("✏️ Enter text to analyze:", height=150)

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

        exp = explainer.explain_instance(
            text_instance=cleaned_text,
            classifier_fn=predict_proba,
            num_features=8,
            num_samples=100
        )

        st.components.v1.html(exp.as_html(), height=500, scrolling=True)

st.markdown("---")
st.caption("Built using fine-tuned Marathi BERT model for hate speech detection.")
