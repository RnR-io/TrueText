import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PegasusForConditionalGeneration, PegasusTokenizer
import nltk
import numpy as np

# --- Configuration & Setup ---
st.set_page_config(page_title="AI Essay Detector & Humanizer", layout="wide")

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# --- Model Loading (Cached) ---
@st.cache_resource
def load_detection_model():
    model_name = "roberta-base-openai-detector"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading detection model: {e}")
        return None, None, None

@st.cache_resource
def load_humanizer_model():
    model_name = "tuner007/pegasus_paraphrase"
    try:
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading humanizer model: {e}")
        return None, None, None

# --- Core Logic ---

def detect_ai_content(text, tokenizer, model, device):
    if not text or not tokenizer or not model:
        return 0, []

    sentences = nltk.sent_tokenize(text)
    fake_sentences = []
    
    # Process sentences in batches for efficiency could be added, but simple loop for now as per req
    # The model outputs logits. We'll use the logic often associated with this model:
    # Label 0: Real, Label 1: Fake (Check specific model config if needed, usually roberta-base-openai-detector follows this)
    
    fake_count = 0
    results = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
            
        # roberta-base-openai-detector: 0 -> Real, 1 -> Fake
        label = "Fake" if predicted_class_id == 1 else "Real"
        results.append((sentence, label))
        if label == "Fake":
            fake_count += 1

    score = (fake_count / len(sentences)) * 100 if sentences else 0
    return score, results

def humanize_text(text, tokenizer, model, device):
    if not text or not tokenizer or not model:
        return ""
    
    # Split text into paragraphs to preserve structure
    paragraphs = text.split('\n\n')
    humanized_paragraphs = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            humanized_paragraphs.append("")
            continue
            
        sentences = nltk.sent_tokenize(paragraph)
        humanized_sentences = []
        
        for sentence in sentences:
            batch = tokenizer([sentence], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(device)
            with torch.no_grad():
                translated = model.generate(**batch, max_length=60, num_beams=10, num_return_sequences=1, temperature=1.5)
            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
            humanized_sentences.append(tgt_text[0])
        
        humanized_paragraphs.append(" ".join(humanized_sentences))
        
    return "\n\n".join(humanized_paragraphs)

# --- UI Components ---

def render_gradient_circle(percentage):
    # Determine color based on percentage (Green -> Red)
    # 0% = Green (0, 255, 0), 100% = Red (255, 0, 0)
    # Simple linear interpolation
    r = int((percentage / 100) * 255)
    g = int((1 - (percentage / 100)) * 255)
    color = f"rgb({r}, {g}, 0)"
    
    html_code = f"""
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
        <div style="
            position: relative;
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: conic-gradient({color} {percentage}%, #e0e0e0 0);
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="
                width: 120px;
                height: 120px;
                background: white;
                border-radius: 50%;
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 24px;
                font-weight: bold;
                color: #333;
            ">
                {percentage:.1f}%
            </div>
        </div>
    </div>
    <div style="text-align: center; font-weight: bold; margin-bottom: 20px;">AI Probability</div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def render_highlighted_text(results):
    html_text = ""
    for sentence, label in results:
        if label == "Fake":
            html_text += f'<span style="background-color: #ffcccc; padding: 2px; border-radius: 3px;">{sentence}</span> '
        else:
            html_text += f"{sentence} "
    
    st.markdown(f'<div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; height: 300px; overflow-y: auto; background-color: #f9f9f9; color: black;">{html_text}</div>', unsafe_allow_html=True)


# --- Main Application ---

st.title("🕵️ AI Essay Detector & Humanizer")
st.markdown("Analyze text for AI generation and humanize it to bypass detection.")

# Load models
det_tokenizer, det_model, det_device = load_detection_model()
hum_tokenizer, hum_model, hum_device = load_humanizer_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Text")
    input_text = st.text_area("Paste your essay here...", height=300)
    
    col_btns = st.columns(2)
    with col_btns[0]:
        detect_btn = st.button("🔍 Detect AI", use_container_width=True)
    with col_btns[1]:
        humanize_btn = st.button("✨ Humanize", use_container_width=True)

with col2:
    st.subheader("Analysis & Result")
    
    if detect_btn and input_text:
        with st.spinner("Analyzing text..."):
            score, results = detect_ai_content(input_text, det_tokenizer, det_model, det_device)
            render_gradient_circle(score)
            st.markdown("### Detailed Analysis")
            render_highlighted_text(results)
            
    elif humanize_btn and input_text:
        with st.spinner("Humanizing text..."):
            humanized_text = humanize_text(input_text, hum_tokenizer, hum_model, hum_device)
            st.success("Humanization Complete!")
            st.text_area("Humanized Output", value=humanized_text, height=300)
            
    else:
        st.info("Enter text and choose an action to see results here.")

st.markdown("---")
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Powered by RoBERTa & Pegasus | Built with Streamlit</p>
        <p>
            <a href="https://github.com/RnR-io/TrueText" target="_blank" style="text-decoration: none; color: #666;">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" height="20" style="vertical-align: middle; margin-right: 5px;">
                View on GitHub
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
