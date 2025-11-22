import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PegasusForConditionalGeneration, PegasusTokenizer
import nltk
import numpy as np

# --- Configuration & Setup ---
st.set_page_config(page_title="TrueText - AI Detector", layout="wide", initial_sidebar_state="collapsed")

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# --- Custom CSS ---
st.markdown("""
<style>
    /* Global Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --primary: #6d28d9;
        --primary-dark: #5b21b6;
        --secondary: #1e1b4b;
        --background: #0f0e17;
        --surface: #1e1e24;
        --text: #e2e8f0;
        --text-muted: #94a3b8;
        --accent: #8b5cf6;
    }

    .stApp {
        background-color: var(--background);
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom Header */
    .custom-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background: rgba(15, 14, 23, 0.8);
        backdrop-filter: blur(10px);
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .logo {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .logo span {
        color: var(--accent);
    }

    /* Hero Section */
    .hero-container {
        margin-top: 60px;
        background: linear-gradient(135deg, #2e1065 0%, #0f0e17 100%);
        padding: 4rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(109, 40, 217, 0.3);
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(to right, #fff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: var(--text-muted);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Input Area */
    .stTextArea textarea {
        background-color: rgba(30, 30, 36, 0.8) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: var(--text) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(to right, var(--primary), var(--accent)) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
    }
    
    .secondary-btn button {
        background: transparent !important;
        border: 1px solid var(--accent) !important;
    }

    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 3rem 0;
    }

    .feature-card {
        background: var(--surface);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        transition: transform 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        border-color: var(--accent);
    }

    .feature-icon {
        background: rgba(139, 92, 246, 0.1);
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--accent);
        margin-bottom: 1rem;
    }

    /* Results */
    .result-box {
        background: var(--surface);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.05);
        margin-top: 2rem;
    }

    .highlight-fake {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        padding: 2px 4px;
        border-radius: 4px;
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-muted);
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 4rem;
    }
</style>

<div class="custom-header">
    <div class="logo">✨ True<span>Text</span></div>
    <div style="color: var(--text-muted);">v1.0</div>
</div>
""", unsafe_allow_html=True)

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
    fake_count = 0
    results = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
            
        label = "Fake" if predicted_class_id == 1 else "Real"
        results.append((sentence, label))
        if label == "Fake":
            fake_count += 1

    score = (fake_count / len(sentences)) * 100 if sentences else 0
    return score, results

def humanize_text(text, tokenizer, model, device):
    if not text or not tokenizer or not model:
        return ""
    
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
def render_circular_progress(percentage):
    color = "#ef4444" if percentage > 50 else "#22c55e"
    html_code = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
        <div style="
            position: relative;
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: conic-gradient({color} {percentage}%, #333 0);
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 0 30px rgba(0,0,0,0.5);
            margin-bottom: 1rem;
        ">
            <div style="
                width: 170px;
                height: 170px;
                background: #1e1e24;
                border-radius: 50%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            ">
                <span style="font-size: 3.5rem; font-weight: 800; color: {color};">{percentage:.0f}%</span>
                <span style="font-size: 1rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">AI Content</span>
            </div>
        </div>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def render_highlighted_text(results):
    html_text = ""
    for sentence, label in results:
        if label == "Fake":
            html_text += f'<span class="highlight-fake">{sentence}</span> '
        else:
            html_text += f'<span style="color: #e2e8f0;">{sentence}</span> '
    
    st.markdown(
        f"""
        <div style="
            padding: 1.5rem; 
            background: #27272a; 
            border-radius: 12px; 
            line-height: 1.8; 
            font-size: 1.05rem;
            height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(255,255,255,0.05);
        ">
            {html_text}
        </div>
        """, 
        unsafe_allow_html=True
    )

# --- Main Layout ---

# Load models
det_tokenizer, det_model, det_device = load_detection_model()
hum_tokenizer, hum_model, hum_device = load_humanizer_model()

# Hero Section
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Detect AI-Generated Text<br>with Confidence</h1>
    <p class="hero-subtitle">For educators, editors, and content creators. Ensure authenticity in seconds.</p>
</div>
""", unsafe_allow_html=True)

# Main Input Area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📄 Paste Text")
    input_text = st.text_area("Input Text", height=300, label_visibility="collapsed", placeholder="Paste your text here to analyze...")
    
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        detect_btn = st.button("🔍 Analyze Text", use_container_width=True)
    with btn_col2:
        humanize_btn = st.button("✨ Humanize", use_container_width=True)

with col2:
    if detect_btn and input_text:
        with st.spinner("🔮 Analyzing patterns..."):
            score, results = detect_ai_content(input_text, det_tokenizer, det_model, det_device)
            
            st.markdown("### 📊 Analysis Results")
            render_circular_progress(score)
            
            st.markdown("### 📝 Detailed Breakdown")
            render_highlighted_text(results)
            
    elif humanize_btn and input_text:
        with st.spinner("✨ Rewriting content..."):
            humanized_text = humanize_text(input_text, hum_tokenizer, hum_model, hum_device)
            
            st.markdown("### ✨ Humanized Result")
            st.success("Text successfully rewritten!")
            st.text_area("Output", value=humanized_text, height=400, label_visibility="collapsed")
            
    else:
        # Default State / Landing Page Content
        st.markdown("### Why Choose TrueText?")
        st.markdown("""
        <div class="feature-grid" style="margin: 0; grid-template-columns: 1fr;">
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h4 style="margin: 0 0 0.5rem 0;">Pinpoint Accuracy</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">Leverage state-of-the-art RoBERTa models for precise AI detection.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h4 style="margin: 0 0 0.5rem 0;">Instant Results</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">Get comprehensive analysis and scoring in just a few seconds.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔒</div>
                <h4 style="margin: 0 0 0.5rem 0;">Privacy First</h4>
                <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">Your data is yours. We never store or share your analyzed content.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="custom-footer">
    <p>TrueText AI Detector &copy; 2024. All rights reserved.</p>
    <p style="margin-top: 1rem;">
        <a href="https://github.com/RnR-io/TrueText" target="_blank" style="color: #8b5cf6; text-decoration: none; display: inline-flex; align-items: center; gap: 0.5rem;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark-Light-64px.png" width="20" height="20">
            View on GitHub
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
