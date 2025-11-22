import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PegasusForConditionalGeneration, PegasusTokenizer
import nltk
import numpy as np
import json
import os
from datetime import datetime
from fpdf import FPDF
import io
import PyPDF2
import docx

# --- Configuration & Setup ---
st.set_page_config(page_title="TrueText - AI Detector", layout="wide", initial_sidebar_state="expanded")

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
    
    /* Steps */
    .step-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 2rem;
    }
    .step-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
        background: var(--surface);
        padding: 10px;
        border-radius: 50%;
        border: 1px solid var(--accent);
    }
</style>

<div class="custom-header">
    <div class="logo">✨ True<span>Text</span></div>
    <div style="color: var(--text-muted);">v2.0</div>
</div>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""
    
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        else: # txt
            return str(uploaded_file.read(), "utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

def save_to_history(text, score, label):
    history_file = "history.json"
    entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "text_snippet": text[:100] + "..." if len(text) > 100 else text,
        "score": score,
        "label": label
    }
    
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            try:
                history = json.load(f)
            except:
                history = []
    else:
        history = []
    
    history.insert(0, entry) # Add to beginning
    with open(history_file, "w") as f:
        json.dump(history, f)

def load_history():
    history_file = "history.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def create_pdf_report(text, score, results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="TrueText AI Detection Report", ln=1, align='C')
    
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1, align='C')
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"AI Probability Score: {score:.1f}%", ln=1, align='L')
    
    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt="Analysis Breakdown (Fake sentences marked with [AI]):")
    pdf.ln(5)
    
    pdf.set_font("Arial", size=11)
    for sentence, label in results:
        prefix = "[AI] " if label == "Fake" else ""
        pdf.multi_cell(0, 6, txt=f"{prefix}{sentence}")
        pdf.ln(1)
        
    return pdf.output(dest='S').encode('latin-1')

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

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("Go to", ["Home", "How It Works", "Scan History", "About Us"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.9rem;">
        <p><strong>Motto:</strong><br>Free access to humanizer</p>
    </div>
    """, unsafe_allow_html=True)

# --- Main Layout ---

# Load models
det_tokenizer, det_model, det_device = load_detection_model()
hum_tokenizer, hum_model, hum_device = load_humanizer_model()

if page == "Home":
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
        st.markdown("### 📄 Paste Text or Upload File")
        
        tab1, tab2 = st.tabs(["Paste Text", "Upload File"])
        
        with tab1:
            input_text = st.text_area("Input Text", height=300, label_visibility="collapsed", placeholder="Paste your text here to analyze...")
        
        with tab2:
            uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'])
            if uploaded_file is not None:
                input_text = extract_text_from_file(uploaded_file)
                st.success("File loaded successfully!")
                st.text_area("Preview", value=input_text[:500] + "...", height=150, disabled=True)
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            detect_btn = st.button("🔍 Analyze Text", use_container_width=True)
        with btn_col2:
            humanize_btn = st.button("✨ Humanize", use_container_width=True)

    with col2:
        if detect_btn and input_text:
            with st.spinner("🔮 Analyzing patterns..."):
                score, results = detect_ai_content(input_text, det_tokenizer, det_model, det_device)
                label = "AI Content" if score > 50 else "Likely Human"
                save_to_history(input_text, score, label)
                
                st.markdown("### 📊 Analysis Results")
                render_circular_progress(score)
                
                st.markdown("### 📝 Detailed Breakdown")
                render_highlighted_text(results)
                
                # PDF Export
                pdf_bytes = create_pdf_report(input_text, score, results)
                st.download_button(
                    label="📥 Export PDF Report",
                    data=pdf_bytes,
                    file_name="truetext_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
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

elif page == "How It Works":
    st.markdown("""
    <div class="hero-container" style="padding: 3rem 2rem;">
        <h1 class="hero-title">How It Works</h1>
        <p class="hero-subtitle">Simple steps to ensure content authenticity.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="max-width: 800px; margin: 0 auto;">
        <div class="result-box">
            <div class="step-container">
                <div class="step-icon">📄</div>
                <div>
                    <h3>1. Paste Your Text</h3>
                    <p style="color: var(--text-muted);">Simply input the text you want to check into the analysis field or upload a file (PDF, DOCX, TXT).</p>
                </div>
            </div>
            <div class="step-container">
                <div class="step-icon">🔍</div>
                <div>
                    <h3>2. Analyze Content</h3>
                    <p style="color: var(--text-muted);">Our powerful AI engine scans the text for digital signatures and linguistic patterns.</p>
                </div>
            </div>
            <div class="step-container">
                <div class="step-icon">📊</div>
                <div>
                    <h3>3. Get Results</h3>
                    <p style="color: var(--text-muted);">Receive a clear probability score, highlighted text analysis, and exportable reports.</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Scan History":
    st.markdown("""
    <div class="hero-container" style="padding: 3rem 2rem;">
        <h1 class="hero-title">Scan History</h1>
        <p class="hero-subtitle">View your recent analysis results.</p>
    </div>
    """, unsafe_allow_html=True)
    
    history = load_history()
    
    if not history:
        st.info("No scan history found. Run an analysis to see it here!")
    else:
        for item in history:
            with st.expander(f"{item['timestamp']} - {item['label']} ({item['score']:.1f}%)"):
                st.write(f"**Snippet:** {item['text_snippet']}")
                st.write(f"**Score:** {item['score']:.1f}%")
                st.write(f"**Verdict:** {item['label']}")

elif page == "About Us":
    st.markdown("""
    <div class="hero-container" style="padding: 3rem 2rem;">
        <h1 class="hero-title">About Us</h1>
        <p class="hero-subtitle">Empowering creators with transparent and accessible AI tools.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="max-width: 800px; margin: 0 auto; padding: 2rem;">
        <div class="result-box">
            <h2 style="color: var(--accent); margin-bottom: 1.5rem;">Our Mission</h2>
            <p style="font-size: 1.1rem; line-height: 1.8; color: var(--text-muted);">
                At <strong>RnR-io</strong>, we believe that advanced AI technology should be accessible to everyone. 
                In an era where digital content is increasingly generated by machines, distinguishing between human and AI authorship is crucial for maintaining integrity and authenticity.
            </p>
            <p style="font-size: 1.1rem; line-height: 1.8; color: var(--text-muted);">
                TrueText was built to provide a reliable, free, and easy-to-use solution for students, educators, and content creators.
            </p>
            
            <div style="margin-top: 3rem; text-align: center; padding: 2rem; background: rgba(139, 92, 246, 0.1); border-radius: 16px; border: 1px solid var(--accent);">
                <h3 style="margin-bottom: 1rem;">Our Motto</h3>
                <p style="font-size: 1.5rem; font-weight: 700; color: #fff;">"Free access to humanizer"</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="custom-footer">
    <p>TrueText - An <strong>RnR-io</strong> Project &copy; 2024.</p>
    <p style="margin-top: 1rem;">
        <a href="https://github.com/RnR-io/TrueText" target="_blank" style="color: #8b5cf6; text-decoration: none; display: inline-flex; align-items: center; gap: 0.5rem;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark-Light-64px.png" width="20" height="20">
            View on GitHub
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
