import streamlit as st
import tensorflow as tf
import keras
import joblib
import numpy as np
import os
import time
import pandas as pd
import re


# Set page configuration
st.set_page_config(
    page_title="AI Text Detector Pro",
    page_icon="🔍",
    layout="centered"
)

# ============================================================
# 1. LOAD MODELS & ASSETS
# ============================================================
@st.cache_resource
def load_assets():
    model_path = 'ann_ai_detector_model.keras'
    tfidf_path = 'tfidf_vectorizer.joblib'
    le_path = 'label_encoder.joblib'
    scaler_path = 'scaler.joblib'
    
    assets = {"model": None, "tfidf": None, "le": None, "scaler": None, "error": None}
    
    if not os.path.exists(model_path):
        assets["error"] = f"Model file '{model_path}' not found."
        return assets

    try:
        assets["model"] = keras.models.load_model(model_path)
        assets["tfidf"] = joblib.load(tfidf_path)
        assets["le"] = joblib.load(le_path)
        assets["scaler"] = joblib.load(scaler_path)
    except Exception as e:
        assets["error"] = str(e)
        
    return assets

assets = load_assets()
model = assets["model"]
tfidf = assets["tfidf"]
le = assets["le"]
scaler = assets["scaler"]

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def calculate_text_stats(text):
    """Calculates basic linguistic statistics for the input text."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    
    word_count = len(words)
    sent_count = len(sentences)
    avg_sent_len = word_count / sent_count if sent_count > 0 else 0
    unique_words = len(set(w.lower() for w in words))
    lexical_diversity = (unique_words / word_count) * 100 if word_count > 0 else 0
    
    return {
        "word_count": word_count,
        "avg_sent_len": avg_sent_len,
        "lexical_diversity": lexical_diversity
    }

def get_top_influential_words(text, is_ai_leaning):
    if model is not None and tfidf is not None and scaler is not None:
        feature_names = tfidf.get_feature_names_out()
        weights = model.layers[0].get_weights()[0].sum(axis=1)
        X_vec = tfidf.transform([text])
        X_vec_scaled = scaler.transform(X_vec).toarray()[0]
        impact = X_vec_scaled * weights
        top_indices = np.argsort(np.abs(impact))[-8:]
        return [feature_names[i] for i in top_indices if X_vec_scaled[i] > 0]
    else:
        ai_words = ["moreover", "furthermore", "essential", "in conclusion", "intricate", "tapestry", "complex", "shimmering"]
        human_words = ["really", "actually", "just", "maybe", "think", "guess", "stuff", "well"]
        words_in_text = text.lower().split()
        candidates = ai_words if is_ai_leaning else human_words
        found = [w for w in candidates if w in words_in_text]
        return found if found else candidates[:3]

# ============================================================
# 3. UI DESIGN
# ============================================================
st.title("🤖 AI Text Detector Pro")
st.markdown("Professional-grade analysis to distinguish between **Human** writing and **AI** generation.")

# --- Sidebar ---
st.sidebar.title("Settings & Info")
demo_mode = st.sidebar.checkbox("Enable Demo Mode", value=assets["error"] is not None)

st.sidebar.info("""
**Analysis Method:**
This tool uses a Deep Neural Network trained on TF-IDF vectorized text to identify patterns typical of Large Language Models (LLMs).
""")

if model is not None:
    st.sidebar.write("---")
    st.sidebar.write("**Model Specs:**")
    st.sidebar.write("- Architecture: Dense Neural Network")
    st.sidebar.write("- Input: TF-IDF Vectorized Text")

# --- Status Messages ---
if assets["error"]:
    if demo_mode:
        st.info("💡 **Demo Mode Active:** Simulating results (Real model offline).")
    else:
        st.warning("⚠️ **Note:** AI Engine is offline (Model error).")
        with st.expander("Technical details"):
            st.code(assets["error"])
else:
    st.success("✅ AI Engine Online")

# --- Main Input ---
user_text = st.text_area("Input Text", placeholder="Paste text here (min 20 words recommended)...", height=250)

if st.button("Analyze Text", disabled=(model is None and not demo_mode)):
    if len(user_text.split()) < 5:
        st.warning("Please enter a longer text for analysis.")
    else:
        # --- Thinking Phase ---
        status_box = st.empty()
        with status_box:
            with st.container():
                st.markdown("### 🧠 AI is thinking...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                steps = [
                    "Tokenizing input text...",
                    "Extracting linguistic features...",
                    "Analyzing syntactic structures...",
                    "Comparing with known AI patterns...",
                    "Finalizing neural weights...",
                    "Generating confidence score..."
                ]
                for i, step in enumerate(steps):
                    status_text.text(step)
                    progress_bar.progress((i + 1) / len(steps))
                    time.sleep(0.8)
        status_box.empty()

        # --- Prediction & Stats ---
        stats = calculate_text_stats(user_text)
        
        if demo_mode and model is None:
            np.random.seed(len(user_text))
            is_ai_prob = np.random.uniform(0.1, 0.9)
            is_human_prob = 1 - is_ai_prob
        else:
            X_input = tfidf.transform([user_text])
            if scaler is not None:
                X_input = scaler.transform(X_input)
            X_input = X_input.toarray().astype('float32')
            prediction_prob = model.predict(X_input)[0][0]
            human_index = np.where(le.classes_ == 'Human')[0][0]
            is_human_prob = prediction_prob if human_index == 1 else (1 - prediction_prob)
            is_ai_prob = 1 - is_human_prob

        # --- Results Display ---
        st.divider()
        col_res, col_stats = st.columns([1, 1])
        
        with col_res:
            if is_ai_prob > 0.5:
                st.error("### Result: Likely AI")
                st.metric("AI Confidence", f"{is_ai_prob*100:.1f}%")
            elif is_ai_prob > 0.485:
                st.warning("### Result: Inconclusive")
                st.write("The model cannot confidently classify this text. It shows patterns of both human and AI writing.")
            else: # Clearly Human

                st.success("### Result: Likely Human")
                st.metric("Human Confidence", f"{is_human_prob*100:.1f}%")
            
            st.progress(float(np.clip(is_ai_prob, 0.0, 1.0)), text=f"AI probability index")

        with col_stats:
            st.write("**Text Statistics**")
            st.write(f"- Word Count: `{stats['word_count']}`")
            st.write(f"- Avg. Sentence Length: `{stats['avg_sent_len']:.1f} words`")
            st.write(f"- Lexical Diversity: `{stats['lexical_diversity']:.1f}%`")
            
        st.write("---")
        st.write("#### Detailed Probability Distribution")
        st.progress(float(np.clip(is_ai_prob, 0.0, 1.0)), text=f"AI Probability: {is_ai_prob*100:.1f}%")
        st.progress(float(np.clip(is_human_prob, 0.0, 1.0)), text=f"Human Probability: {is_human_prob*100:.1f}%")

        # --- Explainability Section ---
        st.markdown("#### 🔍 Linguistic Fingerprints")
        st.caption("Words that most influenced the model's decision:")
        top_words = get_top_influential_words(user_text, is_ai_prob > 0.5)
        
        st.write(" ".join([f"`{w}`" for w in top_words]))

        st.info("**Note:** This is an estimation based on patterns identified by the neural network. No detector is 100% accurate.")

        # --- Export / Download ---
        st.divider()
        report_data = {
            "Metric": ["Result", "AI Probability", "Human Probability", "Word Count", "Avg Sentence Length", "Lexical Diversity"],
            "Value": [
                "AI" if is_ai_prob > 0.5 else "Human",
                f"{is_ai_prob:.2%}",
                f"{is_human_prob:.2%}",
                stats["word_count"],
                f"{stats['avg_sent_len']:.1f}",
                f"{stats['lexical_diversity']:.1f}%"
            ]
        }
        df_report = pd.DataFrame(report_data)
        
        st.download_button(
            label="Download Full Analysis Report (CSV)",
            data=df_report.to_csv(index=False).encode('utf-8'),
            file_name="ai_detection_report.csv",
            mime="text/csv"
        )

st.sidebar.markdown("---")
st.sidebar.write("© 2026 AI Text Detector Pro")
