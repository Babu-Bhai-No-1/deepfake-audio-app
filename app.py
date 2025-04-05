import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from google.generativeai import configure, GenerativeModel

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="🎙️ Deepfake Audio Detection", layout="wide")

# ---------------------- Custom Styling ----------------------
st.markdown("""
    <style>
        body {
            background-color: #0d1117;
            color: white;
        }
        .stApp {
            background-color: #0d1117;
        }
        h1, h2, h3, h4 {
            color: #58a6ff;
        }
        .stTextInput>div>div>input {
            background-color: #161b22;
            color: white;
        }
        .stButton>button {
            background-color: #238636;
            color: white;
        }
        .css-1aumxhk {
            background-color: #161b22;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Gemini API Setup ----------------------
configure(api_key=st.secrets["GEMINI_API_KEY"])
chat_model = GenerativeModel("gemini-pro")
chat = chat_model.start_chat(history=[])

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("cnn_lstm_model_final.h5")
    except TypeError:
        return tf.keras.models.load_model("cnn_lstm_model_final.h5", compile=False)

model_dl = load_model()

# ---------------------- Feature Extraction ----------------------
def extract_features(file):
    y, sr = librosa.load(file, sr=16000)
    y_trimmed, _ = librosa.effects.trim(y)
    mel_spec = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel_spec)
    padded = np.zeros((40, 128))
    if log_mel.shape[1] < 40:
        padded[:log_mel.shape[1], :] = log_mel.T
    else:
        padded = log_mel.T[:40, :]
    return padded

# ---------------------- App UI ----------------------
st.title("🎙️ Deepfake Audio Detection App with Gemini 🤖")
st.markdown("Upload an audio file to detect if it's fake, and ask Gemini anything about deepfake detection or audio forensics.")

col1, col2 = st.columns([1.3, 1])

# 🎧 Audio Detection
with col1:
    st.header("🧪 Audio Detector")
    uploaded_file = st.file_uploader("📂 Upload audio file (wav/mp3/flac)", type=["wav", "mp3", "flac"])

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        features = extract_features(uploaded_file)
        input_data = np.expand_dims(features, axis=0)
        prediction = model_dl.predict(input_data)[0][0]

        label = "🎭 Deepfake Audio" if prediction >= 0.5 else "✅ Real Audio"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** `{confidence * 100:.2f}%`")

        st.subheader("📊 Spectrogram")
        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.specshow(features.T, sr=16000, x_axis="time", y_axis="mel", ax=ax)
        ax.set(title="Log-Mel Spectrogram")
        st.pyplot(fig)

# 💬 Gemini Chat
with col2:
    st.header("💬 Chat with Gemini")
    user_question = st.text_input("Ask a question related to deepfake audio:")

    if st.button("Ask Gemini"):
        if not user_question.strip():
            st.warning("🚨 Please enter a question.")
        else:
            with st.spinner("Gemini is thinking..."):
                try:
                    response = chat.send_message(user_question)
                    st.markdown("**Gemini says:**")
                    st.success(response.text)
                except Exception as e:
                    st.error("⚠️ Error fetching response. Please try again.")

