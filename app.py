import streamlit as st
st.set_page_config(page_title="FaceTune", layout="centered", page_icon="🎵")

import numpy as np
import cv2
from PIL import Image
import time
from tensorflow.keras.models import load_model
from recommend_song import recommend_song
from generate_lyrics import generate_lyrics_with_langchain

# Load model once
@st.cache_resource
def load_emotion_model():
    return load_model("emotion_detection_cnn.h5")

emotion_model = load_emotion_model()

# RAF-DB emotion labels
emotion_labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion detection from uploaded image
def detect_emotion(image):
    # Convert PIL image to RGB (if not already)
    image = image.convert("RGB")
    img_np = np.array(image)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return "neutral", {"neutral": 1.0}

    x, y, w, h = faces[0]  # Take the first detected face
    face_rgb = img_np[y:y+h, x:x+w]
    
    # Preprocess for model
    face_resized = cv2.resize(face_rgb, (100, 100))
    face_normalized = face_resized / 255.0
    input_data = np.expand_dims(face_normalized, axis=0)
    
    # Predict
    prediction = emotion_model.predict(input_data, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # Build probability dictionary
    emotion_probs = {emotion_labels[i]: float(prediction[0][i]) for i in range(len(emotion_labels))}

    return emotion_labels[predicted_class], emotion_probs

# Simulated lyrics generator
# def generate_lyrics(emotion):
#     return f"Here are some {emotion} lyrics: 🎶\n\n[Generated Lyrics Here]"

# Simulated music recommender
def recommend_music(emotion):
    songs = recommend_song(emotion)
    return "**🎵 Recommended Songs:**\n" + "\n".join(songs)

# --- Main App ---
st.title(":musical_note: FaceTune: Emotion-Based Music & Lyrics")

# Initialize session state
if 'emotion' not in st.session_state:
    st.session_state.emotion = None

mode = st.sidebar.selectbox("Select Mode", ["Image Upload", "Real-Time Webcam"])

# ------------------- Image Upload Mode -------------------
if mode == "Image Upload":
    st.subheader(":camera: Upload a Photo")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting emotion..."):
            emotion, emotion_probs = detect_emotion(image)
            st.session_state.emotion = emotion

        st.success(f"Detected Emotion: **{emotion.capitalize()}**")
        st.bar_chart(emotion_probs)

        col1, col2 = st.columns(2)

        with col1:
            if st.button(":microphone: Generate Lyrics"):
                with st.spinner("Generating lyrics..."):
                    output = generate_lyrics_with_langchain(emotion)
                    st.markdown("### ✨ Generated Text:")
                    st.write(output)

        with col2:
            if st.button(":headphones: Recommend Music"):
                with st.spinner("Finding recommendations..."):
                    music = recommend_music(emotion)
                    st.markdown(music, unsafe_allow_html=True)

# ------------------- Webcam Mode -------------------
elif mode == "Real-Time Webcam":
    st.subheader(":movie_camera: Real-Time Emotion Detection")
    st.write("Live prediction using your trained CNN model from RAF-DB.")

    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        st.warning("Press STOP to close webcam properly.")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_rgb = rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face_rgb, (100, 100))
                face_normalized = face_resized / 255.0
                input_data = np.expand_dims(face_normalized, axis=0)

                prediction = emotion_model.predict(input_data, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                label = f"{emotion_labels[predicted_class]} ({confidence:.1f}%)"

                cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(rgb, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                st.session_state.emotion = emotion_labels[predicted_class]

            FRAME_WINDOW.image(rgb)

        cap.release()
