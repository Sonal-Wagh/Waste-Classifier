import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
import gdown
import time



model = Sequential([
    Input(shape=(128, 128, 3), name="input_layer"),
    Conv2D(32, (3, 3), activation='relu', name="conv2d"),
    MaxPooling2D(2, 2, name="max_pooling2d"),
    Conv2D(64, (3, 3), activation='relu', name="conv2d_1"),
    MaxPooling2D(2, 2, name="max_pooling2d_1"),
    Conv2D(128, (3,3), activation='relu', name="conv2d_2"),  
    MaxPooling2D(2,2, name="max_pooling2d_2"),
    Flatten(name="flatten"),
    Dense(256, activation='relu',name="dense"),
    Dense(128, activation='relu', name="dense_1"),
    Dense(3, activation='softmax', name="dense_2")
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


file_id = "1df8WQrhTDaZqEZ_4HczdF_BvtpP0JFc4"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "waste_classifier5.weights.h5"

if not os.path.exists(model_path):
    with st.spinner("🔄 Downloading model weights from Google Drive..."):
        gdown.download(url, model_path, quiet=False)

model.load_weights(model_path)


model.load_weights("waste_classifier5.weights.h5")


class_names = ['Biodegradable', 'Hazardous', 'Recyclable']


st.set_page_config(page_title="♻️ Waste Classifier", page_icon="🧪", layout="centered")


st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #98eaf5,#ed6fe7 );
        font-family: 'Segoe UI', sans-serif;
    }
    .reportview-container {
        background: linear-gradient(to right, #7ced6f, #f3e5f5);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #81d4fa, #ce93d8);
    }
    .title {
        font-size: 40px;
        color: #4a148c;
        text-align: center;
        margin-bottom: 20px;
    }
    .confidence {
        font-size: 20px;
        color: #00695c;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 12px;
        padding-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="title">♻️ Smart Waste Classifier</div>', unsafe_allow_html=True)


uploaded_file = st.file_uploader("📤 Upload an image of waste", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    
    if st.button("🔍 Analyze Waste Type"):
        

         img_resized = img.resize((128, 128))
         img_array = image.img_to_array(img_resized) / 255.0
         img_array = np.expand_dims(img_array, axis=0)

            
         prediction = model.predict(img_array)
         predicted_index = np.argmax(prediction[0])
         predicted_class = class_names[predicted_index]
         confidence = prediction[0][predicted_index] * 100
 
           
         st.success(f"🧠 Prediction: **{predicted_class}**")
         st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
         st.progress(confidence / 100)

            

st.markdown('<div class="footer">🚀 Built with TensorFlow, Keras & Streamlit | © 2025</div>', unsafe_allow_html=True)
