import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from train import train_model  # Import the train_model function from train.py

# Streamlit UI
st.set_page_config(page_title='Handwritten Digit Classifier', page_icon="🚀")

st.title('Handwritten Digit Recognizer')
st.subheader('This webapp predicts the number that you have drawn on the canvas from 0-9')

# Define the model directory and model file path
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model.keras')
MODEL_FILE = 'model.keras'

# Check if the model file exists
if not os.path.isfile(MODEL_DIR):
    st.warning(f"Model file not found at {MODEL_FILE}. Please train the data first.")
    if st.button("Train Data", type="primary"):
        with st.spinner("Training the model..."):
            train_model(MODEL_FILE)
else:
    try:
        # Load the model
        model = load_model(MODEL_FILE)
        st.success(f"Model loaded successfully from {MODEL_FILE}.")
        
        # Divide the layout into two columns
        col1, col2 = st.columns(2)
        
        with col1:
            SIZE = 192
            mode = st.checkbox("Draw (or Delete)?", True)
            canvas_result = st_canvas(
                fill_color='#000000',
                stroke_width=20,
                stroke_color='#FFFFFF',
                background_color='#000000',
                width=SIZE,
                height=SIZE,
                drawing_mode="freedraw" if mode else "transform",
                key='canvas'
            )
        
        with col2:
            if canvas_result.image_data is not None:
                # Resize the image to 28x28
                img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))

                # Convert the image to grayscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.write("Your drawn image:")
                rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
                st.image(rescaled)
            else:
                st.warning("Please draw a digit on the canvas.")
        
        if canvas_result.image_data is not None:
            if np.any(gray_img):  # Check if there is at least one non-zero pixel
                gray_img = gray_img / 255.0  # Normalize the image
                
                # Add a batch dimension
                img_array = gray_img.reshape((1, 28, 28, 1))

                # Make a prediction
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction)

                # Display the prediction and chart below columns
                st.write(f'Predicted Class: {predicted_class}')
                st.bar_chart(prediction[0])
            else:
                st.warning("Please draw a digit on the canvas.")
        
        with st.expander("Data Loaded Information"):
            st.write('''
                The MNIST dataset consists of handwritten digits with 60,000 training images and 10,000 test images.
                Each image is grayscale and has a size of 28x28 pixels.
            ''')
            st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png")
        
    except Exception as e:
        st.error(f"Error loading the model: {e}")
