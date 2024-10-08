import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the pre-trained model (replace 'your_cifar10_model.h5' with your model file)
model = load_model('CIFAR10_V4.h5')

# Preprocessing function for CIFAR-10 images (32x32 images)
def preprocess_image(image):
    # Resize image to 32x32 for CIFAR-10
    img = image.resize((32, 32))  
    # Convert to numpy array and scale the pixel values to [0, 1]
    img = np.array(img) / 255.0
    # Expand dimensions to make it batch-ready (1, 32, 32, 3)
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit app
st.title('CIFAR-10 Image Classification')

# File uploader for image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img = preprocess_image(image)

    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])  # Get the predicted class index
    confidence = np.max(predictions[0])  # Get the confidence level

    # Display the prediction and confidence
    st.write(f'Predicted Class: {class_names[predicted_class]}')
    st.write(f'Confidence: {confidence * 100:.2f}%')
