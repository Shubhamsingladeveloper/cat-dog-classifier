import pathlib
import streamlit as st
from fastai.vision.all import *
from PIL import Image


pathlib.WindowsPath = pathlib.PosixPath


# Load trained model
learn = load_learner("cat_dog_classifier.pkl")


# Page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱"
)


# Title
st.title("🐱 Cat vs 🐶 Dog Classifier")

st.write("Upload an image and the model will predict whether it is a cat or dog.")


# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# Prediction
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )

    pred, pred_idx, probabilities = learn.predict(image)

    confidence = probabilities[pred_idx].item()

    st.subheader("Prediction")

    st.success(f"Prediction: {pred}")

    st.write(f"Confidence: {confidence:.2%}")

    st.subheader("Probabilities")

    for label, probability in zip(
        learn.dls.vocab,
        probabilities
    ):
        st.write(
            f"{label}: {probability:.2%}"
        )