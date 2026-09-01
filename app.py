import streamlit as st
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Crack Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Crack Detection System")
st.write("Upload an image to detect and classify cracks.")


# -----------------------------
# Hugging Face model repository
# -----------------------------
HF_REPO = "PepsTechRnD/crack-detection-classification"


# -----------------------------
# Download models from Hugging Face
# -----------------------------
@st.cache_resource
def load_models():

    classifier_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="crack_classifier.pth"
    )

    yolo_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="crack_yolo.pt"
    )

    # Load YOLO model
    yolo_model = YOLO(yolo_path)

    # Load classifier
    classifier_model = torch.load(
        classifier_path,
        map_location=torch.device("cpu"),
        weights_only=False
    )

    if isinstance(classifier_model, torch.nn.Module):
        classifier_model.eval()

    return classifier_model, yolo_model


# -----------------------------
# Load models
# -----------------------------
try:
    classifier_model, yolo_model = load_models()
    st.success("✅ Models loaded successfully")

except Exception as e:
    st.error("❌ Could not load the models.")
    st.exception(e)
    st.stop()


# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a crack image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Run detection
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    if st.button("🔍 Detect Cracks"):

        with st.spinner("Analyzing image..."):

            # YOLO detection
            results = yolo_model(image)

            # Display YOLO result
            result_image = results[0].plot()

        st.subheader("Detection Result")
        st.image(result_image, use_container_width=True)

        st.success("Detection completed successfully!")
