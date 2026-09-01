import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from torchvision import models, transforms


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
# Class names
# -----------------------------
class_names = [
    "Background_Clean",
    "Car_Body",
    "Metal_Surface",
    "PCB",
    "Road",
    "Synthetic_Concrete",
    "Wall_Concrete",
    "Welding",
    "Wood"
]


# -----------------------------
# Classifier preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Download and load models
# -----------------------------
@st.cache_resource
def load_models():

    # Download classifier
    classifier_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="crack_classifier.pth"
    )

    # Download YOLO detector
    yolo_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="crack_yolo.pt"
    )


    # -------------------------
    # Load YOLO model
    # -------------------------
    yolo_model = YOLO(yolo_path)


    # -------------------------
    # Recreate ResNet18
    # -------------------------
    classifier_model = models.resnet18(weights=None)

    classifier_model.fc = nn.Linear(
        classifier_model.fc.in_features,
        9
    )


    # -------------------------
    # Load trained classifier
    # -------------------------
    state_dict = torch.load(
        classifier_path,
        map_location=torch.device("cpu"),
        weights_only=True
    )

    classifier_model.load_state_dict(state_dict)

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


    # -------------------------
    # Display uploaded image
    # -------------------------
    st.subheader("Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )


    # -------------------------
    # Detect button
    # -------------------------
    if st.button("🔍 Detect Cracks"):

        with st.spinner("Analyzing image..."):


            # -------------------------
            # YOLO crack detection
            # -------------------------
            results = yolo_model(image)

            result_image = results[0].plot()


            # -------------------------
            # ResNet18 classification
            # -------------------------
            input_tensor = transform(image)

            input_tensor = input_tensor.unsqueeze(0)


            with torch.no_grad():

                output = classifier_model(
                    input_tensor
                )

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                predicted_class = torch.argmax(
                    probabilities,
                    dim=1
                ).item()

                confidence = probabilities[
                    0,
                    predicted_class
                ].item()


            predicted_label = class_names[
                predicted_class
            ]


        # -----------------------------
        # Detection result
        # -----------------------------
        st.subheader("Detection Result")

        st.image(
            result_image,
            use_container_width=True
        )


        # -----------------------------
        # Classification result
        # -----------------------------
        st.subheader("Classification Result")

        st.write(
            f"**Surface Type:** {predicted_label}"
        )

        st.write(
            f"**Classification Confidence:** "
            f"{confidence * 100:.2f}%"
        )


        # -----------------------------
        # Final status
        # -----------------------------
        st.success(
            "Detection and classification completed successfully!"
        )
