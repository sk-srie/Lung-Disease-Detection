
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import streamlit as st


# Step 1: Recreate the same model architecture
model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 5)
)
# Step 2: Load the weights
state_dict = torch.load("../modals/best_model.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Step 3: Set model to eval
model.eval()

# Define class names
class_names = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # adapt to your model's input size
    transforms.ToTensor(),
])

# Streamlit UI
st.title("Chest X-Ray Classifier 🫁")

uploaded_file = st.file_uploader("Upload a Chest X-Ray image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)  # add batch dimension

    if st.button("Predict"):
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).numpy().flatten()
            pred_class = class_names[np.argmax(probs)]

            st.success(f"Predicted Class: **{pred_class}**")


            confidence = probs[np.argmax(probs)]
            st.subheader("Prediction Confidence")
            st.info(f"{pred_class}: **{confidence:.2%}**")


