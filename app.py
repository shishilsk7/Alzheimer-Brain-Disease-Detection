import gradio as gr
from PIL import Image
import torch
from torchvision import transforms

from model import AlzheimerCNN  # Assuming model is in model.py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your model (assuming it has been saved as model.pth)
model = AlzheimerCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Image transformation (resize and tensor conversion)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Define classes and their descriptions
classes = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

descriptions = {
    "Mild Demented": "Mild cognitive impairment (MCI) is characterized by noticeable memory problems that are not severe enough to interfere with daily life.",
    "Moderate Demented": "Moderate dementia leads to significant memory loss, confusion, and difficulty with daily activities. It can be challenging for the person to remember basic information.",
    "Non Demented": "The person shows no signs of cognitive impairment. They have normal cognitive function and memory retention.",
    "Very Mild Demented": "Very mild dementia is the early stage of cognitive decline, where subtle memory lapses begin but don’t significantly impact daily activities."
}

# Add hypothetical survival rates or critical information for each class
survival_rates = {
    "Mild Demented": "Survival rate: 85% over 10 years",
    "Moderate Demented": "Survival rate: 65% over 5 years",
    "Non Demented": "No risk, normal survival rate",
    "Very Mild Demented": "Survival rate: 90% over 10 years"
}

# Define function for prediction
def classify_image_gradio(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    
    predicted_class = classes[predicted.item()]
    description = descriptions[predicted_class]
    survival_rate = survival_rates[predicted_class]
    
    # Return class, description, and survival rate
    return f"Prediction: {predicted_class}\n\nDescription: {description}\n\nSurvival Rate: {survival_rate}"

# Automatically load sample images
sample_images = [
    'sample_image_1.jpg',
    'sample_image_2.jpg',
    'sample_image_3.jpg',
    'sample_image_4.jpg',
    'sample_image_5.jpg'
]

def load_sample_images():
    images = []
    for img_path in sample_images:
        img = Image.open(img_path)
        images.append(img)
    return images

# Gradio interface
iface = gr.Interface(
    fn=classify_image_gradio,
    inputs=gr.Image(type="pil", label="Upload an image (or try a sample image)"),
    outputs="text",
    live=True,
    examples=load_sample_images()  # Show the sample images as examples
)

# Launch the Gradio interface
iface.launch(share=True)
