import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Class labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load your trained model
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    model.load_state_dict(torch.load("brain_tumor_model.pth",
                          map_location=torch.device("cpu")))
    model.eval()
    return model

# Run detection on image
def detect_tumor(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    tumor_type = class_names[predicted.item()]
    confidence_score = confidence.item()
    return tumor_type, confidence_score

# Generate radiology report using GPT
def generate_report(tumor_type, confidence):
    prompt = f"""You are an experienced radiologist.
A brain MRI scan has been analyzed by an AI system which detected: {tumor_type} with {confidence:.1%} confidence.

Generate a structured radiology report with the following sections:
1. FINDINGS
2. IMPRESSION
3. RECOMMENDATION

Use professional medical language. Be specific about the tumor type.
End with a disclaimer that this is AI-generated and requires radiologist review."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an experienced radiologist writing clinical reports."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

# Main pipeline
if __name__ == "__main__":
    image_path = "/Users/klea/Desktop/thesis/data/archive /Testing/glioma/Te-gl_1.jpg"

    print("Loading trained model...")
    model = load_model()

    print("Analyzing MRI image...")
    tumor_type, confidence = detect_tumor(image_path, model)
    print(f"\nDetection result: {tumor_type} ({confidence:.1%} confidence)")

    print("\nGenerating radiology report...")
    report = generate_report(tumor_type, confidence)

    print("\n" + "="*50)
    print("RADIOLOGY REPORT")
    print("="*50)
    print(report)
    print("="*50)