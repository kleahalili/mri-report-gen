from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import os
import io

# Load env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Brain MRI Radiology Report Generator")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load model once at startup
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    model.load_state_dict(torch.load("brain_tumor_model.pth",
                          map_location=torch.device("cpu")))
    model.eval()
    return model

print("Loading model...")
model = load_model()
print("Model ready!")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def detect_tumor(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    return class_names[predicted.item()], confidence.item()

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

@app.get("/")
def root():
    return {"message": "Brain MRI Report Generator API is running"}

@app.post("/analyze")
async def analyze_mri(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        return JSONResponse(
            status_code=400,
            content={"error": "Only JPG and PNG images are supported"}
        )
    
    # Read image
    image_bytes = await file.read()
    
    # Detect tumor
    tumor_type, confidence = detect_tumor(image_bytes)
    
    # Generate report
    report = generate_report(tumor_type, confidence)
    
    return {
        "tumor_type": tumor_type,
        "confidence": f"{confidence:.1%}",
        "report": report
    }