from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn

# Class labels matching your dataset folders
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Build the model architecture
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model.eval()

# Image preprocessing — standard for MRI classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load your image
image_path = "/Users/klea/Desktop/thesis/data/archive /Training/pituitary/Tr-pi_766.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Run detection
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = probabilities.max(1)

print(f"\n--- RESULT ---")
print(f"Tumor type : {class_names[predicted.item()]}")
print(f"Confidence : {confidence.item():.1%}")