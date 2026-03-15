import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- CONFIG ---
MODEL_PATH = "models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BREED_NAMES = ["Alambadi","Amritmahal","Ayrshire","Banni","Bargur","Bhadawari","Brown_Swiss","Dangi","Deoni","Gir","Guernsey","Hallikar","Hariana","Holstein_Friesian","Jaffrabadi","Jersey","Kangayam","Kankrej","Kasargod","Kenkatha","Kherigarh","Khillari","Krishna_Valley","Malnad_gidda","Mehsana","Murrah","Nagori","Nagpuri","Nili_Ravi","Nimari","Ongole","Pulikulam","Rathi","Red_Dane","Red_Sindhi","Sahiwal","Surti","Tharparkar","Toda","Umblachery","Vechur"]  # replace with your actual class names
CONF_THRESHOLD = 0.5  # confidence threshold for unknown images

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- MODEL ---
num_classes = len(BREED_NAMES)
model = models.resnet50(weights=None)  # ResNet50 instead of 18
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# --- PREDICTION FUNCTION ---
def predict(image_path, threshold=CONF_THRESHOLD):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    confidence = conf.item()
    if confidence < threshold:
        return "Unknown / Not a cow", confidence * 100

    breed = BREED_NAMES[pred_idx.item()]
    return breed, confidence * 100

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    test_images = ["test_images/m.jpg"     
    ]

    for img_path in test_images:
        breed, conf = predict(img_path)
        print(f" Predicted: {breed}, Confidence: {conf:.2f}%")
