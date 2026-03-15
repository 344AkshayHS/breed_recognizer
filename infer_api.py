# infer_api.py
import os
import io
import time
from PIL import Image
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from gradcam import GradCAM
import numpy as np
import cv2

# --- CONFIG ---
MODEL_PATH = "models/best_model.pth"
IMG_SIZE = 224
BREED_NAMES = ["Alambadi","Amritmahal","Ayrshire","Banni","Bargur","Bhadawari","Brown_Swiss","Dangi","Deoni","Gir","Guernsey","Hallikar","Hariana","Holstein_Friesian","Jaffrabadi","Jersey","Kangayam","Kankrej","Kasargod","Kenkatha","Kherigarh","Khillari","Krishna_Valley","Malnad_gidda","Mehsana","Murrah","Nagori","Nagpuri","Nili_Ravi","Nimari","Ongole","Pulikulam","Rathi","Red_Dane","Red_Sindhi","Sahiwal","Surti","Tharparkar","Toda","Umblachery","Vechur"]  # replace with your actual class names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- UTILS ---
def load_model():
    num_classes = len(BREED_NAMES)
    model = models.resnet50(weights=None)  # ResNet50
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()

    # last conv layer for Grad-CAM
    target_layer = model.layer4[-1]
    return model, BREED_NAMES, target_layer

model, classes, target_layer = load_model()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --- FLASK APP ---
app = Flask(__name__)
os.makedirs("outputs/heatmaps", exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file sent (form field 'image')"}), 400

    file = request.files["image"]
    pil = Image.open(file.stream).convert("RGB")
    input_tensor = preprocess(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    topk = 3
    topk_idx = probs.argsort()[::-1][:topk]
    predictions = [{"class": classes[i], "confidence": float(probs[i])} for i in topk_idx]

    # Grad-CAM heatmap for top1 prediction
    try:
        gcam = GradCAM(model, target_layer)
        heat = gcam(input_tensor, target_class=int(topk_idx[0]))

        img_np = np.array(pil.resize((IMG_SIZE, IMG_SIZE)))[:, :, ::-1]  # RGB->BGR
        heatmap = cv2.applyColorMap(np.uint8(255*heat), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        heat_filename = f"outputs/heatmaps/heat_{int(time.time())}.png"
        cv2.imwrite(heat_filename, overlay)
    except Exception as e:
        heat_filename = None
        print("GradCAM failed:", e)

    return jsonify({
        "predictions": predictions,
        "model_version": MODEL_PATH,
        "heatmap_path": heat_filename
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
