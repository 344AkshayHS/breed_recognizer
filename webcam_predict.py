import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# ---------------- CONFIG ----------------
MODEL_PATH = "models/best_model.pth"
BREEDS = ["Alambadi","Amritmahal","Ayrshire","Banni","Bargur","Bhadawari","Brown_Swiss","Dangi","Deoni","Gir","Guernsey","Hallikar","Hariana","Holstein_Friesian","Jaffrabadi","Jersey","Kangayam","Kankrej","Kasargod","Kenkatha","Kherigarh","Khillari","Krishna_Valley","Malnad_gidda","Mehsana","Murrah","Nagori","Nagpuri","Nili_Ravi","Nimari","Ongole","Pulikulam","Rathi","Red_Dane","Red_Sindhi","Sahiwal","Surti","Tharparkar","Toda","Umblachery","Vechur"]  # Your dataset classes
IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------

# --------- Grad-CAM Class ----------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.gradients = None
        self.activations = None

        # hooks
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class=None):
        self.model.zero_grad()
        out = self.model(input_tensor)
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        score = out[0, target_class]
        score.backward(retain_graph=True)

        grads = self.gradients[0].cpu().numpy()
        acts = self.activations[0].cpu().numpy()
        weights = np.mean(grads, axis=(1, 2))
        gcam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            gcam += w * acts[i]
        gcam = np.maximum(gcam, 0)
        gcam = cv2.resize(gcam, (input_tensor.size(3), input_tensor.size(2)))
        gcam -= gcam.min()
        if gcam.max() != 0:
            gcam /= gcam.max()
        return gcam

# -------- Load Model ----------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(BREEDS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# For Grad-CAM, use last conv layer
target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)

# -------- Transforms ----------
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------- Webcam ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("📷 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't receive frame. Exiting...")
        break

    # Preprocess frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
        breed = BREEDS[preds.item()]
        confidence = conf.item() * 100

    # Generate Grad-CAM heatmap
    try:
        heat = gradcam(img_tensor, target_class=preds.item())
        heatmap = cv2.applyColorMap(np.uint8(255*heat), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    except Exception as e:
        overlay = frame

    # Display
    cv2.putText(overlay, f"{breed} ({confidence:.1f}%)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Breed Recognition + Grad-CAM", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
