from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)

# =========================
# 1. MODEL SETUP
# =========================

model = models.shufflenet_v2_x1_5(weights=None)
model.fc = nn.Linear(model.fc.in_features, 9)

state_dict = torch.load("best_shufflenetv2_fabric_defect.pth", map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

# =========================
# 2. EXACT CLASS ORDER (FROM NOTEBOOK)
# =========================

classes = [
    "Broken stitch",
    "Needle mark",
    "Pinched fabric",
    "Vertical",
    "defect free",
    "hole",
    "horizontal",
    "lines",
    "stain"
]

# =========================
# 3. TRANSFORM (MATCH TRAINING EXACTLY)
# =========================

# FIX: removed normalization to match training pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# 4. MAP 9 → 3 CLASSES
# =========================

def map_to_main_defect(label):
    if label.lower() == "hole":
        return "hole"
    elif label.lower() == "stain":
        return "stain"
    elif label.lower() in [
        "vertical", "horizontal", "lines",
        "broken stitch", "needle mark", "pinched fabric"
    ]:
        return "misweave"
    else:
        return "normal"

# =========================
# 5. RECOMMENDATION ENGINE
# =========================

def get_recommendation(label):
    if label == "hole":
        return "Discard or recut fabric"
    elif label == "stain":
        return "Clean or repurpose"
    elif label == "misweave":
        return "Repair if possible"
    else:
        return "Accept fabric"

# =========================
# 6. ROUTES
# =========================

@app.route('/')
def home():
    return "Fabric Defect API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        image = Image.open(file).convert("RGB")

        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        predicted_idx = predicted.item()
        raw_label = classes[predicted_idx]

        label = map_to_main_defect(raw_label)
        recommendation = get_recommendation(label)

        return jsonify({
            "raw_defect": raw_label,
            "defect": label,
            "confidence": round(confidence.item() * 100, 2),
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# 7. RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(debug=True)