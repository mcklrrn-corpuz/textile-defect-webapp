from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)

# 1. MODEL SETUP
model = models.shufflenet_v2_x1_5(weights=None)
model.fc = nn.Linear(model.fc.in_features, 9)

state_dict = torch.load("best_shufflenetv2_fabric_defect.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 2. CLASS ORDER
classes = [
    "broken stitch",
    "needle mark",
    "pinched fabric",
    "vertical",
    "defect free",
    "hole",
    "horizontal",
    "lines",
    "stain"
]

# 3. TRANSFORM
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. MAP TO MAIN DEFECTS
def map_to_main_defect(label):
    if label == "hole":
        return "hole"
    elif label == "stain":
        return "stain"
    elif label in [
        "vertical", "horizontal", "lines",
        "broken stitch", "needle mark", "pinched fabric"
    ]:
        return "misweave"
    else:
        return "normal"

# 5. RECOMMENDATION LOGIC
def get_recommendation(label, confidence):

    # 🔴 HOLE PRIORITY
    if label == "hole":
        if confidence >= 0.50:
            return {
                "level": "Critical",
                "action": "Discard",
                "message": "Hole detected — high-risk defect even at moderate confidence"
            }
        else:
            return {
                "level": "Critical",
                "action": "Inspect",
                "message": "Possible hole detected — verify before discarding"
            }

    # NORMAL FLOW
    if confidence >= 0.85:
        if label == "stain":
            return {
                "level": "Moderate",
                "action": "Repurpose",
                "message": "Stain detected; cleaning may be possible"
            }

        elif label == "misweave":
            return {
                "level": "Moderate",
                "action": "Repair",
                "message": "Weaving defect detected"
            }

        else:
            return {
                "level": "Acceptable",
                "action": "Accept",
                "message": "No significant defects detected"
            }

    elif confidence >= 0.60:
        return {
            "level": "Uncertain",
            "action": "Inspect",
            "message": "Moderate confidence — manual inspection recommended"
        }

    else:
        return {
            "level": "Low Confidence",
            "action": "Re-capture",
            "message": "Prediction unreliable — capture better image"
        }

# 6. ROUTES
@app.route('/')
def home():
    return "Fabric Defect API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        image = Image.open(file).convert("RGB")

        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        raw_label = classes[predicted.item()]
        confidence_value = confidence.item()
        confidence_percent = round(confidence_value * 100, 2)

        if confidence_value < 0.60:
            return jsonify({
                "raw_defect": raw_label,
                "defect": "unknown", 
                "confidence": confidence_percent,
                "recommendation": {
                    "level": "Low Confidence",
                    "action": "Re-capture",
                    "message": "Prediction unreliable — capture better image"
                }
            })

        label = map_to_main_defect(raw_label)

        recommendation = get_recommendation(label, confidence_value)

        return jsonify({
            "raw_defect": raw_label,
            "defect": label,
            "confidence": confidence_percent,
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "recommendation": {
                "level": "System Error",
                "action": "Retry",
                "message": "An error occurred during processing"
            }
        })

# =========================
# 7. RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True)