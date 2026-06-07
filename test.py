import argparse
import os
from PIL import Image
import torch
import torch.nn as nn
import timm
from torchvision import transforms

# -----------------------------
# SETTINGS — EDIT THIS IF NEEDED
# -----------------------------
MODEL_PATH = "best_vit_lsd_classifier.pth"   # saved model
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ViT normalization
vit_mean = [0.485, 0.456, 0.406]
vit_std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(vit_mean, vit_std),
])

# -----------------------------
# Severity Mapping
# -----------------------------
def severity_from_prob(prob):
    return float(prob) * 10.0

def categorize_severity(severity):
    if severity < 4.0:
        return "Mild"
    elif severity <= 7.0:
        return "Moderate"
    else:
        return "Severe"

# -----------------------------
# Load Model
# -----------------------------
def load_model():
    model = timm.create_model("vit_base_patch16_224.augreg_in21k", pretrained=False)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------------
# Predict Function
# -----------------------------
def predict(model, image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    severity = severity_from_prob(prob)
    category = categorize_severity(severity)

    return prob, severity, category

# -----------------------------
# Main CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("Error: image not found:", args.image)
        return

    print("Loading model...")
    model = load_model()

    print("🔍 Running inference...")
    prob, severity, category = predict(model, args.image)

    print("\n========== RESULT ==========")
    print(f"Image: {args.image}")
    print(f"Probability (Lumpy): {prob:.4f}")
    print(f"Severity (0–10):     {severity:.2f}")
    print(f"Category:            {category}")
    print("============================\n")

if __name__ == "__main__":
    main()
