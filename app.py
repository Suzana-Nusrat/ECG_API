import os
import json
import traceback
import numpy as np
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F

from image_preprocess import (
    load_image_from_request,
    preprocess_image,
    parse_mean_std,
)

# ----------------- Config -----------------
MODEL_PATH   = os.environ.get("ECG_MODEL_PATH", "ecg_classification_model.pth")  # TorchScript OR state_dict
LABELS_PATH  = os.environ.get("ECG_LABELS_PATH", "labels.json")
IMAGE_SIZE   = int(os.environ.get("ECG_IMAGE_SIZE", "224"))
IMG_MEAN_STR = os.environ.get("ECG_IMAGE_MEAN", "0.5,0.5,0.5")     # ECG-friendly default
IMG_STD_STR  = os.environ.get("ECG_IMAGE_STD",  "0.5,0.5,0.5")     # ECG-friendly default

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = None
labels = None

# ----------------- Helpers -----------------
def load_labels():
    """Load class names. Default order matches the common ECG image datasets: HB, MI, PMI, Normal."""
    global labels
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
    else:
        labels = ["HB", "MI", "PMI", "Normal"]

def _try_load_torchscript(path: str):
    m = torch.jit.load(path, map_location=device)
    m.eval().to(device)
    return m

def _strip_module_prefix(sd: dict) -> dict:
    # Handle DataParallel "module." prefix
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def _looks_like_conv_fc_style(sd: dict) -> bool:
    # Your checkpoint style: conv_layers.* and fc_layers.* present
    has_conv = any(k.startswith("conv_layers.") for k in sd.keys())
    has_fc   = any(k.startswith("fc_layers.") for k in sd.keys())
    return has_conv and has_fc

def _build_model_from_conv_fc_sd(sd: dict, num_classes_file: int):
    from model_def import build_from_state_dict
    net = build_from_state_dict(sd, num_classes_file)  # builds a model that matches your keys exactly
    net.eval().to(device)
    return net

def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "net", "model"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj  # maybe already a pure state_dict
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise RuntimeError("Unsupported checkpoint: expected TorchScript or a state_dict-like object.")

def _try_load_state_dict(path: str, num_classes_file: int):
    obj = torch.load(path, map_location=device)
    sd = _strip_module_prefix(_extract_state_dict(obj))
    if not _looks_like_conv_fc_style(sd):
        sample = list(sd.keys())[:8]
        raise RuntimeError(
            "Expected checkpoint with keys like 'conv_layers.*' and 'fc_layers.*'. "
            f"Found sample keys: {sample}"
        )
    # Build a network that EXACTLY fits those keys and load STRICTLY.
    return _build_model_from_conv_fc_sd(sd, num_classes_file)

def load_model():
    global model
    # 1) TorchScript first
    try:
        model = _try_load_torchscript(MODEL_PATH)
        print("[INFO] Loaded TorchScript:", MODEL_PATH)
        return
    except Exception as e_ts:
        print("[INFO] Not TorchScript or failed jit.load -> falling back to state_dict. Reason:", e_ts)

    # 2) State dict (conv_layers.* / fc_layers.*), strict
    num_classes_file = len(labels) if labels else 4
    model = _try_load_state_dict(MODEL_PATH, num_classes_file)
    print("[INFO] Loaded state_dict with conv_layers/fc_layers:", MODEL_PATH)

def warmup():
    try:
        x = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32, device=device)
        with torch.no_grad():
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception as e:
        print("[WARN] Warmup failed (not fatal):", e)

def startup():
    load_labels()
    load_model()
    warmup()
    print(f"[READY] Device={device}  Model={type(model).__name__}  Classes={len(labels)}  ImageSize={IMAGE_SIZE}")

def ensure_loaded():
    if globals().get("model") is None or globals().get("labels") is None:
        startup()

def _last_linear(module: nn.Module):
    last = None
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            last = (name, m)
    return last

# ----------------- Endpoints -----------------
@app.get("/health")
def health():
    ensure_loaded()
    return jsonify(status="ok", device=str(device), classes=len(labels), image_size=IMAGE_SIZE)

@app.get("/labels")
def get_labels():
    ensure_loaded()
    return jsonify(labels=labels)

@app.get("/config")
def config_view():
    ensure_loaded()
    return jsonify(
        image_mode=os.environ.get("ECG_IMAGE_MODE", "pad"),
        force_grayscale=os.environ.get("ECG_FORCE_GRAYSCALE", "1"),
        mean=IMG_MEAN_STR,
        std=IMG_STD_STR,
        image_size=IMAGE_SIZE,
    )

@app.get("/final_bias")
def final_bias():
    ensure_loaded()
    last = _last_linear(model)
    if last is None or last[1].bias is None:
        return jsonify(info="no final linear bias")
    b = last[1].bias.detach().cpu().numpy().tolist()
    return jsonify(bias=b, labels=labels)

@app.post("/predict_image")
def predict_image():
    ensure_loaded()
    try:
        img = load_image_from_request(request)
        if img is None:
            return jsonify(error="No image provided. Upload an image file in field 'file'."), 400

        mean, std = parse_mean_std(IMG_MEAN_STR, IMG_STD_STR)
        arr = preprocess_image(img, size=IMAGE_SIZE, mean=mean, std=std)  # (C,H,W)
        xt  = torch.from_numpy(arr).float().unsqueeze(0).to(device)       # (1,3,H,W)

        with torch.no_grad():
            out = model(xt)
            if isinstance(out, (list, tuple)):
                out = out[0]

        logits = out
        if isinstance(logits, torch.Tensor):
            if logits.ndim == 1:   # (C,)
                logits = logits.unsqueeze(0)  # (1,C)
        else:
            raise RuntimeError("Model output is not a torch.Tensor")

        probs = F.softmax(logits, dim=1).cpu().numpy()[0].tolist()
        top_idx = int(np.argmax(probs))
        top_label = labels[top_idx] if labels and top_idx < len(labels) else str(top_idx)

        return jsonify(
            predicted_index=top_idx,
            predicted_label=top_label,
            probabilities=[{"label": (labels[i] if labels and i < len(labels) else str(i)), "p": float(p)} for i, p in enumerate(probs)]
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

# ----------------- Main -----------------
if __name__ == "__main__":
    with app.app_context():
        startup()
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
