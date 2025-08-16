# image_preprocess.py
import io, os
import numpy as np
from PIL import Image

def load_image_from_request(req):
    if "file" not in req.files:
        return None
    f = req.files["file"]
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
        return img
    except Exception:
        return None

def parse_mean_std(mean_str, std_str):
    def _p(s, default):
        try:
            vals = [float(x.strip()) for x in s.split(",")]
            if len(vals) == 1: return [vals[0]] * 3
            return vals
        except Exception:
            return default
    mean = _p(mean_str, [0.5, 0.5, 0.5])
    std  = _p(std_str,  [0.5, 0.5, 0.5])
    return mean, std

def _center_crop_resize(img, size):
    w, h = img.size
    scale = max(size / h, size / w)
    new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))

def _resize(img, size):
    return img.resize((size, size), Image.BILINEAR)

def _pad_letterbox(img, size, pad_val=0.0):
    # keep aspect via padding
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
    img_r = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (int(pad_val*255),)*3)
    left = (size - new_w) // 2
    top  = (size - new_h) // 2
    canvas.paste(img_r, (left, top))
    return canvas

def preprocess_image(img, size=224, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
    """
    ECG_IMAGE_MODE:
      - "resize": simple resize to (size,size)
      - "center": keep aspect, then center-crop to size
      - "pad"   : keep aspect, letterbox pad to (size,size)  [default]
    ECG_FORCE_GRAYSCALE: "1" to convert to L and replicate to 3 channels [default on]
    """
    mode = os.environ.get("ECG_IMAGE_MODE", "pad").lower()
    if mode == "resize":
        img = _resize(img, size)
    elif mode == "center":
        img = _center_crop_resize(img, size)
    else:
        img = _pad_letterbox(img, size, pad_val=0.0)

    if os.environ.get("ECG_FORCE_GRAYSCALE", "1") == "1":
        g = np.asarray(img.convert("L")).astype("float32")/255.0
        arr = np.stack([g,g,g], axis=2)
    else:
        arr = np.asarray(img).astype("float32")/255.0

    mean = np.array(mean, dtype="float32")[None,None,:]
    std  = np.array(std,  dtype="float32")[None,None,:]
    arr = (arr - mean) / std
    return np.transpose(arr, (2,0,1))  # (C,H,W)
