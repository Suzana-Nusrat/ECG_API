
# ECG Classification Flask API

This is a production-ready **Flask** API scaffold to serve a PyTorch ECG model.

## Quick Start

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Put your model checkpoint as `ecg_classification_model.pth` in this folder
python app.py
```

The server starts on `http://0.0.0.0:8000`.

## Endpoints

- `GET /health` → basic health info
- `GET /labels` → class labels (from `labels.json`)
- `POST /predict` → returns predicted label and probabilities

## Request Formats

1) **JSON** (preferred)
```json
{
  "signal": [0.012, -0.03, ...]   // 1D ECG samples
}
```

2) **File upload (CSV)** — send a single-column CSV file as `file` field.

## Environment Variables

- `ECG_MODEL_PATH` (default: `ecg_classification_model.pth`)
- `ECG_LABELS_PATH` (default: `labels.json`)
- `ECG_TARGET_LEN` (default: `5000` samples)
- `ECG_USE_ZSCORE` (default: `1` → enabled)

## Model Loading

- Tries **TorchScript** (`torch.jit.load`) first.
- If that fails, imports **`ECGModel`** from `model_def.py`, creates it with `num_classes = len(labels)`,
  then loads a **state_dict** from the checkpoint.

**Important:** Replace the placeholder `ECGModel` with your actual architecture if your checkpoint
is a state dict.

## Example Client

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"signal\":[0.01,0.02,0.03,0.02,0.01,-0.01,-0.02]}"
```

or

```bash
curl -X POST http://localhost:8000/predict \
  -F file=@sample.csv
```

## Notes

- The API resamples any input to `ECG_TARGET_LEN` with linear interpolation and applies z-score
  normalization (configurable).
- GPU will be used automatically if available.


### Image-based prediction (for ECG **images** datasets)
If your model is trained on ECG **images** (e.g., Kaggle *ECG Images Dataset of Cardiac Patients*), use:

```bash
curl -X POST http://localhost:8000/predict_image \
  -F file=@path/to/your/ecg_image.png
```

Environment knobs:
- `ECG_IMAGE_SIZE` (default `224`)
- `ECG_IMAGE_MEAN` (default `0.485,0.456,0.406`)
- `ECG_IMAGE_STD`  (default `0.229,0.224,0.225`)
