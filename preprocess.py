import numpy as np

def _read_csv_bytes(b: bytes):
    s = b.decode("utf-8", errors="ignore")
    data = []
    for line in s.strip().splitlines():
        line = line.strip().replace("\t", ",").replace(";", ",")
        if not line: continue
        try:
            val = float(line.split(",")[0])
            data.append(val)
        except Exception:
            continue
    return np.array(data, dtype=np.float32) if data else None

def load_signal_from_request(req):
    if req.is_json:
        j = req.get_json(silent=True) or {}
        if "signal" in j:
            try:
                arr = np.asarray(j["signal"], dtype=np.float32).ravel()
                if arr.size > 0: return arr
            except Exception:
                pass
    if "file" in req.files:
        b = req.files["file"].read()
        arr = _read_csv_bytes(b)
        if arr is not None and arr.size > 0: return arr
    return None

def resample_to_length(sig: np.ndarray, target_len: int) -> np.ndarray:
    n = sig.size
    if n == target_len: return sig.astype(np.float32)
    if n <= 1: return np.zeros(target_len, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, n, endpoint=True)
    x_new = np.linspace(0.0, 1.0, target_len, endpoint=True)
    return np.interp(x_new, x_old, sig).astype(np.float32)

def zscore(sig: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(np.mean(sig)); s = float(np.std(sig))
    return (sig - m) / (s + eps)

def preprocess_signal(sig: np.ndarray, target_len: int = 5000, zscore: bool = True) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float32).ravel()
    sig = resample_to_length(sig, target_len)
    if zscore: sig = zscore(sig)
    return sig