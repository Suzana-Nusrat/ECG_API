
import requests, json

URL = "http://localhost:8000/predict"

# Example JSON signal
signal = [0.01, 0.02, 0.03, 0.02, 0.01, -0.01, -0.02] * 1000  # dummy

r = requests.post(URL, json={"signal": signal})
print(r.status_code, r.json())
