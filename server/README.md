# Malware Detection API Server

This API server provides malware detection capabilities using a Random Forest machine learning model trained on PE file metadata features.

## 🚀 Quick Start

### 1. Start the Server

```bash
cd server
python api_server.py
```

The server will start on `http://localhost:8000`

### 2. Test the Server

Open your browser and go to:
- **Health Check**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs (Interactive Swagger UI)

## 📡 API Endpoints

### 1. Health Check
**GET** `/` or `/health`

Check if the server is running and the model is loaded.

**Response:**
```json
{
  "status": "online",
  "model_loaded": true,
  "feature_extractor_loaded": true,
  "num_features": 29,
  "model_info": {
    "training_date": "2024-01-01T12:00:00",
    "accuracy": 0.95,
    "roc_auc": 0.98
  }
}
```

### 2. Scan File (Basic)
**POST** `/scan`

Upload a PE file (.exe, .dll, .sys) to check if it's malware.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (binary)

**Response:**
```json
{
  "filename": "sample.exe",
  "prediction": "malware",
  "confidence": 0.95,
  "malware_probability": 0.95,
  "benign_probability": 0.05,
  "timestamp": "2024-01-01T12:00:00",
  "features_extracted": 29,
  "file_size": 102400
}
```

### 3. Scan File (Detailed)
**POST** `/scan/detailed`

Upload a PE file and get detailed analysis including feature values.

**Response:**
```json
{
  "filename": "sample.exe",
  "prediction": "malware",
  "confidence": 0.95,
  "malware_probability": 0.95,
  "benign_probability": 0.05,
  "timestamp": "2024-01-01T12:00:00",
  "features_extracted": 29,
  "file_size": 102400,
  "top_features": [
    {
      "feature": "general.size",
      "value": 102400,
      "importance": 0.15
    }
  ],
  "all_features": {
    "general.size": 102400,
    "general.vsize": 98304,
    ...
  }
}
```

### 4. Model Information
**GET** `/model/info`

Get information about the loaded model.

**Response:**
```json
{
  "num_features": 29,
  "training_date": "2024-01-01T12:00:00",
  "model_params": {
    "n_estimators": 200,
    "max_depth": 30
  },
  "metrics": {
    "accuracy": 0.95,
    "roc_auc": 0.98
  },
  "top_features": [...]
}
```

## 🧪 Testing with cURL

### Health Check
```bash
curl http://localhost:8000/health
```

### Scan a File
```bash
curl -X POST "http://localhost:8000/scan" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/file.exe"
```

### Detailed Scan
```bash
curl -X POST "http://localhost:8000/scan/detailed" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/file.exe"
```

## 🐍 Testing with Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Scan a file
with open("sample.exe", "rb") as f:
    files = {"file": ("sample.exe", f, "application/octet-stream")}
    response = requests.post("http://localhost:8000/scan", files=files)
    result = response.json()
    
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Malware probability: {result['malware_probability']:.2%}")
```

## 📋 Requirements

The server requires:
- **Model file**: `Model/randomforest_malware_detector.pkl`
- **Feature list**: `Model/feature_list.json`
- **Feature extractor**: `server/feature_extractor.py` (included)

### Python Dependencies
```
fastapi
uvicorn
pydantic
joblib
pefile
numpy
```

Install with:
```bash
pip install -r ../requirements.txt
```

## ⚙️ Configuration

Edit the configuration in `api_server.py`:

```python
# Model paths
MODEL_DIR = r"C:\Users\Admin\github-classroom\Caty175\poly_trial\Model"
MODEL_PATH = os.path.join(MODEL_DIR, "randomforest_malware_detector.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.json")

# File size limit
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# Server settings (in __main__)
uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
```

## 🔒 Security Considerations

⚠️ **Important**: This is a development server. For production use:

1. **Add Authentication**: Implement API key or OAuth authentication
2. **Rate Limiting**: Prevent abuse with rate limiting
3. **HTTPS**: Use SSL/TLS encryption
4. **Sandboxing**: Run file analysis in isolated environment
5. **Input Validation**: Additional file type and size validation
6. **Logging**: Implement comprehensive logging and monitoring
7. **Firewall**: Restrict access to trusted networks

## 🎯 Features Extracted

The model analyzes **29 metadata features** from PE files:

### General Features (10)
- File size, virtual size
- Imports, exports count
- Debug, relocations, resources, signature, TLS presence

### Header Features (17)
- COFF: timestamp, machine, characteristics
- Optional: subsystem, DLL characteristics, versions, sizes

### Section Features (2)
- Section count and entries

## 🐛 Troubleshooting

### Model not found
```bash
# Train the model first
cd ../Model
python train_randomforest.py
```

### Port already in use
Change the port in `api_server.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
```

### Feature extraction error
Make sure the uploaded file is a valid PE file (.exe, .dll, .sys)

### Import error for feature_extractor
The `feature_extractor.py` file should be in the same `server` folder as `api_server.py`

## 📊 Performance

- **Prediction time**: <0.1 seconds per file
- **Max file size**: 100 MB (configurable)
- **Supported formats**: PE files (.exe, .dll, .sys)

## 📚 More Information

See the main project documentation:
- `../QUICK_START.md` - Quick start guide
- `../Model/README.md` - Model details
- `../IMPLEMENTATION_SUMMARY.md` - Implementation overview

