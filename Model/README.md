# Polymorphic Malware Detection System

A machine learning-based malware detection system that uses Random Forest classification on PE file metadata features. The system includes training scripts, feature extraction, and a REST API for scanning files.

## Overview

This system detects polymorphic malware by analyzing metadata features from PE (Portable Executable) files. Unlike signature-based detection, it focuses on structural and behavioral characteristics that are harder for malware to modify.

### Key Features

- **Metadata-based Detection**: Analyzes PE file headers, sections, and data directories
- **Random Forest Classifier**: Robust ensemble learning approach
- **REST API**: Easy integration with other systems
- **Feature Extraction**: Automatic extraction of 50+ features from PE files
- **High Performance**: Fast prediction with pre-trained model

## Architecture

```
┌─────────────────┐
│  Training Data  │
│  (EMBER 2018)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  train_randomforest.py  │
│  - Load dataset         │
│  - Train RF model       │
│  - Save model & features│
└────────┬────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Trained Model Files     │
│  - .pkl model            │
│  - feature_list.json     │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  API Server              │
│  (api_server.py)         │
│  - Load model            │
│  - Accept file uploads   │
│  - Extract features      │
│  - Make predictions      │
└──────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation**:
   ```bash
   python -c "import sklearn, pefile, fastapi; print('All dependencies installed!')"
   ```

## Usage

### Step 1: Train the Model

Train the Random Forest model on the EMBER 2018 dataset:

```bash
python train_randomforest.py
```

This will:
- Load the preprocessed dataset from `dataset/ember2018/processed/ember_train_sample_2pct.parquet`
- Train a Random Forest classifier
- Evaluate the model on test data
- Save the trained model to `randomforest_malware_detector.pkl`
- Save feature metadata to `feature_list.json`

**Expected Output**:
```
✅ Loaded X samples with Y columns
✅ Features shape: (X, Y)
🌲 Training Random Forest model...
📊 Evaluating model...
Accuracy: 0.XXXX
ROC-AUC Score: 0.XXXX
💾 Model saved to: Model/randomforest_malware_detector.pkl
```

### Step 2: Start the API Server

Launch the FastAPI server:

```bash
python api_server.py
```

The server will start on `http://localhost:8000`

**API Endpoints**:

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /scan` - Scan a file (basic results)
- `POST /scan/detailed` - Scan a file (detailed results with features)
- `GET /model/info` - Get model information

### Step 3: Scan Files

#### Using the Test Client

```bash
# Basic scan
python test_api_client.py path/to/file.exe

# Detailed scan with feature values
python test_api_client.py path/to/file.exe --detailed
```

#### Using cURL

```bash
# Basic scan
curl -X POST "http://localhost:8000/scan" \
  -F "file=@path/to/file.exe"

# Detailed scan
curl -X POST "http://localhost:8000/scan/detailed" \
  -F "file=@path/to/file.exe"
```

#### Using Python requests

```python
import requests

# Scan a file
with open('file.exe', 'rb') as f:
    files = {'file': ('file.exe', f, 'application/octet-stream')}
    response = requests.post('http://localhost:8000/scan', files=files)
    result = response.json()
    
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Malware probability: {result['malware_probability']:.2%}")
```

## Features Extracted

The system extracts the following categories of features from PE files:

### General Features (10 features)
- File size and virtual size
- Debug information presence
- Number of exports and imports
- Relocations, resources, signature, TLS presence
- Symbol count

### Header Features (16 features)
- COFF header: timestamp, machine type, characteristics
- Optional header: subsystem, DLL characteristics, magic number
- Version information: image, linker, OS, subsystem versions
- Size information: code, headers, heap commit

### Section Features (2 features)
- Section count
- Section entry count

### Data Directories Features (30 features)
- Size and virtual address for 15 data directories:
  - Export table, Import table, Resource table
  - Exception table, Certificate table, Base relocation table
  - Debug, Architecture, Global pointer
  - TLS table, Load config table, Bound import
  - IAT, Delay import descriptor, CLR runtime header

**Total: 58 metadata features**

## Model Performance

The Random Forest model is trained with the following parameters:
- `n_estimators`: 100 trees
- `max_depth`: 20
- `min_samples_split`: 5
- `min_samples_leaf`: 2

Expected performance metrics:
- **Accuracy**: ~95%+ (depends on dataset)
- **ROC-AUC**: ~0.98+ (depends on dataset)

## API Response Format

### Basic Scan Response

```json
{
  "filename": "sample.exe",
  "prediction": "malware",
  "confidence": 0.95,
  "malware_probability": 0.95,
  "benign_probability": 0.05,
  "timestamp": "2024-01-01T12:00:00",
  "features_extracted": 58,
  "file_size": 102400
}
```

### Detailed Scan Response

Includes all fields from basic scan plus:
- `top_features`: Top 10 most important features with their values
- `all_features`: All 58 extracted features

## File Structure

```
Model/
├── train_randomforest.py      # Training script
├── feature_extractor.py       # Feature extraction module
├── api_server.py              # FastAPI server
├── test_api_client.py         # Test client
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── randomforest_malware_detector.pkl  # Trained model (generated)
└── feature_list.json          # Feature metadata (generated)
```

## Troubleshooting

### Model not found error
- Make sure you've run `train_randomforest.py` first
- Check that `randomforest_malware_detector.pkl` exists in the Model directory

### Feature extraction error
- Ensure the uploaded file is a valid PE file (.exe, .dll, .sys)
- Check that `pefile` library is installed correctly

### API connection error
- Verify the API server is running on port 8000
- Check firewall settings if accessing from another machine

## Security Considerations

⚠️ **Important**: This API accepts file uploads. In production:
- Implement authentication and authorization
- Run in a sandboxed environment
- Add rate limiting
- Scan uploaded files in an isolated container
- Implement file type validation
- Add virus scanning before processing
- Use HTTPS for encrypted communication

## Future Improvements

- [ ] Add support for more file types (ELF, Mach-O)
- [ ] Implement batch scanning endpoint
- [ ] Add model retraining capability
- [ ] Create web UI for file uploads
- [ ] Add detailed logging and monitoring
- [ ] Implement model versioning
- [ ] Add explainability features (SHAP values)

## License

This project is for educational and research purposes.

## References

- EMBER Dataset: https://github.com/elastic/ember
- pefile library: https://github.com/erocarrera/pefile
- FastAPI: https://fastapi.tiangolo.com/

