# Polymorphic Malware Detection System - Implementation Summary

## Overview

Successfully implemented a complete machine learning pipeline for detecting polymorphic malware using Random Forest classification on PE file metadata features.

## Key Achievements

### ✅ Problem Resolution

**Original Issues:**
1. ❌ Too few samples (491) vs too many features (5,578) → **Severe overfitting risk**
2. ❌ Class imbalance (2:1 ratio)
3. ❌ High dimensionality problem

**Solutions Implemented:**
1. ✅ **Used full dataset**: 24,562 samples (19,649 training, 4,913 test)
2. ✅ **Feature selection**: Reduced from 5,578 to **29 metadata features** (polymorphism-relevant)
3. ✅ **Class weighting**: Implemented `class_weight='balanced'` to handle 2.35:1 imbalance
4. ✅ **Optional SMOTE**: Available for more aggressive balancing if needed
5. ✅ **Optional PCA**: Available for further dimensionality reduction if needed

### 📊 Model Performance

**Excellent Results:**
- **Accuracy**: 92.94%
- **ROC-AUC**: 97.79%
- **Precision (Benign)**: 95%
- **Recall (Benign)**: 95%
- **Precision (Malware)**: 89%
- **Recall (Malware)**: 87%

**Confusion Matrix:**
```
                Predicted
              Benign  Malware
Actual Benign   3289     156
       Malware   191    1277
```

**Key Insights:**
- Very low false positive rate (156/3445 = 4.5%)
- Low false negative rate (191/1468 = 13%)
- Balanced performance across both classes
- High ROC-AUC indicates excellent discrimination ability

## Architecture

### 1. Training Pipeline (`Model/train_randomforest.py`)

**Features:**
- Loads full EMBER 2018 dataset (24,562 samples)
- Filters to 29 metadata features only
- Handles class imbalance with balanced class weights
- Trains Random Forest with 200 trees
- Saves model and feature metadata

**Configuration:**
```python
# Dataset
DATASET_PATH = "dataset/ember2018/processed/ember_train_full.parquet"

# Feature Selection
METADATA_PREFIXES = ('general.', 'header.', 'section.', 'datadirectories.')

# Class Imbalance
USE_SMOTE = False  # Using class_weight='balanced' instead
USE_PCA = False    # Not needed with only 29 features

# Random Forest Parameters
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 30,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}
```

### 2. Feature Extraction (`Model/feature_extractor.py`)

**Extracts 29 metadata features from PE files:**

**General Features (10):**
- `general.size` - File size
- `general.vsize` - Virtual size
- `general.has_debug` - Debug info presence
- `general.exports` - Number of exports
- `general.imports` - Number of imports
- `general.has_relocations` - Relocations presence
- `general.has_resources` - Resources presence
- `general.has_signature` - Digital signature presence
- `general.has_tls` - TLS presence
- `general.symbols` - Symbol count

**Header Features (17):**
- COFF header: timestamp, machine, characteristics
- Optional header: subsystem, DLL characteristics, magic
- Version info: image, linker, OS, subsystem versions
- Size info: code, headers, heap commit

**Section Features (2):**
- `section.entry` - Section entry count
- `section.sections` - Number of sections

### 3. API Server (`Model/api_server.py`)

**FastAPI REST API with endpoints:**

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /scan` - Scan a file (basic results)
- `POST /scan/detailed` - Scan with feature details
- `GET /model/info` - Model information

**Features:**
- Accepts PE file uploads
- Extracts features using `feature_extractor.py`
- Returns prediction with confidence scores
- Handles errors gracefully
- Validates file size and format

### 4. Test Client (`Model/test_api_client.py`)

**Command-line client for testing:**
```bash
# Basic scan
python test_api_client.py file.exe

# Detailed scan with features
python test_api_client.py file.exe --detailed
```

## Top Important Features

The model identified these as the most important features for detection:

1. **header.coff.timestamp** (20.18%) - Compilation timestamp
2. **general.vsize** (9.97%) - Virtual size
3. **general.imports** (9.52%) - Number of imports
4. **general.size** (8.85%) - File size
5. **header.optional.sizeof_code** (8.54%) - Code section size
6. **header.optional.minor_linker_version** (6.37%)
7. **header.optional.major_linker_version** (4.69%)
8. **general.has_tls** (4.65%) - TLS presence
9. **header.optional.major_operating_system_version** (4.62%)
10. **general.exports** (3.72%) - Number of exports

## Why These Features Work for Polymorphic Malware

**Polymorphic malware** changes its code signature but maintains:
1. **Structural characteristics** - PE header structure
2. **Behavioral patterns** - Import/export patterns
3. **Size relationships** - Code vs. data ratios
4. **Compilation artifacts** - Linker versions, timestamps

These metadata features are **harder to modify** without breaking functionality, making them effective for detection.

## Usage Instructions

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python Model/train_randomforest.py
```

**Output:**
- `Model/randomforest_malware_detector.pkl` - Trained model
- `Model/feature_list.json` - Feature metadata

### Step 3: Start the API Server

```bash
python Model/api_server.py
```

Server runs on `http://localhost:8000`

### Step 4: Scan Files

**Using test client:**
```bash
python Model/test_api_client.py path/to/file.exe
```

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/scan" -F "file=@file.exe"
```

**Using Python:**
```python
import requests

with open('file.exe', 'rb') as f:
    files = {'file': ('file.exe', f)}
    response = requests.post('http://localhost:8000/scan', files=files)
    result = response.json()
    
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Files Created

```
Model/
├── train_randomforest.py          # Training script
├── feature_extractor.py           # Feature extraction module
├── api_server.py                  # FastAPI server
├── test_api_client.py             # Test client
├── requirements.txt               # Dependencies (in root)
├── README.md                      # Documentation
├── randomforest_malware_detector.pkl  # Trained model (generated)
└── feature_list.json              # Feature metadata (generated)

IMPLEMENTATION_SUMMARY.md          # This file
check_data.py                      # Data inspection script
```

## Advanced Configuration

### Enable SMOTE for More Aggressive Balancing

In `Model/train_randomforest.py`:
```python
USE_SMOTE = True  # Will oversample minority class
```

### Enable PCA for Dimensionality Reduction

In `Model/train_randomforest.py`:
```python
USE_PCA = True
PCA_COMPONENTS = 20  # Reduce to 20 components
```

## Security Considerations

⚠️ **Production Deployment:**
- Run API in sandboxed environment
- Implement authentication/authorization
- Add rate limiting
- Use HTTPS
- Validate file types strictly
- Scan uploads with antivirus first
- Implement logging and monitoring

## Performance Metrics Summary

| Metric | Value |
|--------|-------|
| **Accuracy** | 92.94% |
| **ROC-AUC** | 97.79% |
| **Precision (Benign)** | 95% |
| **Recall (Benign)** | 95% |
| **Precision (Malware)** | 89% |
| **Recall (Malware)** | 87% |
| **Training Samples** | 19,649 |
| **Test Samples** | 4,913 |
| **Features** | 29 |
| **Training Time** | ~2 seconds |
| **Prediction Time** | <0.1 seconds |

## Conclusion

Successfully created a production-ready malware detection system that:
- ✅ Solves the dimensionality problem (29 features vs 5,578)
- ✅ Uses sufficient training data (24,562 samples vs 491)
- ✅ Handles class imbalance effectively
- ✅ Achieves excellent performance (92.94% accuracy, 97.79% ROC-AUC)
- ✅ Provides easy-to-use API for integration
- ✅ Extracts same features from uploaded files
- ✅ Ready for deployment and testing

The system is now ready to scan PE files via the API and detect polymorphic malware with high accuracy!

