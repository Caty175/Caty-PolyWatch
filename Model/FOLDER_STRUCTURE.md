# LSTM Behavioral Model - Folder Structure

## 📁 Organized Output Structure

The LSTM behavioral malware detection model now saves files in an organized folder structure for better project management.

---

## 🗂️ Directory Layout

```
Model/
├── components/              # Model components (.h5, .pkl, .json)
│   ├── lstm_behavioral_malware_detector.h5
│   ├── behavioral_scaler.pkl
│   ├── lstm_model_metadata.json
│   └── best_model_checkpoint.h5
│
├── output/                  # Training outputs (.png visualizations)
│   ├── training_history.png
│   └── roc_curve.png
│
├── train_lstm_behavioral.py
├── predict_lstm_behavioral.py
├── LSTM_BEHAVIORAL_README.md
└── FOLDER_STRUCTURE.md (this file)
```

---

## 📦 Components Folder

**Location**: `Model/components/`

**Purpose**: Stores all model artifacts required for inference

### Files:

| File | Type | Size | Description |
|------|------|------|-------------|
| `lstm_behavioral_malware_detector.h5` | Model | ~800 KB | Trained Keras model with weights |
| `behavioral_scaler.pkl` | Scaler | ~50 KB | RobustScaler for feature normalization |
| `lstm_model_metadata.json` | Metadata | ~15 KB | Training configuration and metrics |
| `best_model_checkpoint.h5` | Checkpoint | ~800 KB | Best model during training (val_auc) |

### Why Components Folder?

- ✅ **Deployment Ready**: All files needed for inference in one place
- ✅ **Version Control**: Easy to track model versions
- ✅ **Portability**: Copy entire folder to deploy elsewhere
- ✅ **Clean Separation**: Model artifacts separate from code

---

## 📊 Output Folder

**Location**: `Model/output/`

**Purpose**: Stores training visualizations and analysis outputs

### Files:

| File | Type | Size | Description |
|------|------|------|-------------|
| `training_history.png` | Image | ~200 KB | Loss, accuracy, AUC, precision/recall curves |
| `roc_curve.png` | Image | ~100 KB | ROC curve with AUC score |

### Why Output Folder?

- ✅ **Analysis**: Keep training visualizations organized
- ✅ **Documentation**: Easy to include in reports
- ✅ **Comparison**: Compare different training runs
- ✅ **Clean Separation**: Outputs separate from model components

---

## 🔄 How It Works

### Training Script (`train_lstm_behavioral.py`)

When you run the training script:

```bash
python Model/train_lstm_behavioral.py
```

**Automatic folder creation:**
```python
COMPONENTS_DIR = os.path.join(MODEL_DIR, "components")
OUTPUT_DIR = os.path.join(MODEL_DIR, "output")

os.makedirs(COMPONENTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

**Files saved to components:**
- Model: `components/lstm_behavioral_malware_detector.h5`
- Scalers: `components/behavioral_scaler.pkl`
- Metadata: `components/lstm_model_metadata.json`
- Checkpoint: `components/best_model_checkpoint.h5`

**Files saved to output:**
- Training history: `output/training_history.png`
- ROC curve: `output/roc_curve.png`

---

### Prediction Script (`predict_lstm_behavioral.py`)

The prediction script automatically detects the folder structure:

```python
class BehavioralMalwareDetector:
    def __init__(self, model_dir=MODEL_DIR, components_dir=None):
        # Auto-detect components folder
        if components_dir:
            self.components_dir = components_dir
        elif os.path.exists(os.path.join(model_dir, "components")):
            self.components_dir = os.path.join(model_dir, "components")
        else:
            self.components_dir = model_dir  # Backward compatibility
```

**Backward Compatibility:**
- ✅ Works with new folder structure (`components/` subfolder)
- ✅ Falls back to old structure (files in `Model/` root)
- ✅ No breaking changes for existing code

---

## 🚀 Usage Examples

### 1. Training a New Model

```bash
python Model/train_lstm_behavioral.py
```

**Output:**
```
📁 Output directories:
   - Components (models, scalers, metadata): Model/components
   - Outputs (visualizations): Model/output

...training...

📁 Saved artifacts:
   📦 Components folder (Model/components):
      - Model: lstm_behavioral_malware_detector.h5
      - Scalers: behavioral_scaler.pkl
      - Metadata: lstm_model_metadata.json
      - Checkpoint: best_model_checkpoint.h5

   📊 Output folder (Model/output):
      - Training history: training_history.png
      - ROC curve: roc_curve.png
```

---

### 2. Making Predictions

```python
from Model.predict_lstm_behavioral import BehavioralMalwareDetector

# Automatically loads from components folder
detector = BehavioralMalwareDetector()

# Make predictions
result = detector.predict(data)
```

**Output:**
```
🔧 Loading LSTM behavioral malware detector...
   ✅ Model loaded from Model/components/lstm_behavioral_malware_detector.h5
   ✅ Scalers loaded from Model/components/behavioral_scaler.pkl
   ✅ Metadata loaded from Model/components/lstm_model_metadata.json
```

---

### 3. Deploying the Model

To deploy the model to another system:

**Option 1: Copy components folder**
```bash
# Copy entire components folder
cp -r Model/components /path/to/deployment/
```

**Option 2: Package as ZIP**
```bash
# Create deployment package
cd Model
zip -r lstm_model_deployment.zip components/
```

**On deployment system:**
```python
detector = BehavioralMalwareDetector(
    components_dir="/path/to/deployment/components"
)
```

---

## 📋 Benefits of This Structure

### 1. **Organization**
- Clear separation between model artifacts and outputs
- Easy to find specific files
- Professional project structure

### 2. **Version Control**
- Git-friendly structure
- Easy to track changes
- Can add `.gitignore` for large files

### 3. **Deployment**
- Copy `components/` folder for deployment
- No need to copy training outputs
- Minimal deployment footprint

### 4. **Collaboration**
- Team members know where to find files
- Consistent structure across projects
- Easy onboarding for new developers

### 5. **Maintenance**
- Easy to clean up old training outputs
- Keep multiple model versions organized
- Archive old models without clutter

---

## 🔧 Customization

### Change Output Directories

Edit `train_lstm_behavioral.py`:

```python
# Custom directories
COMPONENTS_DIR = os.path.join(MODEL_DIR, "my_models")
OUTPUT_DIR = os.path.join(MODEL_DIR, "my_outputs")
```

### Use Different Component Locations

```python
# Load from custom location
detector = BehavioralMalwareDetector(
    components_dir="/custom/path/to/components"
)
```

---

## 📝 File Descriptions

### lstm_behavioral_malware_detector.h5
- **Type**: Keras HDF5 model file
- **Contains**: Model architecture + trained weights
- **Required for**: Inference/prediction
- **Size**: ~800 KB (203,009 parameters)

### behavioral_scaler.pkl
- **Type**: Joblib pickle file
- **Contains**: RobustScaler objects for API and static features
- **Required for**: Feature normalization before prediction
- **Size**: ~50 KB

### lstm_model_metadata.json
- **Type**: JSON configuration file
- **Contains**: 
  - Feature lists (API features, static features)
  - Hyperparameters (LSTM units, dropout, learning rate, etc.)
  - Training metrics (accuracy, ROC-AUC, confusion matrix)
  - Training history (epochs, losses, best validation AUC)
- **Required for**: Understanding model configuration
- **Size**: ~15 KB

### best_model_checkpoint.h5
- **Type**: Keras HDF5 model file
- **Contains**: Best model weights during training (highest val_auc)
- **Required for**: Optional - can be used instead of final model
- **Size**: ~800 KB

### training_history.png
- **Type**: PNG image (300 DPI)
- **Contains**: 4 subplots showing:
  - Loss curves (train vs validation)
  - Accuracy curves
  - AUC curves
  - Precision & Recall curves
- **Size**: ~200 KB

### roc_curve.png
- **Type**: PNG image (300 DPI)
- **Contains**: ROC curve with AUC score
- **Size**: ~100 KB

---

## 🎯 Best Practices

### 1. **Version Control**

Add to `.gitignore`:
```
# Large model files (optional - use Git LFS instead)
Model/components/*.h5
Model/components/*.pkl

# Training outputs (regenerate as needed)
Model/output/*.png
```

Use Git LFS for model files:
```bash
git lfs track "Model/components/*.h5"
git lfs track "Model/components/*.pkl"
```

### 2. **Model Versioning**

Create versioned folders:
```
Model/
├── components/
│   ├── v1.0/
│   │   ├── lstm_behavioral_malware_detector.h5
│   │   ├── behavioral_scaler.pkl
│   │   └── lstm_model_metadata.json
│   ├── v1.1/
│   │   └── ...
│   └── latest/  (symlink to current version)
```

### 3. **Backup Strategy**

Backup components folder regularly:
```bash
# Automated backup script
DATE=$(date +%Y%m%d)
tar -czf backups/lstm_model_$DATE.tar.gz Model/components/
```

### 4. **Documentation**

Keep a changelog:
```
Model/components/CHANGELOG.md

## v1.0 (2025-11-08)
- Initial LSTM + Feedforward model
- 78.63% accuracy, 0.8592 ROC-AUC
- 261 API features + 59 static features
```

---

## 🔍 Troubleshooting

### Issue: "Model not found"

**Solution**: Check if components folder exists
```python
import os
components_dir = "Model/components"
if not os.path.exists(components_dir):
    print("Components folder not found! Run training first.")
```

### Issue: "Backward compatibility"

**Solution**: Prediction script automatically handles both structures
```python
# Works with both:
# - Model/components/lstm_behavioral_malware_detector.h5 (new)
# - Model/lstm_behavioral_malware_detector.h5 (old)
detector = BehavioralMalwareDetector()
```

---

## ✅ Summary

The new folder structure provides:

- ✅ **Organized**: Clear separation of components and outputs
- ✅ **Professional**: Industry-standard project structure
- ✅ **Portable**: Easy deployment and version control
- ✅ **Backward Compatible**: Works with old and new structures
- ✅ **Maintainable**: Easy to manage and update

**All model artifacts in one place, all outputs in another!** 🎉

