# Fixes Applied to hybrid_training.ipynb

## Issues Fixed

### 1. **Missing Imports**
- **Problem**: `compute_class_weight` was used but not imported
- **Fix**: Added `from sklearn.utils.class_weight import compute_class_weight`
- **Problem**: `Bidirectional` layer was used but not imported  
- **Fix**: Added `Bidirectional` to the keras layers import

### 2. **Hard-coded Absolute Paths**
- **Problem**: Paths like `C:/Users/Admin/github-classroom/Caty175/Caty-PolyWatch/ML model/` were hard-coded
- **Fix**: Replaced with relative paths using `os.path.join()` and `BASE_DIR`
- **Before**: 
  ```python
  DATA_PATH = "C:/Users/Admin/github-classroom/Caty175/Caty-PolyWatch/ML model/Malware_Analysis.csv"
  rf_path = r"C:\Users\Admin\github-classroom\Caty175\Caty-PolyWatch\ML model\rf_model.joblib"
  ```
- **After**:
  ```python
  BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
  DATA_PATH = os.path.join(BASE_DIR, 'Malware_Analysis.csv')
  models_dir = os.path.join(BASE_DIR, 'models')
  rf_path = os.path.join(models_dir, 'rf_model.joblib')
  ```

### 3. **Variable Scope Issues**
- **Problem**: `MAX_NUM_WORDS` was defined conditionally but used unconditionally
- **Fix**: Moved initialization to the top of the dynamic features section
- **Problem**: `tokenizer` was used without proper null checks
- **Fix**: Added proper null checks and initialization

### 4. **Undefined Variables**
- **Problem**: Variables like `api_sequences`, `train_indices`, `test_indices` were referenced but never defined
- **Fix**: Removed or replaced these references with proper variable names

### 5. **Model Architecture Issues**
- **Problem**: Inconsistent model compilation and architecture definition
- **Fix**: Streamlined model creation with proper conditional logic
- **Problem**: Duplicate model training code
- **Fix**: Consolidated training logic into a single, clean section

### 6. **File Organization**
- **Problem**: Models were saved to hard-coded paths
- **Fix**: Created a `models` directory structure and used relative paths
- **Added**: Automatic directory creation with `os.makedirs(models_dir, exist_ok=True)`

## Key Improvements

### 1. **Cross-Platform Compatibility**
- All file paths now use `os.path.join()` for cross-platform compatibility
- No more Windows-specific hard-coded paths

### 2. **Better Error Handling**
- Added fallback logic for dataset location
- Improved error messages with actionable guidance

### 3. **Code Organization**
- Removed duplicate code sections
- Consolidated variable initialization
- Cleaner model architecture definition

### 4. **Robust Variable Management**
- Proper initialization of all variables before use
- Added null checks where necessary
- Fixed variable scope issues

## How to Use the Fixed Notebook

### 1. **File Structure**
Ensure your files are organized as follows:
```
ML model/
├── hybrid_training.ipynb
├── Malware_Analysis.csv
└── models/                 (created automatically)
    ├── rf_model.joblib
    ├── lstm_model.keras
    ├── lstm_model_best.h5
    └── predictions_ensemble.csv
```

### 2. **Required Dependencies**
Make sure you have installed:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib joblib
```

### 3. **Running the Notebook**
1. Place your `Malware_Analysis.csv` file in the same directory as the notebook
2. Run all cells in order
3. Models will be automatically saved to the `models/` subdirectory

### 4. **Expected Outputs**
- Random Forest model: `models/rf_model.joblib`
- LSTM model: `models/lstm_model.keras`
- Best checkpoint: `models/lstm_model_best.h5`
- Predictions: `models/predictions_ensemble.csv`
- Training plots and metrics displayed in notebook

## Error Prevention

### 1. **Path Issues**
- The notebook now works regardless of your system's directory structure
- No need to modify paths for different users or systems

### 2. **Missing Files**
- Clear error messages if dataset is not found
- Automatic directory creation for outputs

### 3. **Import Errors**
- All required imports are now properly included
- No more `NameError` exceptions for missing functions

### 4. **Variable Errors**
- All variables are properly initialized before use
- No more `NameError` exceptions for undefined variables

## Testing the Fixes

To verify the fixes work:

1. **Clean Environment Test**: Try running the notebook in a fresh Python environment
2. **Different Directory Test**: Move the notebook to a different folder and run it
3. **Cross-Platform Test**: Test on different operating systems (Windows/Mac/Linux)

The notebook should now run without the errors you were experiencing!
