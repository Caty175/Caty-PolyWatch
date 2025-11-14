#!/usr/bin/env python3
"""
Train Random Forest model on polymorphic malware metadata features.
This script:
1. Loads the preprocessed EMBER 2018 dataset (FULL dataset for better training)
2. Filters to metadata features only (polymorphism-relevant)
3. Handles class imbalance with class weighting
4. Trains a Random Forest classifier
5. Evaluates the model
6. Saves the trained model and feature list for API usage
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib
import json
import os
from datetime import datetime

# ========== CONFIG ==========
# Use FULL dataset for better training (24k+ samples)
DATASET_PATH = r"C:\Users\Admin\github-classroom\Caty175\poly_trial\dataset\ember2018\processed\ember_train_full.parquet"
MODEL_DIR = r"C:\Users\Admin\github-classroom\Caty175\poly_trial\Model"
MODEL_NAME = "randomforest_malware_detector.pkl"
FEATURE_LIST_NAME = "feature_list.json"
PCA_MODEL_NAME = "pca_transformer.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature selection: Use only metadata features (polymorphism-relevant)
# These are harder for malware to modify without breaking functionality
METADATA_PREFIXES = ('general.', 'header.', 'section.', 'datadirectories.')

# Class imbalance handling
USE_SMOTE = False  # Set to True to use SMOTE, False to use class weighting
USE_PCA = False  # Set to True to apply PCA dimensionality reduction
PCA_COMPONENTS = 20  # Number of PCA components if USE_PCA is True

# Random Forest hyperparameters
RF_PARAMS = {
    'n_estimators': 200,  # Increased for better performance
    'max_depth': 30,  # Increased depth for more complex patterns
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',  # Use sqrt of features at each split
    'class_weight': 'balanced' if not USE_SMOTE else None,  # Handle imbalance
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': 1
}
# ============================


def load_data(parquet_path):
    """Load the preprocessed dataset."""
    print(f"📂 Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
    return df


def prepare_features(df):
    """
    Prepare features and labels for training.
    Filters to metadata features only (polymorphism-relevant).
    """
    print("\n🔍 Preparing features...")

    # Filter to metadata features only
    all_cols = [col for col in df.columns if col not in ['label', 'sha256']]
    metadata_cols = [col for col in all_cols if any(col.startswith(p) for p in METADATA_PREFIXES)]

    print(f"📊 Total columns in dataset: {len(all_cols)}")
    print(f"📊 Metadata feature columns: {len(metadata_cols)}")
    print(f"   Metadata features: {metadata_cols}")

    # Extract features and labels
    X = df[metadata_cols].copy()
    y = df['label'].copy()

    # Handle missing values (fill with 0)
    X = X.fillna(0)

    # Convert to numeric (in case there are any string values)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    print(f"\n✅ Features shape: {X.shape}")
    print(f"✅ Labels distribution:")
    print(f"   Benign (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"   Malware (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
    print(f"   Imbalance ratio: {(y == 0).sum() / (y == 1).sum():.2f}:1")

    return X, y, metadata_cols


def apply_smote(X_train, y_train):
    """Apply SMOTE to balance classes."""
    print("\n⚖️ Applying SMOTE to balance classes...")
    print(f"   Before SMOTE:")
    print(f"     Benign (0): {(y_train == 0).sum()}")
    print(f"     Malware (1): {(y_train == 1).sum()}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"   After SMOTE:")
    print(f"     Benign (0): {(y_resampled == 0).sum()}")
    print(f"     Malware (1): {(y_resampled == 1).sum()}")

    return X_resampled, y_resampled


def apply_pca(X_train, X_test, n_components):
    """Apply PCA dimensionality reduction."""
    print(f"\n🔬 Applying PCA (reducing to {n_components} components)...")
    print(f"   Original shape: {X_train.shape}")

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"   Reduced shape: {X_train_pca.shape}")
    print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    return X_train_pca, X_test_pca, pca


def train_model(X_train, y_train):
    """Train Random Forest classifier."""
    print("\n🌲 Training Random Forest model...")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Parameters: {RF_PARAMS}")

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    print("✅ Model training complete!")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    print("\n📊 Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malware']))
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }


def get_feature_importance(model, feature_cols, top_n=20):
    """Get and display feature importance."""
    print(f"\n🔝 Top {top_n} Most Important Features:")
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(feature_importance_df.head(top_n).to_string(index=False))
    
    return feature_importance_df


def save_model_and_metadata(model, feature_cols, metrics, feature_importance_df, pca_model=None):
    """Save the trained model, feature list, and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")

    # Save PCA model if used
    pca_path = None
    if pca_model is not None:
        pca_path = os.path.join(MODEL_DIR, PCA_MODEL_NAME)
        joblib.dump(pca_model, pca_path)
        print(f"💾 PCA transformer saved to: {pca_path}")

    # Save feature list (critical for API to extract features in the same order)
    feature_list_path = os.path.join(MODEL_DIR, FEATURE_LIST_NAME)
    feature_metadata = {
        'features': feature_cols,
        'num_features': len(feature_cols),
        'use_pca': USE_PCA,
        'pca_components': PCA_COMPONENTS if USE_PCA else None,
        'use_smote': USE_SMOTE,
        'training_date': datetime.now().isoformat(),
        'model_params': RF_PARAMS,
        'metrics': metrics,
        'top_features': feature_importance_df.head(20)[['feature', 'importance']].to_dict('records'),
        'metadata_prefixes': list(METADATA_PREFIXES)
    }

    with open(feature_list_path, 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    print(f"💾 Feature metadata saved to: {feature_list_path}")

    return model_path, feature_list_path


def main():
    """Main training pipeline."""
    print("="*70)
    print("RANDOM FOREST MALWARE DETECTOR - TRAINING PIPELINE")
    print("="*70)
    print(f"Configuration:")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Metadata features only: {METADATA_PREFIXES}")
    print(f"  Use SMOTE: {USE_SMOTE}")
    print(f"  Use PCA: {USE_PCA}")
    if USE_PCA:
        print(f"  PCA components: {PCA_COMPONENTS}")
    print(f"  Class weighting: {'balanced' if not USE_SMOTE else 'None (using SMOTE)'}")
    print("="*70)

    # 1. Load data
    df = load_data(DATASET_PATH)

    # 2. Prepare features (filter to metadata only)
    X, y, feature_cols = prepare_features(df)

    # 3. Split data
    print(f"\n✂️ Splitting data (test size: {TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # 4. Handle class imbalance with SMOTE (if enabled)
    pca_model = None
    if USE_SMOTE:
        X_train, y_train = apply_smote(X_train, y_train)

    # 5. Apply PCA (if enabled)
    if USE_PCA:
        X_train, X_test, pca_model = apply_pca(X_train, X_test, PCA_COMPONENTS)

    # 6. Train model
    model = train_model(X_train, y_train)

    # 7. Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # 8. Feature importance
    if USE_PCA:
        # For PCA, feature importance is on PCA components
        pca_feature_names = [f'PC{i+1}' for i in range(PCA_COMPONENTS)]
        feature_importance_df = get_feature_importance(model, pca_feature_names)
    else:
        feature_importance_df = get_feature_importance(model, feature_cols)

    # 9. Save model and metadata
    model_path, feature_list_path = save_model_and_metadata(
        model, feature_cols, metrics, feature_importance_df, pca_model
    )

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Model ready for deployment at: {model_path}")
    print(f"Feature list for API at: {feature_list_path}")
    if pca_model:
        print(f"PCA transformer at: {os.path.join(MODEL_DIR, PCA_MODEL_NAME)}")
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features used: {len(feature_cols)} metadata features")
    print("\nModel Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print("\nNext steps:")
    print("1. Start the API server: python api_server.py")
    print("2. Test with: python test_api_client.py <file.exe>")
    print("="*70)


if __name__ == "__main__":
    main()

