#!/usr/bin/env python3
"""
LSTM + Feedforward Neural Network for Polymorphic Malware Detection
Uses behavioral features from Malware_Analysis_kaggle.csv dataset

Architecture:
- LSTM branch: Models API call sequences (temporal patterns)
- Feedforward branch: Models static behavioral features (file ops, DLL loading, etc.)
- Merged output: Combined prediction
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve, precision_recall_curve
)

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import plot_model

# ========== CONFIG ==========
DATASET_PATH = r"C:\Users\Admin\github-classroom\Caty175\poly_trial\dataset\Malware_Analysis_kaggle.csv"
MODEL_DIR = r"C:\Users\Admin\github-classroom\Caty175\poly_trial\Model"

# Directory structure for organized output
COMPONENTS_DIR = os.path.join(MODEL_DIR, "components")  # For .h5, .pkl, .json files
OUTPUT_DIR = os.path.join(MODEL_DIR, "output")  # For .png visualization files

MODEL_NAME = "lstm_behavioral_malware_detector.h5"
SCALER_NAME = "behavioral_scaler.pkl"
METADATA_NAME = "lstm_model_metadata.json"
CHECKPOINT_NAME = "best_model_checkpoint.h5"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # 20% of training data

# Model hyperparameters
LSTM_UNITS = 128
LSTM_DROPOUT = 0.3
DENSE_UNITS = 256
DENSE_DROPOUT = 0.4
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10  # Early stopping patience

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Create directories if they don't exist
os.makedirs(COMPONENTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("LSTM + FEEDFORWARD NEURAL NETWORK FOR BEHAVIORAL MALWARE DETECTION")
print("=" * 80)
print(f"\n🔧 TensorFlow version: {tf.__version__}")
print(f"🔧 GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"\n📁 Output directories:")
print(f"   - Components (models, scalers, metadata): {COMPONENTS_DIR}")
print(f"   - Outputs (visualizations): {OUTPUT_DIR}")


def load_and_preprocess_data(csv_path):
    """Load and preprocess the Kaggle malware dataset."""
    print(f"\n📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Categorize features
    api_cols = [c for c in df.columns if c.startswith('API_')]
    file_cols = [c for c in df.columns if 'file_' in c and c != 'file_opened']
    dll_cols = [c for c in df.columns if 'dll_freq_' in c]
    behavioral_cols = ['regkey_read', 'directory_enumerated', 'dll_loaded_count', 
                       'resolves_host', 'command_line']
    
    # Filter to existing columns
    behavioral_cols = [c for c in behavioral_cols if c in df.columns]
    
    print(f"\n📊 Feature breakdown:")
    print(f"   - API calls (LSTM input): {len(api_cols)}")
    print(f"   - File operations: {len(file_cols)}")
    print(f"   - DLL loading: {len(dll_cols)}")
    print(f"   - Other behavioral: {len(behavioral_cols)}")
    
    # Prepare labels (convert Score to binary: >= 5.0 = malware)
    if 'Score' in df.columns:
        y = (df['Score'] >= 5.0).astype(int).values
        print(f"\n✅ Labels created from Score column (threshold >= 5.0)")
        print(f"   Benign (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
        print(f"   Malware (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
    else:
        raise ValueError("No 'Score' column found in dataset!")
    
    # Prepare API call sequences (for LSTM)
    X_api = df[api_cols].fillna(0).values
    print(f"\n✅ API call features shape: {X_api.shape}")
    
    # Prepare static behavioral features (for Feedforward)
    static_cols = file_cols + dll_cols + behavioral_cols
    X_static = df[static_cols].fillna(0).values
    print(f"✅ Static behavioral features shape: {X_static.shape}")
    
    feature_metadata = {
        'api_features': api_cols,
        'static_features': static_cols,
        'num_api_features': len(api_cols),
        'num_static_features': len(static_cols),
        'label_threshold': 5.0
    }
    
    return X_api, X_static, y, feature_metadata


def create_lstm_feedforward_model(api_input_dim, static_input_dim, lstm_units=128, 
                                   dense_units=256, lstm_dropout=0.3, dense_dropout=0.4):
    """
    Create hybrid LSTM + Feedforward architecture.
    
    Architecture:
    - LSTM branch: Processes API call sequences
    - Feedforward branch: Processes static behavioral features
    - Merged: Concatenates both branches and makes final prediction
    """
    print(f"\n🏗️  Building LSTM + Feedforward model...")
    
    # ===== LSTM Branch (API Call Sequences) =====
    api_input = layers.Input(shape=(api_input_dim, 1), name='api_input')
    
    # LSTM layers with dropout
    lstm_out = layers.LSTM(lstm_units, return_sequences=True, name='lstm_1')(api_input)
    lstm_out = layers.Dropout(lstm_dropout, name='lstm_dropout_1')(lstm_out)
    lstm_out = layers.LSTM(lstm_units // 2, return_sequences=False, name='lstm_2')(lstm_out)
    lstm_out = layers.Dropout(lstm_dropout, name='lstm_dropout_2')(lstm_out)
    lstm_out = layers.Dense(64, activation='relu', name='lstm_dense')(lstm_out)
    
    # ===== Feedforward Branch (Static Behavioral Features) =====
    static_input = layers.Input(shape=(static_input_dim,), name='static_input')
    
    # Dense layers with batch normalization and dropout
    ff_out = layers.Dense(dense_units, activation='relu', name='ff_dense_1')(static_input)
    ff_out = layers.BatchNormalization(name='ff_bn_1')(ff_out)
    ff_out = layers.Dropout(dense_dropout, name='ff_dropout_1')(ff_out)
    
    ff_out = layers.Dense(dense_units // 2, activation='relu', name='ff_dense_2')(ff_out)
    ff_out = layers.BatchNormalization(name='ff_bn_2')(ff_out)
    ff_out = layers.Dropout(dense_dropout, name='ff_dropout_2')(ff_out)
    
    ff_out = layers.Dense(64, activation='relu', name='ff_dense_3')(ff_out)
    
    # ===== Merge Both Branches =====
    merged = layers.Concatenate(name='merge')([lstm_out, ff_out])
    
    # Final classification layers
    merged = layers.Dense(128, activation='relu', name='merged_dense_1')(merged)
    merged = layers.Dropout(0.3, name='merged_dropout')(merged)
    merged = layers.Dense(64, activation='relu', name='merged_dense_2')(merged)
    
    # Output layer (binary classification)
    output = layers.Dense(1, activation='sigmoid', name='output')(merged)
    
    # Create model
    model = models.Model(inputs=[api_input, static_input], outputs=output, name='LSTM_Feedforward_Malware_Detector')
    
    print(f"✅ Model created successfully!")
    print(f"\n📐 Architecture summary:")
    print(f"   - LSTM branch: {api_input_dim} API features → LSTM({lstm_units}) → LSTM({lstm_units//2}) → Dense(64)")
    print(f"   - Feedforward branch: {static_input_dim} static features → Dense({dense_units}) → Dense({dense_units//2}) → Dense(64)")
    print(f"   - Merged: Concatenate → Dense(128) → Dense(64) → Output(1)")
    
    return model


def train_model(model, X_api_train, X_static_train, y_train, 
                X_api_val, X_static_val, y_val, 
                batch_size=32, epochs=50, patience=10):
    """Train the model with early stopping and model checkpointing."""
    print(f"\n🚀 Training model...")
    print(f"   Batch size: {batch_size}")
    print(f"   Max epochs: {epochs}")
    print(f"   Early stopping patience: {patience}")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'), 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks
    checkpoint_path = os.path.join(COMPONENTS_DIR, CHECKPOINT_NAME)
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        [X_api_train, X_static_train],
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X_api_val, X_static_val], y_val),
        callbacks=callback_list,
        verbose=1
    )
    
    print(f"\n✅ Training complete!")
    
    return history


def evaluate_model(model, X_api_test, X_static_test, y_test):
    """Evaluate model performance."""
    print(f"\n📊 Evaluating model on test set...")
    
    # Predictions
    y_pred_proba = model.predict([X_api_test, X_static_test], verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ ROC-AUC Score: {roc_auc:.4f}")
    print(f"\n📊 Confusion Matrix:")
    print(f"   TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
    print(f"   FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")
    print(f"\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malware']))
    
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'true_positives': int(cm[1,1]),
        'true_negatives': int(cm[0,0]),
        'false_positives': int(cm[0,1]),
        'false_negatives': int(cm[1,0])
    }
    
    return metrics, y_pred_proba


def main():
    """Main training pipeline."""
    
    # Step 1: Load and preprocess data
    X_api, X_static, y, feature_metadata = load_and_preprocess_data(DATASET_PATH)
    
    # Step 2: Split data (train/val/test)
    print(f"\n🔀 Splitting data...")
    X_api_train, X_api_test, X_static_train, X_static_test, y_train, y_test = train_test_split(
        X_api, X_static, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_api_train, X_api_val, X_static_train, X_static_val, y_train, y_val = train_test_split(
        X_api_train, X_static_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )
    
    print(f"   Train: {len(y_train)} samples")
    print(f"   Val:   {len(y_val)} samples")
    print(f"   Test:  {len(y_test)} samples")
    
    # Step 3: Normalize features
    print(f"\n🔧 Normalizing features...")
    api_scaler = RobustScaler()
    static_scaler = RobustScaler()
    
    X_api_train = api_scaler.fit_transform(X_api_train)
    X_api_val = api_scaler.transform(X_api_val)
    X_api_test = api_scaler.transform(X_api_test)
    
    X_static_train = static_scaler.fit_transform(X_static_train)
    X_static_val = static_scaler.transform(X_static_val)
    X_static_test = static_scaler.transform(X_static_test)
    
    # Reshape API features for LSTM (samples, timesteps, features)
    X_api_train = X_api_train.reshape(X_api_train.shape[0], X_api_train.shape[1], 1)
    X_api_val = X_api_val.reshape(X_api_val.shape[0], X_api_val.shape[1], 1)
    X_api_test = X_api_test.reshape(X_api_test.shape[0], X_api_test.shape[1], 1)
    
    print(f"   API features (LSTM): {X_api_train.shape}")
    print(f"   Static features (FF): {X_static_train.shape}")

    # Step 4: Build model
    model = create_lstm_feedforward_model(
        api_input_dim=X_api_train.shape[1],
        static_input_dim=X_static_train.shape[1],
        lstm_units=LSTM_UNITS,
        dense_units=DENSE_UNITS,
        lstm_dropout=LSTM_DROPOUT,
        dense_dropout=DENSE_DROPOUT
    )

    # Print model summary
    print(f"\n📋 Model Summary:")
    model.summary()

    # Step 5: Train model
    history = train_model(
        model,
        X_api_train, X_static_train, y_train,
        X_api_val, X_static_val, y_val,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE
    )

    # Step 6: Evaluate model
    metrics, y_pred_proba = evaluate_model(model, X_api_test, X_static_test, y_test)

    # Step 7: Save model and metadata
    print(f"\n💾 Saving model and artifacts...")

    # Save model to components folder
    model_path = os.path.join(COMPONENTS_DIR, MODEL_NAME)
    model.save(model_path)
    print(f"   ✅ Model saved to: {model_path}")

    # Save scalers to components folder
    import joblib
    scaler_path = os.path.join(COMPONENTS_DIR, SCALER_NAME)
    joblib.dump({'api_scaler': api_scaler, 'static_scaler': static_scaler}, scaler_path)
    print(f"   ✅ Scalers saved to: {scaler_path}")

    # Save metadata to components folder
    metadata = {
        'model_name': MODEL_NAME,
        'training_date': datetime.now().isoformat(),
        'dataset_path': DATASET_PATH,
        'total_samples': len(y),
        'train_samples': len(y_train),
        'val_samples': len(y_val),
        'test_samples': len(y_test),
        'feature_metadata': feature_metadata,
        'hyperparameters': {
            'lstm_units': LSTM_UNITS,
            'lstm_dropout': LSTM_DROPOUT,
            'dense_units': DENSE_UNITS,
            'dense_dropout': DENSE_DROPOUT,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'patience': PATIENCE
        },
        'metrics': metrics,
        'training_history': {
            'final_epoch': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'best_val_auc': float(max(history.history['val_auc']))
        }
    }

    metadata_path = os.path.join(COMPONENTS_DIR, METADATA_NAME)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata saved to: {metadata_path}")

    # Step 8: Plot training history (save to output folder)
    plot_training_history(history, OUTPUT_DIR)

    # Step 9: Plot ROC curve (save to output folder)
    plot_roc_curve(y_test, y_pred_proba, OUTPUT_DIR)

    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 Saved artifacts:")
    print(f"   📦 Components folder ({COMPONENTS_DIR}):")
    print(f"      - Model: {MODEL_NAME}")
    print(f"      - Scalers: {SCALER_NAME}")
    print(f"      - Metadata: {METADATA_NAME}")
    print(f"      - Checkpoint: {CHECKPOINT_NAME}")
    print(f"\n   📊 Output folder ({OUTPUT_DIR}):")
    print(f"      - Training history: training_history.png")
    print(f"      - ROC curve: roc_curve.png")

    return model, metadata


def plot_training_history(history, save_dir):
    """Plot training history (loss and accuracy)."""
    print(f"\n📊 Plotting training history...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # AUC
    axes[1, 0].plot(history.history['auc'], label='Train AUC')
    axes[1, 0].plot(history.history['val_auc'], label='Val AUC')
    axes[1, 0].set_title('Model AUC')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Precision & Recall
    axes[1, 1].plot(history.history['precision'], label='Train Precision')
    axes[1, 1].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Training history saved to: {plot_path}")
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_dir):
    """Plot ROC curve."""
    print(f"\n📊 Plotting ROC curve...")

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ ROC curve saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()

