#!/usr/bin/env python3
"""
Adaptive Retraining Pipeline for Malware Detection Models
Retrains models with new samples while preserving knowledge from original training data
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import RobustScaler

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.adaptive_learning import AdaptiveLearningManager
from server.performance_monitor import PerformanceMonitor

# Import original training functions - use importlib to handle path issues
import importlib.util

# Load train_randomforest module
rf_train_path = os.path.join(os.path.dirname(__file__), "train_randomforest.py")
rf_spec = importlib.util.spec_from_file_location("train_randomforest", rf_train_path)
rf_module = importlib.util.module_from_spec(rf_spec)
rf_spec.loader.exec_module(rf_module)

load_rf_data = rf_module.load_data
prepare_rf_features = rf_module.prepare_features
RF_PARAMS = rf_module.RF_PARAMS
FEATURE_LIST_NAME = rf_module.FEATURE_LIST_NAME
RANDOM_STATE = rf_module.RANDOM_STATE
MODEL_DIR = rf_module.MODEL_DIR
COMPONENTS_DIR = os.path.join(MODEL_DIR, "components")
RF_MODEL_NAME = rf_module.MODEL_NAME

# Load train_lstm_behavioral module
lstm_train_path = os.path.join(os.path.dirname(__file__), "train_lstm_behavioral.py")
lstm_spec = importlib.util.spec_from_file_location("train_lstm_behavioral", lstm_train_path)
lstm_module = importlib.util.module_from_spec(lstm_spec)
lstm_spec.loader.exec_module(lstm_module)

load_lstm_data = lstm_module.load_and_preprocess_data
create_lstm_feedforward_model = lstm_module.create_lstm_feedforward_model
LSTM_UNITS = lstm_module.LSTM_UNITS
DENSE_UNITS = lstm_module.DENSE_UNITS
LSTM_DROPOUT = lstm_module.LSTM_DROPOUT
DENSE_DROPOUT = lstm_module.DENSE_DROPOUT
LEARNING_RATE = lstm_module.LEARNING_RATE
BATCH_SIZE = lstm_module.BATCH_SIZE
EPOCHS = lstm_module.EPOCHS
PATIENCE = lstm_module.PATIENCE
LSTM_MODEL_NAME = lstm_module.MODEL_NAME
LSTM_SCALER_NAME = lstm_module.SCALER_NAME
LSTM_METADATA_NAME = lstm_module.METADATA_NAME


class AdaptiveRetrainer:
    """
    Handles adaptive retraining of both Random Forest and LSTM models.
    """
    
    def __init__(self, model_dir=None, components_dir=None):
        """Initialize adaptive retrainer."""
        self.model_dir = model_dir or MODEL_DIR
        self.components_dir = components_dir or COMPONENTS_DIR
        
        self.adaptive_manager = AdaptiveLearningManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Load current model metrics for comparison
        self.baseline_metrics = self._load_baseline_metrics()
    
    def _load_baseline_metrics(self) -> Dict[str, Dict[str, float]]:
        """Load baseline metrics from current models."""
        metrics = {"rf": {}, "lstm": {}}
        
        # Load RF metrics
        feature_list_path = os.path.join(self.components_dir, FEATURE_LIST_NAME)
        if os.path.exists(feature_list_path):
            with open(feature_list_path, 'r') as f:
                rf_metadata = json.load(f)
                if 'metrics' in rf_metadata:
                    metrics["rf"] = rf_metadata['metrics']
        
        # Load LSTM metrics
        lstm_metadata_path = os.path.join(self.components_dir, LSTM_METADATA_NAME)
        if os.path.exists(lstm_metadata_path):
            with open(lstm_metadata_path, 'r') as f:
                lstm_metadata = json.load(f)
                if 'metrics' in lstm_metadata:
                    metrics["lstm"] = lstm_metadata['metrics']
        
        return metrics
    
    def retrain_random_forest(
        self,
        use_original_data: bool = True,
        max_new_samples: int = 5000
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Retrain Random Forest model with new samples.
        
        Args:
            use_original_data: Whether to include original training data
            max_new_samples: Maximum number of new samples to use
            
        Returns:
            Tuple of (success: bool, results: dict)
        """
        print("="*70)
        print("ADAPTIVE RETRAINING: RANDOM FOREST")
        print("="*70)
        
        try:
            # Load new samples
            new_samples = self.adaptive_manager.get_samples_for_training(
                limit=max_new_samples,
                require_ground_truth=True
            )
            
            if len(new_samples) < 10:
                return False, {"error": "Insufficient new samples with ground truth (need at least 10)"}
            
            print(f"\n📊 New samples available: {len(new_samples)}")
            
            # Prepare new samples data
            new_features_list = []
            new_labels_list = []
            file_hashes = []
            
            for sample in new_samples:
                features = sample.get("features", {})
                ground_truth = sample.get("ground_truth")
                
                if not features or not ground_truth:
                    continue
                
                # Convert features dict to array (matching feature order)
                feature_list_path = os.path.join(self.components_dir, FEATURE_LIST_NAME)
                with open(feature_list_path, 'r') as f:
                    feature_metadata = json.load(f)
                    feature_cols = feature_metadata['features']
                
                feature_array = [features.get(col, 0) for col in feature_cols]
                new_features_list.append(feature_array)
                new_labels_list.append(1 if ground_truth == "malware" else 0)
                file_hashes.append(sample.get("file_hash"))
            
            new_X = np.array(new_features_list)
            new_y = np.array(new_labels_list)
            
            print(f"✅ Prepared {len(new_X)} new samples")
            
            # Load original training data if requested
            if use_original_data:
                print(f"\n📂 Loading original training data...")
                original_df = load_rf_data(
                    r"C:\Users\Admin\github-classroom\Caty175\poly_trial\dataset\ember2018\processed\ember_train_full.parquet"
                )
                original_X, original_y, feature_cols = prepare_rf_features(original_df)
                
                # Combine original and new data
                print(f"🔄 Combining original ({len(original_X)}) and new ({len(new_X)}) samples...")
                combined_X = np.vstack([original_X.values, new_X])
                combined_y = np.hstack([original_y.values, new_y])
                
                print(f"✅ Combined dataset: {len(combined_X)} samples")
            else:
                # Use only new samples
                combined_X = new_X
                combined_y = new_y
                feature_cols = [f"feature_{i}" for i in range(new_X.shape[1])]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                combined_X, combined_y, test_size=0.2, random_state=RANDOM_STATE, stratify=combined_y
            )
            
            print(f"\n✂️ Split data:")
            print(f"   Train: {len(X_train)} samples")
            print(f"   Test:  {len(X_test)} samples")
            
            # Train new model
            print(f"\n🌲 Training Random Forest model...")
            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(X_train, y_train)
            
            # Evaluate
            print(f"\n📊 Evaluating model...")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"\n{'='*50}")
            print(f"EVALUATION RESULTS")
            print(f"{'='*50}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC-AUC Score: {roc_auc:.4f}")
            
            # Compare with baseline
            baseline_acc = self.baseline_metrics.get("rf", {}).get("accuracy", 0)
            baseline_auc = self.baseline_metrics.get("rf", {}).get("roc_auc", 0)
            
            improvement = accuracy - baseline_acc
            auc_improvement = roc_auc - baseline_auc
            
            print(f"\n📈 Comparison with baseline:")
            print(f"   Baseline Accuracy: {baseline_acc:.4f}")
            print(f"   New Accuracy:      {accuracy:.4f}")
            print(f"   Improvement:       {improvement:+.4f} ({improvement*100:+.2f}%)")
            print(f"   Baseline ROC-AUC:  {baseline_auc:.4f}")
            print(f"   New ROC-AUC:       {roc_auc:.4f}")
            print(f"   Improvement:       {auc_improvement:+.4f} ({auc_improvement*100:+.2f}%)")
            
            # Only save if performance is acceptable
            if accuracy >= baseline_acc - 0.02:  # Allow 2% tolerance
                print(f"\n💾 Saving new model...")
                
                # Backup old model
                old_model_path = os.path.join(self.components_dir, RF_MODEL_NAME)
                backup_path = os.path.join(self.components_dir, f"{RF_MODEL_NAME}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                if os.path.exists(old_model_path):
                    import shutil
                    shutil.copy2(old_model_path, backup_path)
                    print(f"   ✅ Backed up old model to {backup_path}")
                
                # Save new model
                model_path = os.path.join(self.components_dir, RF_MODEL_NAME)
                joblib.dump(model, model_path)
                print(f"   ✅ Saved new model to {model_path}")
                
                # Update feature metadata
                feature_list_path = os.path.join(self.components_dir, FEATURE_LIST_NAME)
                with open(feature_list_path, 'r') as f:
                    feature_metadata = json.load(f)
                
                feature_metadata['metrics'] = {
                    'accuracy': float(accuracy),
                    'roc_auc': float(roc_auc),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
                feature_metadata['last_retraining'] = datetime.now().isoformat()
                feature_metadata['retraining_samples'] = len(new_X)
                
                with open(feature_list_path, 'w') as f:
                    json.dump(feature_metadata, f, indent=2)
                
                # Mark samples as used
                self.adaptive_manager.mark_samples_as_used(file_hashes)
                
                # Record retraining
                self.adaptive_manager.record_retraining(
                    model_type="rf",
                    samples_used=len(new_X),
                    feedback_used=0,
                    old_metrics=self.baseline_metrics.get("rf", {}),
                    new_metrics={"accuracy": accuracy, "roc_auc": roc_auc},
                    success=True
                )
                
                return True, {
                    "accuracy": float(accuracy),
                    "roc_auc": float(roc_auc),
                    "improvement": float(improvement),
                    "samples_used": len(new_X),
                    "baseline_accuracy": float(baseline_acc)
                }
            else:
                print(f"\n⚠️ New model performance ({accuracy:.4f}) is worse than baseline ({baseline_acc:.4f})")
                print(f"   Not saving new model. Keeping baseline model.")
                
                self.adaptive_manager.record_retraining(
                    model_type="rf",
                    samples_used=len(new_X),
                    feedback_used=0,
                    old_metrics=self.baseline_metrics.get("rf", {}),
                    new_metrics={"accuracy": accuracy, "roc_auc": roc_auc},
                    success=False,
                    error_message=f"Performance degraded: {accuracy:.4f} < {baseline_acc:.4f}"
                )
                
                return False, {
                    "error": "Performance degraded",
                    "accuracy": float(accuracy),
                    "baseline_accuracy": float(baseline_acc)
                }
                
        except Exception as e:
            import traceback
            error_msg = f"Error during RF retraining: {str(e)}\n{traceback.format_exc()}"
            print(f"\n❌ {error_msg}")
            
            self.adaptive_manager.record_retraining(
                model_type="rf",
                samples_used=0,
                feedback_used=0,
                old_metrics={},
                new_metrics={},
                success=False,
                error_message=error_msg
            )
            
            return False, {"error": error_msg}
    
    def retrain_lstm(
        self,
        use_original_data: bool = True,
        max_new_samples: int = 1000
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Retrain LSTM model with new behavioral samples.
        
        Args:
            use_original_data: Whether to include original training data
            max_new_samples: Maximum number of new samples to use
            
        Returns:
            Tuple of (success: bool, results: dict)
        """
        print("="*70)
        print("ADAPTIVE RETRAINING: LSTM BEHAVIORAL")
        print("="*70)
        
        try:
            # Load new samples with behavioral features
            new_samples = self.adaptive_manager.get_samples_for_training(
                limit=max_new_samples,
                require_ground_truth=True
            )
            
            # Filter to samples with behavioral features
            new_samples = [s for s in new_samples if s.get("behavioral_features")]
            
            if len(new_samples) < 10:
                return False, {"error": "Insufficient new samples with behavioral features (need at least 10)"}
            
            print(f"\n📊 New behavioral samples available: {len(new_samples)}")
            
            # Load LSTM metadata to get feature structure
            lstm_metadata_path = os.path.join(self.components_dir, LSTM_METADATA_NAME)
            with open(lstm_metadata_path, 'r') as f:
                lstm_metadata = json.load(f)
            
            api_features = lstm_metadata['feature_metadata']['api_features']
            static_features = lstm_metadata['feature_metadata']['static_features']
            
            # Prepare new samples
            new_api_data = []
            new_static_data = []
            new_labels = []
            file_hashes = []
            
            for sample in new_samples:
                behavioral = sample.get("behavioral_features", {})
                ground_truth = sample.get("ground_truth")
                
                if not behavioral or not ground_truth:
                    continue
                
                # Extract API features
                api_row = [behavioral.get(feat, 0) for feat in api_features]
                new_api_data.append(api_row)
                
                # Extract static features
                static_row = [behavioral.get(feat, 0) for feat in static_features]
                new_static_data.append(static_row)
                
                new_labels.append(1 if ground_truth == "malware" else 0)
                file_hashes.append(sample.get("file_hash"))
            
            new_X_api = np.array(new_api_data)
            new_X_static = np.array(new_static_data)
            new_y = np.array(new_labels)
            
            print(f"✅ Prepared {len(new_X_api)} new behavioral samples")
            
            # Load original data if requested
            if use_original_data:
                print(f"\n📂 Loading original training data...")
                original_X_api, original_X_static, original_y, _ = load_lstm_data(
                    r"C:\Users\Admin\github-classroom\Caty175\poly_trial\dataset\Malware_Analysis_kaggle.csv"
                )
                
                # Combine
                print(f"🔄 Combining original ({len(original_X_api)}) and new ({len(new_X_api)}) samples...")
                combined_X_api = np.vstack([original_X_api, new_X_api])
                combined_X_static = np.vstack([original_X_static, new_X_static])
                combined_y = np.hstack([original_y, new_y])
            else:
                combined_X_api = new_X_api
                combined_X_static = new_X_static
                combined_y = new_y
            
            # Split data
            X_api_train, X_api_test, X_static_train, X_static_test, y_train, y_test = train_test_split(
                combined_X_api, combined_X_static, combined_y,
                test_size=0.2, random_state=RANDOM_STATE, stratify=combined_y
            )
            
            X_api_train, X_api_val, X_static_train, X_static_val, y_train, y_val = train_test_split(
                X_api_train, X_static_train, y_train,
                test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
            )
            
            # Normalize
            api_scaler = RobustScaler()
            static_scaler = RobustScaler()
            
            X_api_train = api_scaler.fit_transform(X_api_train)
            X_api_val = api_scaler.transform(X_api_val)
            X_api_test = api_scaler.transform(X_api_test)
            
            X_static_train = static_scaler.fit_transform(X_static_train)
            X_static_val = static_scaler.transform(X_static_val)
            X_static_test = static_scaler.transform(X_static_test)
            
            # Reshape for LSTM
            X_api_train = X_api_train.reshape(X_api_train.shape[0], X_api_train.shape[1], 1)
            X_api_val = X_api_val.reshape(X_api_val.shape[0], X_api_val.shape[1], 1)
            X_api_test = X_api_test.reshape(X_api_test.shape[0], X_api_test.shape[1], 1)
            
            # Build and train model
            print(f"\n🏗️ Building LSTM model...")
            model = create_lstm_feedforward_model(
                api_input_dim=X_api_train.shape[1],
                static_input_dim=X_static_train.shape[1],
                lstm_units=LSTM_UNITS,
                dense_units=DENSE_UNITS,
                lstm_dropout=LSTM_DROPOUT,
                dense_dropout=DENSE_DROPOUT
            )
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
            
            # Train with early stopping
            checkpoint_path = os.path.join(self.components_dir, "best_model_checkpoint.h5")
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=PATIENCE,
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_auc',
                    mode='max',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            print(f"\n🚀 Training LSTM model...")
            history = model.fit(
                [X_api_train, X_static_train],
                y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=([X_api_val, X_static_val], y_val),
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Evaluate
            print(f"\n📊 Evaluating model...")
            y_pred_proba = model.predict([X_api_test, X_static_test], verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"\n{'='*50}")
            print(f"EVALUATION RESULTS")
            print(f"{'='*50}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC-AUC Score: {roc_auc:.4f}")
            
            # Compare with baseline
            baseline_acc = self.baseline_metrics.get("lstm", {}).get("accuracy", 0)
            baseline_auc = self.baseline_metrics.get("lstm", {}).get("roc_auc", 0)
            
            improvement = accuracy - baseline_acc
            
            # Save if performance acceptable
            if accuracy >= baseline_acc - 0.02:
                print(f"\n💾 Saving new LSTM model...")
                
                # Backup old model
                old_model_path = os.path.join(self.components_dir, LSTM_MODEL_NAME)
                backup_path = os.path.join(self.components_dir, f"{LSTM_MODEL_NAME}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                if os.path.exists(old_model_path):
                    import shutil
                    shutil.copy2(old_model_path, backup_path)
                
                # Save new model
                model_path = os.path.join(self.components_dir, LSTM_MODEL_NAME)
                model.save(model_path)
                
                # Save scalers
                scaler_path = os.path.join(self.components_dir, LSTM_SCALER_NAME)
                joblib.dump({'api_scaler': api_scaler, 'static_scaler': static_scaler}, scaler_path)
                
                # Update metadata
                lstm_metadata['metrics'] = {
                    'accuracy': float(accuracy),
                    'roc_auc': float(roc_auc)
                }
                lstm_metadata['last_retraining'] = datetime.now().isoformat()
                lstm_metadata['retraining_samples'] = len(new_X_api)
                
                with open(lstm_metadata_path, 'w') as f:
                    json.dump(lstm_metadata, f, indent=2)
                
                # Mark samples as used
                self.adaptive_manager.mark_samples_as_used(file_hashes)
                
                self.adaptive_manager.record_retraining(
                    model_type="lstm",
                    samples_used=len(new_X_api),
                    feedback_used=0,
                    old_metrics=self.baseline_metrics.get("lstm", {}),
                    new_metrics={"accuracy": accuracy, "roc_auc": roc_auc},
                    success=True
                )
                
                return True, {
                    "accuracy": float(accuracy),
                    "roc_auc": float(roc_auc),
                    "improvement": float(improvement),
                    "samples_used": len(new_X_api)
                }
            else:
                print(f"\n⚠️ Performance degraded, not saving")
                return False, {"error": "Performance degraded", "accuracy": float(accuracy)}
                
        except Exception as e:
            import traceback
            error_msg = f"Error during LSTM retraining: {str(e)}\n{traceback.format_exc()}"
            print(f"\n❌ {error_msg}")
            return False, {"error": error_msg}
    
    def retrain_both(self) -> Dict[str, Any]:
        """Retrain both models."""
        print("="*70)
        print("ADAPTIVE RETRAINING: BOTH MODELS")
        print("="*70)
        
        results = {
            "rf": {},
            "lstm": {},
            "overall_success": False
        }
        
        # Retrain RF
        rf_success, rf_results = self.retrain_random_forest()
        results["rf"] = rf_results
        
        # Retrain LSTM
        lstm_success, lstm_results = self.retrain_lstm()
        results["lstm"] = lstm_results
        
        results["overall_success"] = rf_success and lstm_success
        
        return results


def main():
    """Main entry point for adaptive retraining."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive retraining for malware detection models")
    parser.add_argument("--model", choices=["rf", "lstm", "both"], default="both",
                       help="Which model(s) to retrain")
    parser.add_argument("--no-original-data", action="store_true",
                       help="Don't use original training data, only new samples")
    parser.add_argument("--max-samples", type=int, default=5000,
                       help="Maximum number of new samples to use")
    
    args = parser.parse_args()
    
    retrainer = AdaptiveRetrainer()
    
    if args.model == "rf":
        success, results = retrainer.retrain_random_forest(
            use_original_data=not args.no_original_data,
            max_new_samples=args.max_samples
        )
    elif args.model == "lstm":
        success, results = retrainer.retrain_lstm(
            use_original_data=not args.no_original_data,
            max_new_samples=args.max_samples
        )
    else:
        results = retrainer.retrain_both()
        success = results.get("overall_success", False)
    
    if success:
        print("\n✅ Retraining completed successfully!")
    else:
        print("\n❌ Retraining failed or performance degraded")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

