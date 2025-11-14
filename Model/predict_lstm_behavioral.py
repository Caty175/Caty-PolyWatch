#!/usr/bin/env python3
"""
Prediction script for LSTM + Feedforward Behavioral Malware Detector
Loads trained model and makes predictions on new behavioral data
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras

# ========== CONFIG ==========
MODEL_DIR = r"C:\Users\Admin\github-classroom\Caty175\poly_trial\Model"
COMPONENTS_DIR = os.path.join(MODEL_DIR, "components")  # Model components location
MODEL_NAME = "lstm_behavioral_malware_detector.h5"
SCALER_NAME = "behavioral_scaler.pkl"
METADATA_NAME = "lstm_model_metadata.json"


class BehavioralMalwareDetector:
    """
    LSTM + Feedforward behavioral malware detector.
    Predicts malware based on API calls and behavioral features.
    """
    
    def __init__(self, model_dir=MODEL_DIR, components_dir=None):
        """Initialize the detector by loading model and artifacts."""
        self.model_dir = model_dir
        # Use components directory if specified, otherwise try components subfolder, then fall back to model_dir
        if components_dir:
            self.components_dir = components_dir
        elif os.path.exists(os.path.join(model_dir, "components")):
            self.components_dir = os.path.join(model_dir, "components")
        else:
            self.components_dir = model_dir  # Backward compatibility

        self.model = None
        self.scalers = None
        self.metadata = None
        self.api_features = None
        self.static_features = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, scalers, and metadata."""
        print(f"🔧 Loading LSTM behavioral malware detector...")

        # Load model from components directory
        model_path = os.path.join(self.components_dir, MODEL_NAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = keras.models.load_model(model_path)
        print(f"   ✅ Model loaded from {model_path}")

        # Load scalers from components directory
        scaler_path = os.path.join(self.components_dir, SCALER_NAME)
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scalers not found at {scaler_path}")

        self.scalers = joblib.load(scaler_path)
        print(f"   ✅ Scalers loaded from {scaler_path}")

        # Load metadata from components directory
        metadata_path = os.path.join(self.components_dir, METADATA_NAME)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.api_features = self.metadata['feature_metadata']['api_features']
        self.static_features = self.metadata['feature_metadata']['static_features']

        print(f"   ✅ Metadata loaded from {metadata_path}")
        print(f"\n📊 Model info:")
        print(f"   - API features: {len(self.api_features)}")
        print(f"   - Static features: {len(self.static_features)}")
        print(f"   - Training accuracy: {self.metadata['metrics']['accuracy']:.4f}")
        print(f"   - Training ROC-AUC: {self.metadata['metrics']['roc_auc']:.4f}")
        print(f"\n✅ Detector ready!")
    
    def predict(self, behavioral_data):
        """
        Predict malware from behavioral data.
        
        Args:
            behavioral_data: dict or DataFrame with behavioral features
        
        Returns:
            dict with prediction results
        """
        # Convert to DataFrame if dict
        if isinstance(behavioral_data, dict):
            behavioral_data = pd.DataFrame([behavioral_data])
        
        # Extract API features
        api_data = behavioral_data[self.api_features].fillna(0).values
        
        # Extract static features
        static_data = behavioral_data[self.static_features].fillna(0).values
        
        # Normalize
        api_data = self.scalers['api_scaler'].transform(api_data)
        static_data = self.scalers['static_scaler'].transform(static_data)
        
        # Reshape API data for LSTM
        api_data = api_data.reshape(api_data.shape[0], api_data.shape[1], 1)
        
        # Predict
        malware_prob = self.model.predict([api_data, static_data], verbose=0)[0][0]
        benign_prob = 1 - malware_prob
        
        prediction = "malware" if malware_prob >= 0.5 else "benign"
        confidence = max(malware_prob, benign_prob)
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'malware_probability': float(malware_prob),
            'benign_probability': float(benign_prob)
        }
    
    def predict_batch(self, behavioral_data_df):
        """
        Predict malware for multiple samples.
        
        Args:
            behavioral_data_df: DataFrame with behavioral features
        
        Returns:
            list of prediction dicts
        """
        # Extract features
        api_data = behavioral_data_df[self.api_features].fillna(0).values
        static_data = behavioral_data_df[self.static_features].fillna(0).values
        
        # Normalize
        api_data = self.scalers['api_scaler'].transform(api_data)
        static_data = self.scalers['static_scaler'].transform(static_data)
        
        # Reshape API data for LSTM
        api_data = api_data.reshape(api_data.shape[0], api_data.shape[1], 1)
        
        # Predict
        malware_probs = self.model.predict([api_data, static_data], verbose=0).flatten()
        
        results = []
        for malware_prob in malware_probs:
            benign_prob = 1 - malware_prob
            prediction = "malware" if malware_prob >= 0.5 else "benign"
            confidence = max(malware_prob, benign_prob)
            
            results.append({
                'prediction': prediction,
                'confidence': float(confidence),
                'malware_probability': float(malware_prob),
                'benign_probability': float(benign_prob)
            })
        
        return results


def main():
    """Demo: Load model and make predictions on test data."""
    print("=" * 80)
    print("LSTM BEHAVIORAL MALWARE DETECTOR - PREDICTION DEMO")
    print("=" * 80)
    
    # Initialize detector
    detector = BehavioralMalwareDetector()
    
    # Load test data
    print(f"\n📂 Loading test data...")
    dataset_path = r"C:\Users\Admin\github-classroom\Caty175\poly_trial\dataset\Malware_Analysis_kaggle.csv"
    df = pd.read_csv(dataset_path)
    
    # Take a few samples
    test_samples = df.sample(n=10, random_state=42)
    
    print(f"\n🔍 Making predictions on {len(test_samples)} samples...")
    
    # Predict
    results = detector.predict_batch(test_samples)
    
    # Display results
    print(f"\n{'='*80}")
    print("PREDICTION RESULTS")
    print(f"{'='*80}")
    
    for i, (idx, row) in enumerate(test_samples.iterrows()):
        result = results[i]
        actual_score = row['Score'] if 'Score' in row else 'Unknown'
        actual_label = 'Malware' if actual_score >= 5.0 else 'Benign'
        
        print(f"\nSample {i+1} (ID: {idx}):")
        print(f"   Actual: {actual_label} (Score: {actual_score})")
        print(f"   Predicted: {result['prediction'].upper()}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Malware probability: {result['malware_probability']:.2%}")
        print(f"   Benign probability: {result['benign_probability']:.2%}")
        
        # Check if correct
        is_correct = (result['prediction'] == 'malware' and actual_score >= 5.0) or \
                     (result['prediction'] == 'benign' and actual_score < 5.0)
        print(f"   ✅ CORRECT" if is_correct else "   ❌ INCORRECT")
    
    # Calculate accuracy
    correct = sum(
        1 for i, (idx, row) in enumerate(test_samples.iterrows())
        if (results[i]['prediction'] == 'malware' and row['Score'] >= 5.0) or
           (results[i]['prediction'] == 'benign' and row['Score'] < 5.0)
    )
    accuracy = correct / len(test_samples)
    
    print(f"\n{'='*80}")
    print(f"Sample Accuracy: {accuracy:.2%} ({correct}/{len(test_samples)})")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

