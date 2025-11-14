#!/usr/bin/env python3
"""
Test client for the Malware Detection API
Demonstrates how to use the API to scan files.
"""

import requests
import json
import sys
import os
from pathlib import Path


API_URL = "http://localhost:8000"


def check_api_health():
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API is healthy!")
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Features: {data['num_features']}")
            if data.get('model_info'):
                print(f"   Training date: {data['model_info'].get('training_date', 'unknown')}")
                print(f"   Accuracy: {data['model_info'].get('accuracy', 'unknown')}")
                print(f"   ROC-AUC: {data['model_info'].get('roc_auc', 'unknown')}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return False
    except Exception as e:
        print(f"❌ Error checking API health: {e}")
        return False


def scan_file(file_path: str, detailed: bool = False):
    """
    Scan a file using the API.
    
    Args:
        file_path: Path to the file to scan
        detailed: If True, get detailed results with features
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    endpoint = "/scan/detailed" if detailed else "/scan"
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
            response = requests.post(f"{API_URL}{endpoint}", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*60)
            print("SCAN RESULT")
            print("="*60)
            print(f"File: {result['filename']}")
            print(f"Size: {result['file_size']:,} bytes")
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Malware probability: {result['malware_probability']:.2%}")
            print(f"Benign probability: {result['benign_probability']:.2%}")
            print(f"Features extracted: {result['features_extracted']}")
            print(f"Timestamp: {result['timestamp']}")
            
            if detailed and 'top_features' in result:
                print("\n" + "-"*60)
                print("TOP CONTRIBUTING FEATURES")
                print("-"*60)
                for feat in result['top_features']:
                    print(f"  {feat['feature']:50s}: {feat['value']:10.2f} (importance: {feat['importance']:.4f})")
            
            print("="*60)
            return result
        else:
            print(f"❌ Error scanning file: {response.status_code}")
            print(f"   {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error scanning file: {e}")
        return None


def get_model_info():
    """Get information about the loaded model."""
    try:
        response = requests.get(f"{API_URL}/model/info")
        if response.status_code == 200:
            info = response.json()
            print("\n" + "="*60)
            print("MODEL INFORMATION")
            print("="*60)
            print(f"Number of features: {info['num_features']}")
            print(f"Training date: {info['training_date']}")
            print(f"\nModel parameters:")
            for key, value in info['model_params'].items():
                print(f"  {key}: {value}")
            print(f"\nMetrics:")
            for key, value in info['metrics'].items():
                if key != 'confusion_matrix':
                    print(f"  {key}: {value}")
            print(f"\nTop 10 most important features:")
            for i, feat in enumerate(info['top_features'][:10], 1):
                print(f"  {i:2d}. {feat['feature']:50s} (importance: {feat['importance']:.4f})")
            print("="*60)
            return info
        else:
            print(f"❌ Error getting model info: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
        return None


def main():
    """Main function."""
    print("="*60)
    print("MALWARE DETECTION API - TEST CLIENT")
    print("="*60)
    
    # Check API health
    if not check_api_health():
        print("\n⚠️ Please start the API server first:")
        print("   python api_server.py")
        return
    
    # Get model info
    print("\n")
    get_model_info()
    
    # Scan a file if provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        detailed = "--detailed" in sys.argv
        print(f"\n📂 Scanning file: {file_path}")
        scan_file(file_path, detailed=detailed)
    else:
        print("\n" + "="*60)
        print("USAGE")
        print("="*60)
        print("To scan a file:")
        print(f"  python {os.path.basename(__file__)} <file_path>")
        print(f"\nFor detailed results:")
        print(f"  python {os.path.basename(__file__)} <file_path> --detailed")
        print("="*60)


if __name__ == "__main__":
    main()

