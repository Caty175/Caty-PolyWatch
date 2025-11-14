#!/usr/bin/env python3
"""
Test client for the Malware Detection API
Demonstrates how to interact with the API server.
"""

import requests
import sys
import os
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint."""
    print("=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Server Status: {data['status']}")
        print(f"✅ Model Loaded: {data['model_loaded']}")
        print(f"✅ Feature Extractor Loaded: {data['feature_extractor_loaded']}")
        print(f"✅ Number of Features: {data['num_features']}")
        
        if data.get('model_info'):
            print(f"\nModel Information:")
            print(f"  Training Date: {data['model_info'].get('training_date', 'unknown')}")
            print(f"  Accuracy: {data['model_info'].get('accuracy', 'unknown')}")
            print(f"  ROC-AUC: {data['model_info'].get('roc_auc', 'unknown')}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API server")
        print("   Make sure the server is running: python api_server.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_model_info():
    """Test the model info endpoint."""
    print("\n" + "=" * 60)
    print("Testing Model Info Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Number of Features: {data['num_features']}")
        print(f"✅ Training Date: {data['training_date']}")
        
        if data.get('metrics'):
            print(f"\nModel Metrics:")
            for key, value in data['metrics'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        if data.get('top_features'):
            print(f"\nTop 5 Most Important Features:")
            for i, feat in enumerate(data['top_features'][:5], 1):
                print(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def scan_file(file_path: str, detailed: bool = False):
    """
    Scan a file for malware.
    
    Args:
        file_path: Path to the PE file to scan
        detailed: If True, use the detailed scan endpoint
    """
    print("\n" + "=" * 60)
    print(f"Scanning File: {file_path}")
    print(f"Detailed Mode: {detailed}")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return False
    
    # Get file size
    file_size = os.path.getsize(file_path)
    print(f"File Size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    
    try:
        # Prepare the file for upload
        with open(file_path, 'rb') as f:
            files = {
                'file': (os.path.basename(file_path), f, 'application/octet-stream')
            }
            
            # Choose endpoint
            endpoint = "/scan/detailed" if detailed else "/scan"
            
            # Send request
            print(f"\n🔍 Sending request to {API_BASE_URL}{endpoint}...")
            response = requests.post(f"{API_BASE_URL}{endpoint}", files=files)
            response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Display results
        print("\n" + "=" * 60)
        print("SCAN RESULTS")
        print("=" * 60)
        print(f"Filename: {result['filename']}")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Malware Probability: {result['malware_probability']:.2%}")
        print(f"Benign Probability: {result['benign_probability']:.2%}")
        print(f"Features Extracted: {result['features_extracted']}")
        print(f"Timestamp: {result['timestamp']}")
        
        # Display detailed information if available
        if detailed and 'top_features' in result:
            print(f"\nTop Contributing Features:")
            for i, feat in enumerate(result['top_features'][:5], 1):
                print(f"  {i}. {feat['feature']}: {feat['value']} (importance: {feat['importance']:.4f})")
        
        # Color-coded result
        if result['prediction'] == 'malware':
            print("\n⚠️  WARNING: This file is predicted to be MALWARE!")
        else:
            print("\n✅ This file appears to be BENIGN")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Detail: {error_detail.get('detail', 'Unknown error')}")
            except:
                print(f"   Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main function to run tests."""
    print("=" * 60)
    print("MALWARE DETECTION API - TEST CLIENT")
    print("=" * 60)
    
    # Test health check
    if not test_health_check():
        print("\n❌ Server is not running or not responding")
        print("   Start the server with: python api_server.py")
        sys.exit(1)
    
    # Test model info
    test_model_info()
    
    # Check if file path is provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        detailed = "--detailed" in sys.argv or "-d" in sys.argv
        
        # Scan the file
        scan_file(file_path, detailed=detailed)
    else:
        print("\n" + "=" * 60)
        print("Usage Examples")
        print("=" * 60)
        print("1. Test server health:")
        print("   python test_client.py")
        print("\n2. Scan a file (basic):")
        print("   python test_client.py path/to/file.exe")
        print("\n3. Scan a file (detailed):")
        print("   python test_client.py path/to/file.exe --detailed")
        print("\n4. Scan a file (detailed, short flag):")
        print("   python test_client.py path/to/file.exe -d")
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

