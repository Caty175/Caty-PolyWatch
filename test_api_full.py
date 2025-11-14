#!/usr/bin/env python3
"""
Comprehensive API Test Suite
Tests all endpoints of the Malware Detection API using polymorphic_demo files.
"""

import requests
import json
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# API Configuration
API_BASE_URL = "http://localhost:8000"
TEST_FILE = "polymorphic_demo/poly_demo1.py"

# Test results tracking
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_test(name: str, status: str, details: str = ""):
    """Print test result."""
    status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_symbol} {name}")
    if details:
        print(f"   {details}")
    
    if status == "PASS":
        test_results["passed"].append(name)
    elif status == "FAIL":
        test_results["failed"].append(name)
    else:
        test_results["warnings"].append(name)


def test_server_connection() -> bool:
    """Test if the server is accessible."""
    print_header("Test 1: Server Connection")
    
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print_test("Server Connection", "PASS", f"Server is running at {API_BASE_URL}")
            return True
        else:
            print_test("Server Connection", "FAIL", f"Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_test("Server Connection", "FAIL", "Cannot connect to server. Is it running?")
        print("   Start the server with: python server/api_server.py")
        return False
    except Exception as e:
        print_test("Server Connection", "FAIL", f"Error: {str(e)}")
        return False


def test_health_endpoint() -> bool:
    """Test the /health endpoint."""
    print_header("Test 2: Health Check Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Validate response structure
        required_fields = ["status", "model_loaded", "feature_extractor_loaded", "num_features"]
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            print_test("Health Endpoint Structure", "FAIL", f"Missing fields: {missing_fields}")
            return False
        
        print_test("Health Endpoint Structure", "PASS", "All required fields present")
        print(f"   Status: {data['status']}")
        print(f"   Model Loaded: {data['model_loaded']}")
        print(f"   Feature Extractor Loaded: {data['feature_extractor_loaded']}")
        print(f"   Number of Features: {data['num_features']}")
        
        if data.get('model_info'):
            model_info = data['model_info']
            print(f"   Training Date: {model_info.get('training_date', 'unknown')}")
            print(f"   Accuracy: {model_info.get('accuracy', 'unknown')}")
            print(f"   ROC-AUC: {model_info.get('roc_auc', 'unknown')}")
        
        # Check if model is actually loaded
        if not data['model_loaded']:
            print_test("Model Loaded Check", "WARN", "Model is not loaded")
        else:
            print_test("Model Loaded Check", "PASS", "Model is loaded")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print_test("Health Endpoint", "FAIL", f"HTTP Error: {e}")
        return False
    except Exception as e:
        print_test("Health Endpoint", "FAIL", f"Error: {str(e)}")
        return False


def test_root_endpoint() -> bool:
    """Test the root / endpoint."""
    print_header("Test 3: Root Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Root endpoint should return same as /health
        if "status" in data and data["status"] == "online":
            print_test("Root Endpoint", "PASS", "Root endpoint returns health status")
            return True
        else:
            print_test("Root Endpoint", "FAIL", "Root endpoint does not return expected status")
            return False
            
    except Exception as e:
        print_test("Root Endpoint", "FAIL", f"Error: {str(e)}")
        return False


def test_model_info_endpoint() -> bool:
    """Test the /model/info endpoint."""
    print_header("Test 4: Model Info Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Validate required fields
        required_fields = ["num_features", "training_date"]
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            print_test("Model Info Structure", "FAIL", f"Missing fields: {missing_fields}")
            return False
        
        print_test("Model Info Structure", "PASS", "All required fields present")
        print(f"   Number of Features: {data['num_features']}")
        print(f"   Training Date: {data['training_date']}")
        
        if data.get('model_params'):
            print(f"   Model Parameters: {len(data['model_params'])} parameters")
        
        if data.get('metrics'):
            print(f"   Metrics: {len(data['metrics'])} metrics available")
            for key, value in data['metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"      {key}: {value:.4f}")
        
        if data.get('top_features'):
            print(f"   Top Features: {len(data['top_features'])} features available")
            print("   Top 5 Most Important Features:")
            for i, feat in enumerate(data['top_features'][:5], 1):
                print(f"      {i}. {feat['feature']}: {feat.get('importance', 'N/A'):.4f}")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print_test("Model Info Endpoint", "FAIL", f"HTTP Error: {e}")
        if e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Detail: {error_detail.get('detail', 'Unknown error')}")
            except:
                pass
        return False
    except Exception as e:
        print_test("Model Info Endpoint", "FAIL", f"Error: {str(e)}")
        return False


def test_scan_endpoint_basic(file_path: str) -> bool:
    """Test the /scan endpoint (basic scan)."""
    print_header("Test 5: Basic Scan Endpoint")
    
    if not os.path.exists(file_path):
        print_test("Test File Exists", "FAIL", f"Test file not found: {file_path}")
        return False
    
    print_test("Test File Exists", "PASS", f"Using test file: {file_path}")
    file_size = os.path.getsize(file_path)
    print(f"   File Size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
            print(f"\n   Sending request to {API_BASE_URL}/scan...")
            response = requests.post(f"{API_BASE_URL}/scan", files=files, timeout=30)
        
        response.raise_for_status()
        data = response.json()
        
        # Validate response structure
        required_fields = [
            "filename", "prediction", "confidence", "malware_probability",
            "benign_probability", "timestamp", "features_extracted", "file_size"
        ]
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            print_test("Scan Response Structure", "FAIL", f"Missing fields: {missing_fields}")
            return False
        
        print_test("Scan Response Structure", "PASS", "All required fields present")
        print(f"\n   Scan Results:")
        print(f"      Filename: {data['filename']}")
        print(f"      Prediction: {data['prediction'].upper()}")
        print(f"      Confidence: {data['confidence']:.2%}")
        print(f"      Malware Probability: {data['malware_probability']:.2%}")
        print(f"      Benign Probability: {data['benign_probability']:.2%}")
        print(f"      Features Extracted: {data['features_extracted']}")
        print(f"      File Size: {data['file_size']:,} bytes")
        print(f"      Timestamp: {data['timestamp']}")
        
        # Check for weighted results
        if 'random_forest_probability' in data:
            rf_prob = data['random_forest_probability']
            lstm_prob = data.get('lstm_probability')
            sandbox_performed = data.get('sandbox_analysis_performed', False)
            
            print(f"\n   Model Breakdown:")
            print(f"      Random Forest: {rf_prob:.2%}")
            if lstm_prob is not None:
                print(f"      LSTM Behavioral: {lstm_prob:.2%}")
                print(f"      Sandbox Analysis: {'Yes' if sandbox_performed else 'No'}")
            else:
                print(f"      LSTM Behavioral: Not triggered")
                print(f"      Sandbox Analysis: No")
            
            # Verify weighted calculation when LSTM not triggered
            if lstm_prob is None and not sandbox_performed:
                expected_weighted = rf_prob * 0.30  # RF_WEIGHT = 0.30
                actual_weighted = data['malware_probability']
                if abs(actual_weighted - expected_weighted) < 0.01:  # Allow small floating point differences
                    print_test("Weighted Probability Calculation", "PASS", 
                             f"Correctly weighted: {actual_weighted:.2%} = {rf_prob:.2%} × 30%")
                else:
                    print_test("Weighted Probability Calculation", "WARN",
                             f"Expected {expected_weighted:.2%}, got {actual_weighted:.2%}")
        
        # Validate data types
        assert isinstance(data['prediction'], str), "prediction should be string"
        assert data['prediction'] in ['malware', 'benign'], "prediction should be 'malware' or 'benign'"
        assert 0 <= data['confidence'] <= 1, "confidence should be between 0 and 1"
        assert 0 <= data['malware_probability'] <= 1, "malware_probability should be between 0 and 1"
        assert 0 <= data['benign_probability'] <= 1, "benign_probability should be between 0 and 1"
        
        print_test("Data Type Validation", "PASS", "All data types are correct")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print_test("Basic Scan Endpoint", "FAIL", f"HTTP Error: {e}")
        if e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Detail: {error_detail.get('detail', 'Unknown error')}")
            except:
                print(f"   Response: {e.response.text[:200]}")
        return False
    except AssertionError as e:
        print_test("Data Validation", "FAIL", f"Validation error: {str(e)}")
        return False
    except Exception as e:
        print_test("Basic Scan Endpoint", "FAIL", f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_scan_endpoint_detailed(file_path: str) -> bool:
    """Test the /scan/detailed endpoint."""
    print_header("Test 6: Detailed Scan Endpoint")
    
    if not os.path.exists(file_path):
        print_test("Test File Exists", "FAIL", f"Test file not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
            print(f"   Sending request to {API_BASE_URL}/scan/detailed...")
            response = requests.post(f"{API_BASE_URL}/scan/detailed", files=files, timeout=30)
        
        response.raise_for_status()
        data = response.json()
        
        # Detailed scan should have all basic fields plus top_features and all_features
        required_fields = [
            "filename", "prediction", "confidence", "malware_probability",
            "benign_probability", "timestamp", "features_extracted", "file_size",
            "top_features", "all_features"
        ]
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            print_test("Detailed Scan Structure", "FAIL", f"Missing fields: {missing_fields}")
            return False
        
        print_test("Detailed Scan Structure", "PASS", "All required fields present")
        
        # Validate top_features
        if isinstance(data['top_features'], list) and len(data['top_features']) > 0:
            print_test("Top Features", "PASS", f"{len(data['top_features'])} top features returned")
            print("   Top 5 Features:")
            for i, feat in enumerate(data['top_features'][:5], 1):
                feat_name = feat.get('feature', 'unknown')
                feat_value = feat.get('value', 'N/A')
                feat_importance = feat.get('importance', 'N/A')
                print(f"      {i}. {feat_name}: {feat_value} (importance: {feat_importance})")
        else:
            print_test("Top Features", "WARN", "No top features returned or empty list")
        
        # Validate all_features
        if isinstance(data['all_features'], dict) and len(data['all_features']) > 0:
            print_test("All Features", "PASS", f"{len(data['all_features'])} features in all_features")
        else:
            print_test("All Features", "WARN", "No features in all_features or not a dict")
        
        return True
        
    except requests.exceptions.HTTPError as e:
        print_test("Detailed Scan Endpoint", "FAIL", f"HTTP Error: {e}")
        if e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Detail: {error_detail.get('detail', 'Unknown error')}")
            except:
                print(f"   Response: {e.response.text[:200]}")
        return False
    except Exception as e:
        print_test("Detailed Scan Endpoint", "FAIL", f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling() -> bool:
    """Test error handling for invalid requests."""
    print_header("Test 7: Error Handling")
    
    # Test 1: Missing file
    try:
        response = requests.post(f"{API_BASE_URL}/scan", timeout=10)
        if response.status_code in [400, 422]:
            print_test("Missing File Error", "PASS", "Correctly returns error for missing file")
        else:
            print_test("Missing File Error", "WARN", f"Unexpected status code: {response.status_code}")
    except Exception as e:
        print_test("Missing File Error", "WARN", f"Exception: {str(e)}")
    
    # Test 2: Empty file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as temp_file:
            temp_file.write(b'')
            temp_path = temp_file.name
        
        with open(temp_path, 'rb') as f:
            files = {'file': ('empty.exe', f, 'application/octet-stream')}
            response = requests.post(f"{API_BASE_URL}/scan", files=files, timeout=10)
        
        if response.status_code == 400:
            print_test("Empty File Error", "PASS", "Correctly returns error for empty file")
        else:
            print_test("Empty File Error", "WARN", f"Unexpected status code: {response.status_code}")
        
        os.unlink(temp_path)
    except Exception as e:
        print_test("Empty File Error", "WARN", f"Exception: {str(e)}")
    
    return True


def test_api_documentation() -> bool:
    """Test that API documentation is accessible."""
    print_header("Test 8: API Documentation")
    
    try:
        # Test Swagger UI
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print_test("Swagger UI", "PASS", "Swagger documentation accessible at /docs")
        else:
            print_test("Swagger UI", "WARN", f"Status code: {response.status_code}")
        
        # Test OpenAPI schema
        response = requests.get(f"{API_BASE_URL}/openapi.json", timeout=5)
        if response.status_code == 200:
            schema = response.json()
            if 'openapi' in schema or 'swagger' in schema:
                print_test("OpenAPI Schema", "PASS", "OpenAPI schema is valid")
                print(f"   API Version: {schema.get('info', {}).get('version', 'unknown')}")
            else:
                print_test("OpenAPI Schema", "WARN", "Schema format unexpected")
        else:
            print_test("OpenAPI Schema", "WARN", f"Status code: {response.status_code}")
        
        return True
        
    except Exception as e:
        print_test("API Documentation", "WARN", f"Error: {str(e)}")
        return True  # Don't fail the test suite for documentation issues


def print_summary():
    """Print test summary."""
    print_header("Test Summary")
    
    total = len(test_results["passed"]) + len(test_results["failed"]) + len(test_results["warnings"])
    passed = len(test_results["passed"])
    failed = len(test_results["failed"])
    warnings = len(test_results["warnings"])
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Warnings: {warnings}")
    
    if failed > 0:
        print("\nFailed Tests:")
        for test in test_results["failed"]:
            print(f"   ❌ {test}")
    
    if warnings > 0:
        print("\nWarnings:")
        for test in test_results["warnings"]:
            print(f"   ⚠️  {test}")
    
    print("\n" + "=" * 80)
    
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} TEST(S) FAILED")
    
    print("=" * 80 + "\n")


def main():
    """Run all tests."""
    print_header("Comprehensive API Test Suite")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Test File: {TEST_FILE}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if test file exists
    test_file = TEST_FILE
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    if not os.path.exists(test_file):
        print(f"\n⚠️  Warning: Test file not found: {test_file}")
        print("   Using default: polymorphic_demo/poly_demo1.py")
        test_file = TEST_FILE
        if not os.path.exists(test_file):
            print(f"\n❌ Error: Default test file also not found: {test_file}")
            print("   Please provide a valid file path as argument or ensure polymorphic_demo/poly_demo1.py exists")
            sys.exit(1)
    
    # Run tests
    all_passed = True
    
    # Test 1: Server connection (must pass for other tests)
    if not test_server_connection():
        print("\n❌ Cannot proceed without server connection. Please start the server first.")
        sys.exit(1)
    
    # Test 2-8: Other tests
    all_passed &= test_health_endpoint()
    all_passed &= test_root_endpoint()
    all_passed &= test_model_info_endpoint()
    all_passed &= test_scan_endpoint_basic(test_file)
    all_passed &= test_scan_endpoint_detailed(test_file)
    all_passed &= test_error_handling()
    all_passed &= test_api_documentation()
    
    # Print summary
    print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if len(test_results["failed"]) == 0 else 1)


if __name__ == "__main__":
    main()

