#!/usr/bin/env python3
"""
API Test Script for Adaptive Learning System
Tests all adaptive learning endpoints via HTTP requests
"""

import requests
import json
import hashlib
import os
from datetime import datetime

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TEST_FILE_PATH = os.path.join(os.path.dirname(__file__), "test_file.exe")  # Path to a test PE file


def create_test_file():
    """Create a minimal test PE file for testing."""
    # Create a simple test file (this is just for testing, not a real PE)
    test_content = b"MZ\x90\x00" + b"\x00" * 100  # Minimal PE header signature
    with open(TEST_FILE_PATH, "wb") as f:
        f.write(test_content)
    return TEST_FILE_PATH


def test_health_check():
    """Test API health check."""
    print("="*70)
    print("TEST 1: Health Check")
    print("="*70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        print(f"✅ API is online")
        print(f"   Model loaded: {data.get('model_loaded', False)}")
        print(f"   Features: {data.get('num_features', 0)}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_scan_file():
    """Test file scanning endpoint."""
    print("\n" + "="*70)
    print("TEST 2: Scan File (Sample Collection)")
    print("="*70)
    
    # Create test file if it doesn't exist
    if not os.path.exists(TEST_FILE_PATH):
        print("   Creating test file...")
        create_test_file()
    
    try:
        with open(TEST_FILE_PATH, "rb") as f:
            files = {"file": ("test.exe", f, "application/octet-stream")}
            response = requests.post(f"{API_BASE_URL}/scan", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        print(f"✅ File scanned successfully")
        print(f"   Prediction: {data.get('prediction')}")
        print(f"   Confidence: {data.get('confidence', 0):.2%}")
        print(f"   RF Probability: {data.get('random_forest_probability', 0):.2%}")
        
        # Calculate file hash for feedback
        with open(TEST_FILE_PATH, "rb") as f:
            file_content = f.read()
            file_hash = hashlib.sha256(file_content).hexdigest()
        
        return True, file_hash, data
    except Exception as e:
        print(f"❌ Scan failed: {e}")
        return False, None, None


def test_submit_feedback(file_hash, prediction):
    """Test feedback submission."""
    print("\n" + "="*70)
    print("TEST 3: Submit Feedback")
    print("="*70)
    
    if not file_hash:
        print("   ⚠️ Skipping - no file hash available")
        return False
    
    try:
        # Submit feedback (assuming prediction was correct for test)
        feedback_data = {
            "file_hash": file_hash,
            "prediction_was_correct": True,
            "actual_label": prediction,  # Use the prediction as ground truth for test
            "comment": "Test feedback from API test"
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/feedback",
            json=feedback_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        print(f"✅ Feedback submitted successfully")
        print(f"   File hash: {file_hash[:16]}...")
        print(f"   Message: {data.get('message', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Feedback submission failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_statistics():
    """Test getting adaptive learning statistics."""
    print("\n" + "="*70)
    print("TEST 4: Get Adaptive Statistics")
    print("="*70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/adaptive/statistics")
        assert response.status_code == 200
        data = response.json()
        
        print(f"✅ Statistics retrieved")
        print(f"   New samples (total): {data.get('new_samples_total', 0)}")
        print(f"   New samples (with labels): {data.get('new_samples_with_labels', 0)}")
        print(f"   Feedback (unprocessed): {data.get('feedback_unprocessed', 0)}")
        print(f"   Should retrain: {data.get('should_retrain', False)}")
        if data.get('retrain_reason'):
            print(f"   Retrain reason: {data.get('retrain_reason')}")
        
        return True
    except Exception as e:
        print(f"❌ Statistics retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_performance():
    """Test getting performance metrics."""
    print("\n" + "="*70)
    print("TEST 5: Get Performance Metrics")
    print("="*70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/adaptive/performance?days=30")
        assert response.status_code == 200
        data = response.json()
        
        print(f"✅ Performance metrics retrieved")
        
        summary = data.get("summary", {})
        print(f"   Period: {summary.get('period_days', 0)} days")
        print(f"   Total scans: {summary.get('total_scans', 0)}")
        print(f"   Malware detected: {summary.get('malware_detected', 0)}")
        print(f"   Avg confidence: {summary.get('average_confidence', 0):.2%}")
        
        drift = data.get("drift_detection", {})
        print(f"   Drift detected: {drift.get('drift_detected', False)}")
        
        return True
    except Exception as e:
        print(f"❌ Performance metrics retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_trends():
    """Test getting performance trends."""
    print("\n" + "="*70)
    print("TEST 6: Get Performance Trends")
    print("="*70)
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/adaptive/performance/trends?days=30")
        assert response.status_code == 200
        data = response.json()
        
        print(f"✅ Performance trends retrieved")
        
        accuracy_trend = data.get("accuracy", [])
        if accuracy_trend:
            print(f"   Accuracy data points: {len(accuracy_trend)}")
            print(f"   Latest accuracy: {accuracy_trend[-1]:.2%}" if accuracy_trend else "N/A")
        else:
            print(f"   No accuracy data available yet")
        
        return True
    except Exception as e:
        print(f"❌ Trends retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scan_history():
    """Test getting scan history."""
    print("\n" + "="*70)
    print("TEST 7: Get Scan History")
    print("="*70)
    
    try:
        # Note: This requires authentication, so it might fail if not logged in
        response = requests.get(f"{API_BASE_URL}/scan/history?limit=10")
        
        if response.status_code == 401:
            print("   ⚠️ Authentication required (expected)")
            print("   ℹ️ This endpoint requires user login")
            return True  # Not a failure, just needs auth
        
        assert response.status_code == 200
        data = response.json()
        
        print(f"✅ Scan history retrieved")
        print(f"   Total: {data.get('total', 0)}")
        print(f"   Results: {len(data.get('results', []))}")
        
        return True
    except Exception as e:
        print(f"❌ Scan history retrieval failed: {e}")
        return False


def run_all_tests():
    """Run all API tests."""
    print("="*70)
    print("ADAPTIVE LEARNING API - TEST SUITE")
    print("="*70)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    results = []
    
    # Test 1: Health check
    results.append(("Health Check", test_health_check()))
    
    # Test 2: Scan file (collects sample)
    scan_success, file_hash, scan_data = test_scan_file()
    results.append(("Scan File", scan_success))
    
    # Test 3: Submit feedback
    if scan_success and file_hash:
        prediction = scan_data.get("prediction", "benign")
        results.append(("Submit Feedback", test_submit_feedback(file_hash, prediction)))
    else:
        results.append(("Submit Feedback", False))
    
    # Test 4: Get statistics
    results.append(("Get Statistics", test_get_statistics()))
    
    # Test 5: Get performance
    results.append(("Get Performance", test_get_performance()))
    
    # Test 6: Get trends
    results.append(("Get Trends", test_get_trends()))
    
    # Test 7: Scan history (may require auth)
    results.append(("Scan History", test_scan_history()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    import sys
    exit(run_all_tests())

