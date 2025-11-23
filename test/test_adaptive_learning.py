#!/usr/bin/env python3
"""
Test script for Adaptive Learning System
Tests all components: sample collection, feedback, performance monitoring, and retraining
"""

import os
import sys
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.adaptive_learning import AdaptiveLearningManager
from server.performance_monitor import PerformanceMonitor


def test_adaptive_learning_manager():
    """Test AdaptiveLearningManager functionality."""
    print("="*70)
    print("TESTING: AdaptiveLearningManager")
    print("="*70)
    
    manager = AdaptiveLearningManager()
    
    # Test 1: Add sample
    print("\n1. Testing add_sample()...")
    test_file_hash = hashlib.sha256(b"test_file_content").hexdigest()
    test_features = {
        "general.size": 1024,
        "general.imports": 10,
        "header.coff.timestamp": 1234567890
    }
    
    result = manager.add_sample(
        file_hash=test_file_hash,
        features=test_features,
        prediction="malware",
        confidence=0.85,
        rf_probability=0.80,
        lstm_probability=0.90,
        ground_truth="malware",
        filename="test.exe"
    )
    
    assert result == True, "Sample should be added successfully"
    print("   ✅ Sample added successfully")
    
    # Test 2: Add duplicate sample (should return False)
    result2 = manager.add_sample(
        file_hash=test_file_hash,
        features=test_features,
        prediction="malware",
        confidence=0.85
    )
    assert result2 == False, "Duplicate sample should return False"
    print("   ✅ Duplicate detection works")
    
    # Test 3: Add feedback
    print("\n2. Testing add_feedback()...")
    feedback_result = manager.add_feedback(
        file_hash=test_file_hash,
        prediction_was_correct=True,
        actual_label="malware",
        comment="Test feedback"
    )
    assert feedback_result == True, "Feedback should be added"
    print("   ✅ Feedback added successfully")
    
    # Test 4: Get statistics
    print("\n3. Testing get_statistics()...")
    stats = manager.get_statistics()
    assert "new_samples_total" in stats
    assert "feedback_total" in stats
    print(f"   ✅ Statistics retrieved:")
    print(f"      - New samples: {stats['new_samples_total']}")
    print(f"      - Feedback: {stats['feedback_total']}")
    
    # Test 5: Get samples for training
    print("\n4. Testing get_samples_for_training()...")
    samples = manager.get_samples_for_training(limit=10, require_ground_truth=True)
    assert isinstance(samples, list)
    print(f"   ✅ Retrieved {len(samples)} samples for training")
    
    # Test 6: Check retraining trigger
    print("\n5. Testing should_trigger_retraining()...")
    should_retrain, reason = manager.should_trigger_retraining()
    print(f"   ✅ Retraining check: {should_retrain}")
    print(f"      Reason: {reason}")
    
    print("\n✅ All AdaptiveLearningManager tests passed!")
    return True


def test_performance_monitor():
    """Test PerformanceMonitor functionality."""
    print("\n" + "="*70)
    print("TESTING: PerformanceMonitor")
    print("="*70)
    
    monitor = PerformanceMonitor()
    
    # Test 1: Record predictions
    print("\n1. Testing record_prediction()...")
    for i in range(50):
        monitor.record_prediction(
            file_hash=f"hash_{i}",
            prediction="malware" if i % 3 == 0 else "benign",
            confidence=0.7 + (i % 3) * 0.1,
            rf_probability=0.6 + (i % 3) * 0.15
        )
    print("   ✅ Recorded 50 predictions")
    
    # Test 2: Get performance summary
    print("\n2. Testing get_performance_summary()...")
    summary = monitor.get_performance_summary(days=30)
    assert "total_scans" in summary
    assert "average_confidence" in summary
    print(f"   ✅ Performance summary retrieved:")
    print(f"      - Total scans: {summary.get('total_scans', 0)}")
    print(f"      - Avg confidence: {summary.get('average_confidence', 0):.2f}")
    
    # Test 3: Detect confidence drift
    print("\n3. Testing detect_confidence_drift()...")
    drift_detected, details = monitor.detect_confidence_drift()
    print(f"   ✅ Drift detection: {drift_detected}")
    if details:
        print(f"      Details: {details}")
    
    # Test 4: Check all drift indicators
    print("\n4. Testing check_all_drift_indicators()...")
    drift_results = monitor.check_all_drift_indicators(baseline_accuracy=0.90)
    assert "drift_detected" in drift_results
    print(f"   ✅ Drift check complete:")
    print(f"      - Drift detected: {drift_results.get('drift_detected', False)}")
    
    print("\n✅ All PerformanceMonitor tests passed!")
    return True


def test_integration():
    """Test integration between components."""
    print("\n" + "="*70)
    print("TESTING: Integration")
    print("="*70)
    
    manager = AdaptiveLearningManager()
    monitor = PerformanceMonitor()
    
    # Simulate a scan workflow
    print("\n1. Simulating scan workflow...")
    
    # Add multiple samples
    for i in range(10):
        file_hash = hashlib.sha256(f"test_file_{i}".encode()).hexdigest()
        features = {
            "general.size": 1024 * (i + 1),
            "general.imports": 5 + i,
            "header.coff.timestamp": 1234567890 + i
        }
        
        manager.add_sample(
            file_hash=file_hash,
            features=features,
            prediction="malware" if i % 2 == 0 else "benign",
            confidence=0.7 + (i % 3) * 0.1,
            ground_truth="malware" if i % 2 == 0 else "benign",
            filename=f"test_{i}.exe"
        )
        
        monitor.record_prediction(
            file_hash=file_hash,
            prediction="malware" if i % 2 == 0 else "benign",
            confidence=0.7 + (i % 3) * 0.1
        )
    
    print("   ✅ Added 10 samples and recorded predictions")
    
    # Add some feedback
    print("\n2. Adding feedback...")
    for i in range(5):
        file_hash = hashlib.sha256(f"test_file_{i}".encode()).hexdigest()
        manager.add_feedback(
            file_hash=file_hash,
            prediction_was_correct=(i % 2 == 0),
            actual_label="malware" if i % 2 == 0 else "benign"
        )
    print("   ✅ Added 5 feedback items")
    
    # Check statistics
    print("\n3. Checking final statistics...")
    stats = manager.get_statistics()
    print(f"   ✅ Final stats:")
    print(f"      - New samples: {stats['new_samples_total']}")
    print(f"      - Feedback: {stats['feedback_total']}")
    print(f"      - Should retrain: {stats['should_retrain']}")
    
    print("\n✅ Integration tests passed!")
    return True


def cleanup_test_data():
    """Clean up test data (optional)."""
    print("\n" + "="*70)
    print("CLEANUP: Test Data")
    print("="*70)
    
    manager = AdaptiveLearningManager()
    
    # Get test samples (those with "test" in filename)
    test_samples = list(manager.new_samples_collection.find({
        "filename": {"$regex": "^test"}}
    ))
    
    if test_samples:
        print(f"\nFound {len(test_samples)} test samples")
        response = input("Delete test samples? (y/n): ")
        if response.lower() == 'y':
            for sample in test_samples:
                manager.new_samples_collection.delete_one({"_id": sample["_id"]})
            print("   ✅ Test samples deleted")
    else:
        print("\n   ℹ️ No test samples found")


def main():
    """Run all tests."""
    print("="*70)
    print("ADAPTIVE LEARNING SYSTEM - TEST SUITE")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    try:
        # Run tests
        test_adaptive_learning_manager()
        test_performance_monitor()
        test_integration()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        
        # Optional cleanup
        cleanup_test_data()
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

