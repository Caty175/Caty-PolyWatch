#!/usr/bin/env python3
"""
Performance Monitor for Adaptive Malware Detection
Monitors model performance, detects concept drift, and triggers adaptation
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from pymongo import MongoClient
from scipy import stats


class PerformanceMonitor:
    """
    Monitors model performance and detects when adaptation is needed.
    """
    
    def __init__(self, db_name="poly_trial", window_size=1000, drift_threshold=0.05):
        """
        Initialize performance monitor.
        
        Args:
            db_name: MongoDB database name
            window_size: Number of recent predictions to analyze
            drift_threshold: Threshold for detecting concept drift
        """
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.scan_results_collection = self.db["scan_results"]
        self.feedback_collection = self.db["adaptive_feedback"]
        self.drift_alerts_collection = self.db["drift_alerts"]
        
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # In-memory buffers for recent predictions
        self.recent_predictions = deque(maxlen=window_size)
        self.recent_confidences = deque(maxlen=window_size)
    
    def record_prediction(
        self,
        file_hash: str,
        prediction: str,
        confidence: float,
        rf_probability: Optional[float] = None,
        lstm_probability: Optional[float] = None
    ):
        """Record a prediction for monitoring."""
        self.recent_predictions.append({
            "file_hash": file_hash,
            "prediction": prediction,
            "confidence": confidence,
            "rf_probability": rf_probability,
            "lstm_probability": lstm_probability,
            "timestamp": datetime.utcnow()
        })
        self.recent_confidences.append(confidence)
    
    def calculate_accuracy_from_feedback(self, days: int = 7) -> Optional[float]:
        """
        Calculate accuracy based on user feedback.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Accuracy percentage or None if insufficient data
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        feedback = list(self.feedback_collection.find({
            "timestamp": {"$gte": cutoff_date},
            "prediction_was_correct": {"$exists": True}
        }))
        
        if len(feedback) < 10:  # Need at least 10 feedback items
            return None
        
        correct = sum(1 for f in feedback if f.get("prediction_was_correct", False))
        total = len(feedback)
        accuracy = correct / total
        
        return accuracy
    
    def calculate_false_positive_rate(self, days: int = 7) -> Optional[float]:
        """Calculate false positive rate from feedback."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        feedback = list(self.feedback_collection.find({
            "timestamp": {"$gte": cutoff_date},
            "actual_label": "benign",
            "prediction_was_correct": False
        }))
        
        total_benign = self.feedback_collection.count_documents({
            "timestamp": {"$gte": cutoff_date},
            "actual_label": "benign"
        })
        
        if total_benign < 5:
            return None
        
        fpr = len(feedback) / total_benign
        return fpr
    
    def calculate_false_negative_rate(self, days: int = 7) -> Optional[float]:
        """Calculate false negative rate from feedback."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        feedback = list(self.feedback_collection.find({
            "timestamp": {"$gte": cutoff_date},
            "actual_label": "malware",
            "prediction_was_correct": False
        }))
        
        total_malware = self.feedback_collection.count_documents({
            "timestamp": {"$gte": cutoff_date},
            "actual_label": "malware"
        })
        
        if total_malware < 5:
            return None
        
        fnr = len(feedback) / total_malware
        return fnr
    
    def detect_confidence_drift(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if prediction confidence is drifting.
        
        Returns:
            Tuple of (drift_detected: bool, details: dict)
        """
        if len(self.recent_confidences) < 100:
            return False, {"reason": "Insufficient data"}
        
        confidences = np.array(list(self.recent_confidences))
        
        # Split into two halves
        mid = len(confidences) // 2
        first_half = confidences[:mid]
        second_half = confidences[mid:]
        
        # Statistical test for distribution change
        try:
            statistic, p_value = stats.ks_2samp(first_half, second_half)
            
            drift_detected = p_value < 0.05 and abs(statistic) > self.drift_threshold
            
            return drift_detected, {
                "p_value": float(p_value),
                "statistic": float(statistic),
                "first_half_mean": float(np.mean(first_half)),
                "second_half_mean": float(np.mean(second_half)),
                "mean_change": float(np.mean(second_half) - np.mean(first_half))
            }
        except Exception as e:
            return False, {"reason": f"Error in statistical test: {str(e)}"}
    
    def detect_prediction_distribution_drift(self, days: int = 30) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if the distribution of predictions is changing.
        
        Returns:
            Tuple of (drift_detected: bool, details: dict)
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get recent predictions
        recent_scans = list(self.scan_results_collection.find({
            "timestamp": {"$gte": cutoff_date}
        }).sort("timestamp", 1))
        
        if len(recent_scans) < 100:
            return False, {"reason": "Insufficient data"}
        
        # Split into two time periods
        mid = len(recent_scans) // 2
        first_period = recent_scans[:mid]
        second_period = recent_scans[mid:]
        
        # Count predictions
        first_malware = sum(1 for s in first_period if s.get("prediction") == "malware")
        second_malware = sum(1 for s in second_period if s.get("prediction") == "malware")
        
        first_malware_rate = first_malware / len(first_period)
        second_malware_rate = second_malware / len(second_period)
        
        change = abs(second_malware_rate - first_malware_rate)
        drift_detected = change > self.drift_threshold
        
        return drift_detected, {
            "first_period_malware_rate": first_malware_rate,
            "second_period_malware_rate": second_malware_rate,
            "change": change,
            "first_period_samples": len(first_period),
            "second_period_samples": len(second_period)
        }
    
    def detect_performance_degradation(self, baseline_accuracy: float, days: int = 7) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if model performance is degrading.
        
        Args:
            baseline_accuracy: Baseline accuracy to compare against
            days: Number of days to analyze
            
        Returns:
            Tuple of (degradation_detected: bool, details: dict)
        """
        current_accuracy = self.calculate_accuracy_from_feedback(days=days)
        
        if current_accuracy is None:
            return False, {"reason": "Insufficient feedback data"}
        
        degradation = baseline_accuracy - current_accuracy
        degradation_detected = degradation > 0.05  # 5% drop threshold
        
        return degradation_detected, {
            "baseline_accuracy": baseline_accuracy,
            "current_accuracy": current_accuracy,
            "degradation": degradation,
            "degradation_percent": degradation * 100
        }
    
    def check_all_drift_indicators(self, baseline_accuracy: float = 0.90) -> Dict[str, Any]:
        """
        Check all drift indicators and return comprehensive report.
        
        Args:
            baseline_accuracy: Baseline accuracy for comparison
            
        Returns:
            Dictionary with all drift detection results
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "confidence_drift": self.detect_confidence_drift(),
            "prediction_distribution_drift": self.detect_prediction_distribution_drift(),
            "performance_degradation": self.detect_performance_degradation(baseline_accuracy),
            "current_accuracy": self.calculate_accuracy_from_feedback(),
            "false_positive_rate": self.calculate_false_positive_rate(),
            "false_negative_rate": self.calculate_false_negative_rate()
        }
        
        # Determine if any drift detected
        any_drift = (
            results["confidence_drift"][0] or
            results["prediction_distribution_drift"][0] or
            results["performance_degradation"][0]
        )
        
        results["drift_detected"] = any_drift
        
        # Store alert if drift detected
        if any_drift:
            self.drift_alerts_collection.insert_one({
                "drift_detected": True,
                "details": results,
                "timestamp": datetime.utcnow()
            })
        
        return results
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        total_scans = self.scan_results_collection.count_documents({
            "timestamp": {"$gte": cutoff_date}
        })
        
        malware_detected = self.scan_results_collection.count_documents({
            "timestamp": {"$gte": cutoff_date},
            "prediction": "malware"
        })
        
        avg_confidence = self.scan_results_collection.aggregate([
            {"$match": {"timestamp": {"$gte": cutoff_date}}},
            {"$group": {"_id": None, "avg_confidence": {"$avg": "$confidence"}}}
        ])
        
        avg_conf = 0.0
        try:
            result = next(avg_confidence, None)
            if result:
                avg_conf = result.get("avg_confidence", 0.0)
        except:
            pass
        
        return {
            "period_days": days,
            "total_scans": total_scans,
            "malware_detected": malware_detected,
            "benign_detected": total_scans - malware_detected,
            "malware_rate": malware_detected / total_scans if total_scans > 0 else 0,
            "average_confidence": float(avg_conf),
            "accuracy_from_feedback": self.calculate_accuracy_from_feedback(days=days),
            "false_positive_rate": self.calculate_false_positive_rate(days=days),
            "false_negative_rate": self.calculate_false_negative_rate(days=days)
        }

