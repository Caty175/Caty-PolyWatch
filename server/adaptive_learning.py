#!/usr/bin/env python3
"""
Adaptive Learning Manager for Malware Detection System
Manages collection of new samples, user feedback, and triggers retraining
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
from pymongo import MongoClient
import numpy as np


class AdaptiveLearningManager:
    """
    Manages adaptive learning for the malware detection system.
    Collects new samples, user feedback, and triggers retraining when needed.
    """
    
    def __init__(self, db_name="poly_trial", retrain_threshold=1000, 
                 feedback_threshold=100, min_retrain_interval_days=7):
        """
        Initialize adaptive learning manager.
        
        Args:
            db_name: MongoDB database name
            retrain_threshold: Number of new samples before triggering retraining
            feedback_threshold: Number of feedback items before triggering retraining
            min_retrain_interval_days: Minimum days between retraining attempts
        """
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.new_samples_collection = self.db["adaptive_new_samples"]
        self.feedback_collection = self.db["adaptive_feedback"]
        self.retraining_history_collection = self.db["adaptive_retraining_history"]
        self.performance_metrics_collection = self.db["adaptive_performance_metrics"]
        
        self.retrain_threshold = retrain_threshold
        self.feedback_threshold = feedback_threshold
        self.min_retrain_interval_days = min_retrain_interval_days
        
        # Create indexes for efficient queries
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes for efficient queries."""
        # Index on file_hash for quick lookups
        self.new_samples_collection.create_index("file_hash")
        self.new_samples_collection.create_index("timestamp")
        self.new_samples_collection.create_index("used_for_training", sparse=True)
        
        self.feedback_collection.create_index("file_hash")
        self.feedback_collection.create_index("timestamp")
        self.feedback_collection.create_index("processed", sparse=True)
        
        self.retraining_history_collection.create_index("timestamp")
        self.performance_metrics_collection.create_index("timestamp")
    
    def calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA256 hash of file content."""
        return hashlib.sha256(file_content).hexdigest()
    
    def add_sample(
        self,
        file_hash: str,
        features: Dict[str, Any],
        prediction: str,
        confidence: float,
        rf_probability: Optional[float] = None,
        lstm_probability: Optional[float] = None,
        behavioral_features: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[str] = None,
        user_id: Optional[str] = None,
        filename: Optional[str] = None
    ) -> bool:
        """
        Store a new sample for potential retraining.
        
        Args:
            file_hash: SHA256 hash of the file
            features: Static features extracted from PE file
            prediction: Model prediction ("malware" or "benign")
            confidence: Prediction confidence
            rf_probability: Random Forest probability (optional)
            lstm_probability: LSTM probability (optional)
            behavioral_features: Behavioral features from sandbox (optional)
            ground_truth: Actual label if known ("malware" or "benign")
            user_id: User ID who submitted the file
            filename: Original filename
            
        Returns:
            True if sample was added, False if duplicate
        """
        # Check if sample already exists
        existing = self.new_samples_collection.find_one({"file_hash": file_hash})
        if existing:
            # Update if we have new information (e.g., ground truth)
            if ground_truth and not existing.get("ground_truth"):
                self.new_samples_collection.update_one(
                    {"file_hash": file_hash},
                    {"$set": {
                        "ground_truth": ground_truth,
                        "updated_at": datetime.utcnow()
                    }}
                )
            return False
        
        sample_doc = {
            "file_hash": file_hash,
            "filename": filename,
            "features": features,
            "prediction": prediction,
            "confidence": confidence,
            "rf_probability": rf_probability,
            "lstm_probability": lstm_probability,
            "behavioral_features": behavioral_features,
            "ground_truth": ground_truth,
            "user_id": user_id,
            "used_for_training": False,
            "timestamp": datetime.utcnow(),
            "created_at": datetime.utcnow()
        }
        
        self.new_samples_collection.insert_one(sample_doc)
        return True
    
    def add_feedback(
        self,
        file_hash: str,
        prediction_was_correct: bool,
        actual_label: Optional[str] = None,
        user_id: Optional[str] = None,
        comment: Optional[str] = None
    ) -> bool:
        """
        Add user feedback on a prediction.
        
        Args:
            file_hash: SHA256 hash of the file
            prediction_was_correct: Whether the prediction was correct
            actual_label: Actual label ("malware" or "benign") if known
            user_id: User ID providing feedback
            comment: Optional comment from user
            
        Returns:
            True if feedback was added
        """
        feedback_doc = {
            "file_hash": file_hash,
            "prediction_was_correct": prediction_was_correct,
            "actual_label": actual_label,
            "user_id": user_id,
            "comment": comment,
            "processed": False,
            "timestamp": datetime.utcnow(),
            "created_at": datetime.utcnow()
        }
        
        self.feedback_collection.insert_one(feedback_doc)
        
        # Update ground truth in new_samples if available
        if actual_label:
            self.new_samples_collection.update_one(
                {"file_hash": file_hash},
                {"$set": {"ground_truth": actual_label}}
            )
        
        return True
    
    def get_new_samples_count(self, with_ground_truth: bool = False) -> int:
        """Get count of new samples available for training."""
        query = {"used_for_training": False}
        if with_ground_truth:
            query["ground_truth"] = {"$exists": True, "$ne": None}
        return self.new_samples_collection.count_documents(query)
    
    def get_feedback_count(self, processed: bool = False) -> int:
        """Get count of feedback items."""
        return self.feedback_collection.count_documents({"processed": processed})
    
    def get_samples_for_training(
        self,
        limit: Optional[int] = None,
        require_ground_truth: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get samples ready for training.
        
        Args:
            limit: Maximum number of samples to return
            require_ground_truth: Only return samples with ground truth labels
            
        Returns:
            List of sample documents
        """
        query = {"used_for_training": False}
        if require_ground_truth:
            query["ground_truth"] = {"$exists": True, "$ne": None}
        
        cursor = self.new_samples_collection.find(query).sort("timestamp", 1)
        if limit:
            cursor = cursor.limit(limit)
        
        return list(cursor)
    
    def get_feedback_for_training(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get unprocessed feedback items."""
        cursor = self.feedback_collection.find({"processed": False}).sort("timestamp", 1)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)
    
    def mark_samples_as_used(self, file_hashes: List[str]):
        """Mark samples as used for training."""
        self.new_samples_collection.update_many(
            {"file_hash": {"$in": file_hashes}},
            {"$set": {"used_for_training": True, "used_at": datetime.utcnow()}}
        )
    
    def mark_feedback_as_processed(self, feedback_ids: List[str]):
        """Mark feedback items as processed."""
        from bson import ObjectId
        object_ids = [ObjectId(fid) for fid in feedback_ids if ObjectId.is_valid(fid)]
        if object_ids:
            self.feedback_collection.update_many(
                {"_id": {"$in": object_ids}},
                {"$set": {"processed": True, "processed_at": datetime.utcnow()}}
            )
    
    def should_trigger_retraining(self) -> tuple[bool, str]:
        """
        Check if retraining should be triggered.
        
        Returns:
            Tuple of (should_retrain: bool, reason: str)
        """
        # Check minimum interval
        last_retrain = self.retraining_history_collection.find_one(
            sort=[("timestamp", -1)]
        )
        if last_retrain:
            last_retrain_time = last_retrain.get("timestamp")
            if isinstance(last_retrain_time, str):
                last_retrain_time = datetime.fromisoformat(last_retrain_time.replace('Z', '+00:00'))
            elif not isinstance(last_retrain_time, datetime):
                last_retrain_time = datetime.utcnow()
            
            days_since = (datetime.utcnow() - last_retrain_time.replace(tzinfo=None)).days
            if days_since < self.min_retrain_interval_days:
                return False, f"Too soon since last retraining ({days_since} days ago)"
        
        # Check new samples threshold
        new_samples_count = self.get_new_samples_count(with_ground_truth=True)
        if new_samples_count >= self.retrain_threshold:
            return True, f"{new_samples_count} new samples with ground truth available"
        
        # Check feedback threshold
        feedback_count = self.get_feedback_count(processed=False)
        if feedback_count >= self.feedback_threshold:
            return True, f"{feedback_count} feedback items available"
        
        return False, "No retraining triggers met"
    
    def record_retraining(
        self,
        model_type: str,
        samples_used: int,
        feedback_used: int,
        old_metrics: Dict[str, float],
        new_metrics: Dict[str, float],
        success: bool,
        error_message: Optional[str] = None
    ):
        """Record retraining attempt in history."""
        retrain_doc = {
            "model_type": model_type,  # "rf", "lstm", or "both"
            "samples_used": samples_used,
            "feedback_used": feedback_used,
            "old_metrics": old_metrics,
            "new_metrics": new_metrics,
            "success": success,
            "error_message": error_message,
            "timestamp": datetime.utcnow()
        }
        self.retraining_history_collection.insert_one(retrain_doc)
    
    def record_performance_metrics(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        roc_auc: float,
        false_positive_rate: float,
        false_negative_rate: float,
        model_type: str = "combined"
    ):
        """Record current performance metrics."""
        metrics_doc = {
            "model_type": model_type,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "roc_auc": roc_auc,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "timestamp": datetime.utcnow()
        }
        self.performance_metrics_collection.insert_one(metrics_doc)
    
    def get_performance_trends(self, days: int = 30) -> Dict[str, List[float]]:
        """Get performance metrics over time."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        cursor = self.performance_metrics_collection.find(
            {"timestamp": {"$gte": cutoff_date}}
        ).sort("timestamp", 1)
        
        trends = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "roc_auc": [],
            "timestamps": []
        }
        
        for doc in cursor:
            trends["accuracy"].append(doc.get("accuracy", 0))
            trends["precision"].append(doc.get("precision", 0))
            trends["recall"].append(doc.get("recall", 0))
            trends["f1_score"].append(doc.get("f1_score", 0))
            trends["roc_auc"].append(doc.get("roc_auc", 0))
            trends["timestamps"].append(doc.get("timestamp").isoformat() if doc.get("timestamp") else "")
        
        return trends
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about adaptive learning."""
        return {
            "new_samples_total": self.new_samples_collection.count_documents({}),
            "new_samples_available": self.get_new_samples_count(with_ground_truth=False),
            "new_samples_with_labels": self.get_new_samples_count(with_ground_truth=True),
            "new_samples_used": self.new_samples_collection.count_documents({"used_for_training": True}),
            "feedback_total": self.feedback_collection.count_documents({}),
            "feedback_unprocessed": self.get_feedback_count(processed=False),
            "feedback_processed": self.get_feedback_count(processed=True),
            "retraining_attempts": self.retraining_history_collection.count_documents({}),
            "last_retraining": self._get_last_retraining_info(),
            "should_retrain": self.should_trigger_retraining()[0]
        }
    
    def _get_last_retraining_info(self) -> Optional[Dict[str, Any]]:
        """Get information about last retraining."""
        last = self.retraining_history_collection.find_one(sort=[("timestamp", -1)])
        if last:
            return {
                "timestamp": last.get("timestamp").isoformat() if isinstance(last.get("timestamp"), datetime) else str(last.get("timestamp")),
                "model_type": last.get("model_type"),
                "success": last.get("success"),
                "samples_used": last.get("samples_used", 0)
            }
        return None

