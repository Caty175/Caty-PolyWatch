# Adaptive Learning System for Polymorphic Malware Detection

## Overview

This document describes the **adaptive learning system** that makes the malware detection model continuously learn and adapt from new data, user feedback, and evolving malware patterns.

## What Makes It Adaptive?

The system is now **truly adaptive** because it:

1. ✅ **Collects new samples** automatically from every scan
2. ✅ **Accepts user feedback** on prediction accuracy
3. ✅ **Monitors performance** and detects concept drift
4. ✅ **Retrains models automatically** when thresholds are met
5. ✅ **Preserves old knowledge** while learning from new data
6. ✅ **Tracks performance trends** over time

## Architecture

### Components

1. **AdaptiveLearningManager** (`server/adaptive_learning.py`)
   - Manages collection of new samples
   - Stores user feedback
   - Tracks retraining history
   - Determines when retraining is needed

2. **PerformanceMonitor** (`server/performance_monitor.py`)
   - Monitors prediction accuracy
   - Detects concept drift
   - Calculates false positive/negative rates
   - Tracks performance trends

3. **AdaptiveRetrainer** (`Model/adaptive_retrain.py`)
   - Retrains Random Forest model with new samples
   - Retrains LSTM model with new behavioral data
   - Combines original training data with new samples
   - Validates performance before deploying new models

4. **API Integration** (`server/api_server.py`)
   - Automatically collects samples from scans
   - Provides feedback endpoints
   - Exposes statistics and performance metrics

## How It Works

### 1. Sample Collection

Every time a file is scanned:

```python
# Automatically happens in /scan endpoint
adaptive_manager.add_sample(
    file_hash=file_hash,
    features=extracted_features,
    prediction=prediction,
    confidence=confidence,
    rf_probability=rf_prob,
    lstm_probability=lstm_prob,
    behavioral_features=behavioral_data,
    user_id=user_id
)
```

### 2. Feedback Collection

Users can provide feedback on predictions:

```bash
# Submit feedback via API
curl -X POST "http://localhost:8000/api/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "file_hash": "abc123...",
    "prediction_was_correct": false,
    "actual_label": "benign",
    "comment": "False positive"
  }'
```

### 3. Performance Monitoring

The system continuously monitors:

- **Prediction accuracy** from user feedback
- **Confidence drift** (statistical tests)
- **Prediction distribution changes**
- **False positive/negative rates**

### 4. Automatic Retraining

When thresholds are met:

- **1000+ new samples** with ground truth labels, OR
- **100+ feedback items**, AND
- **At least 7 days** since last retraining

The system automatically:

1. Loads new samples
2. Combines with original training data
3. Retrains the model(s)
4. Validates performance
5. Deploys if performance maintained/improved
6. Backs up old model before replacing

## Usage

### Manual Retraining

```bash
# Retrain both models
python Model/adaptive_retrain.py --model both

# Retrain only Random Forest
python Model/adaptive_retrain.py --model rf

# Retrain only LSTM
python Model/adaptive_retrain.py --model lstm

# Use only new samples (no original data)
python Model/adaptive_retrain.py --model both --no-original-data

# Limit number of new samples
python Model/adaptive_retrain.py --model rf --max-samples 1000
```

### Scheduled Retraining

```bash
# Check if retraining is needed (dry run)
python Model/scheduled_retrain.py --dry-run

# Run scheduled retraining
python Model/scheduled_retrain.py

# Force retraining
python Model/scheduled_retrain.py --force

# Retrain specific model
python Model/scheduled_retrain.py --model rf
```

### Setting Up Cron/Task Scheduler

**Linux/Mac (Cron):**
```bash
# Run daily at 2 AM
0 2 * * * cd /path/to/project && python Model/scheduled_retrain.py >> /var/log/adaptive_retrain.log 2>&1
```

**Windows (Task Scheduler):**
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Daily at 2:00 AM
4. Action: Start a program
5. Program: `python`
6. Arguments: `Model/scheduled_retrain.py`
7. Start in: `C:\path\to\project`

## API Endpoints

### Submit Feedback

```http
POST /api/feedback
Content-Type: application/json

{
  "file_hash": "sha256_hash",
  "prediction_was_correct": false,
  "actual_label": "benign",
  "comment": "Optional comment"
}
```

### Get Statistics

```http
GET /api/adaptive/statistics
```

Returns:
```json
{
  "new_samples_total": 1500,
  "new_samples_with_labels": 1200,
  "new_samples_used": 800,
  "feedback_unprocessed": 45,
  "should_retrain": true,
  "retrain_reason": "1200 new samples with ground truth available"
}
```

### Get Performance Metrics

```http
GET /api/adaptive/performance?days=30
```

Returns performance summary and drift detection results.

### Get Performance Trends

```http
GET /api/adaptive/performance/trends?days=30
```

Returns time series data for visualization.

## Database Collections

The adaptive learning system uses these MongoDB collections:

1. **`adaptive_new_samples`**
   - Stores new samples from scans
   - Fields: file_hash, features, prediction, ground_truth, etc.

2. **`adaptive_feedback`**
   - Stores user feedback
   - Fields: file_hash, prediction_was_correct, actual_label, etc.

3. **`adaptive_retraining_history`**
   - Tracks retraining attempts
   - Fields: model_type, samples_used, old_metrics, new_metrics, success

4. **`adaptive_performance_metrics`**
   - Stores periodic performance snapshots
   - Fields: accuracy, precision, recall, roc_auc, timestamp

5. **`drift_alerts`**
   - Stores concept drift detection alerts
   - Fields: drift_detected, details, timestamp

## Configuration

### Adaptive Learning Manager

```python
adaptive_manager = AdaptiveLearningManager(
    retrain_threshold=1000,      # Samples needed to trigger retraining
    feedback_threshold=100,       # Feedback items needed
    min_retrain_interval_days=7  # Minimum days between retraining
)
```

### Performance Monitor

```python
performance_monitor = PerformanceMonitor(
    window_size=1000,        # Recent predictions to analyze
    drift_threshold=0.05     # Threshold for drift detection
)
```

## Monitoring and Maintenance

### Check System Status

```python
from server.adaptive_learning import AdaptiveLearningManager

manager = AdaptiveLearningManager()
stats = manager.get_statistics()

print(f"New samples: {stats['new_samples_with_labels']}")
print(f"Should retrain: {stats['should_retrain']}")
```

### Check Performance

```python
from server.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
summary = monitor.get_performance_summary(days=30)
drift = monitor.check_all_drift_indicators()

print(f"Accuracy: {summary['accuracy_from_feedback']}")
print(f"Drift detected: {drift['drift_detected']}")
```

## Best Practices

1. **Regular Monitoring**
   - Check statistics weekly
   - Review performance trends monthly
   - Investigate drift alerts promptly

2. **Feedback Quality**
   - Encourage users to provide feedback
   - Verify ground truth labels when possible
   - Filter out low-quality feedback

3. **Retraining Strategy**
   - Let automatic retraining run on schedule
   - Manually trigger if drift detected
   - Review retraining history regularly

4. **Model Validation**
   - System automatically validates before deploying
   - Keeps backups of old models
   - Can rollback if needed

## Troubleshooting

### Retraining Not Triggering

- Check if thresholds are met: `GET /api/adaptive/statistics`
- Verify minimum interval has passed
- Check if samples have ground truth labels

### Performance Degrading

- Review drift detection: `GET /api/adaptive/performance`
- Check feedback accuracy
- Consider manual retraining with more data

### Database Issues

- Ensure MongoDB is running
- Check collection indexes are created
- Verify database permissions

## Future Enhancements

Potential improvements:

1. **Active Learning**: Select most informative samples for labeling
2. **Transfer Learning**: Pre-train on larger datasets
3. **Ensemble Updates**: Add new models without retraining all
4. **Online Learning**: Update models incrementally without full retraining
5. **A/B Testing**: Test new models before full deployment

## Conclusion

The adaptive learning system transforms your malware detection from a **static model** into a **continuously improving system** that:

- Learns from every scan
- Adapts to new threats
- Improves with user feedback
- Detects performance issues
- Maintains high accuracy over time

This makes your system truly **adaptive** and ready for production deployment! 🚀

