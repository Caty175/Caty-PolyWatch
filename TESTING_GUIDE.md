# Testing Guide for Adaptive Learning System

This guide explains how to test the entire adaptive learning system, including API endpoints.

## Prerequisites

1. **Start MongoDB:**
   ```bash
   # Windows
   net start MongoDB
   
   # Linux/Mac
   sudo systemctl start mongod
   # or
   mongod
   ```

2. **Start the API Server:**
   ```bash
   cd server
   python api_server.py
   ```
   
   Server should be running on `http://localhost:8000`

3. **Install Test Dependencies:**
   ```bash
   pip install requests
   ```

## Testing Methods

### Method 1: Unit Tests (Python)

Test the core adaptive learning components:

```bash
python test/test_adaptive_learning.py
```

**What it tests:**
- ✅ Sample collection
- ✅ Feedback storage
- ✅ Statistics retrieval
- ✅ Retraining triggers
- ✅ Performance monitoring
- ✅ Drift detection
- ✅ Integration between components

**Expected output:**
```
✅ All AdaptiveLearningManager tests passed!
✅ All PerformanceMonitor tests passed!
✅ Integration tests passed!
✅ ALL TESTS PASSED!
```

### Method 2: API Tests (HTTP Requests)

Test the API endpoints:

```bash
python test/test_adaptive_api.py
```

**What it tests:**
- ✅ Health check
- ✅ File scanning (sample collection)
- ✅ Feedback submission
- ✅ Statistics endpoint
- ✅ Performance metrics
- ✅ Performance trends
- ✅ Scan history

**Expected output:**
```
✅ API is online
✅ File scanned successfully
✅ Feedback submitted successfully
✅ Statistics retrieved
✅ Performance metrics retrieved
...
🎉 All tests passed!
```

### Method 3: Manual API Testing (cURL)

Test endpoints manually using cURL:

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Scan a File
```bash
curl -X POST "http://localhost:8000/scan" \
  -F "file=@path/to/file.exe"
```

**Save the response** - you'll need the file hash for feedback!

#### 3. Submit Feedback
```bash
curl -X POST "http://localhost:8000/api/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "file_hash": "sha256_hash_from_scan_response",
    "prediction_was_correct": true,
    "actual_label": "benign",
    "comment": "Test feedback"
  }'
```

#### 4. Get Statistics
```bash
curl http://localhost:8000/api/adaptive/statistics
```

#### 5. Get Performance Metrics
```bash
curl "http://localhost:8000/api/adaptive/performance?days=30"
```

#### 6. Get Performance Trends
```bash
curl "http://localhost:8000/api/adaptive/performance/trends?days=30"
```

### Method 4: Python Requests Script

Create a custom test script:

```python
import requests
import hashlib

API_URL = "http://localhost:8000"

# 1. Scan a file
with open("test.exe", "rb") as f:
    files = {"file": ("test.exe", f)}
    response = requests.post(f"{API_URL}/scan", files=files)
    result = response.json()
    print(f"Prediction: {result['prediction']}")

# 2. Calculate file hash
with open("test.exe", "rb") as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()

# 3. Submit feedback
feedback = {
    "file_hash": file_hash,
    "prediction_was_correct": True,
    "actual_label": result['prediction']
}
requests.post(f"{API_URL}/api/feedback", json=feedback)

# 4. Check statistics
stats = requests.get(f"{API_URL}/api/adaptive/statistics").json()
print(f"New samples: {stats['new_samples_with_labels']}")
```

## Step-by-Step Testing Workflow

### Phase 1: Basic Functionality

1. **Start the server:**
   ```bash
   python server/api_server.py
   ```

2. **Run unit tests:**
   ```bash
   python test/test_adaptive_learning.py
   ```
   ✅ Should pass all tests

3. **Run API tests:**
   ```bash
   python test/test_adaptive_api.py
   ```
   ✅ Should pass all tests

### Phase 2: Sample Collection

1. **Scan multiple files:**
   ```bash
   # Scan file 1
   curl -X POST "http://localhost:8000/scan" -F "file=@file1.exe"
   
   # Scan file 2
   curl -X POST "http://localhost:8000/scan" -F "file=@file2.exe"
   
   # Scan file 3
   curl -X POST "http://localhost:8000/scan" -F "file=@file3.exe"
   ```

2. **Check statistics:**
   ```bash
   curl http://localhost:8000/api/adaptive/statistics
   ```
   
   Should show:
   ```json
   {
     "new_samples_total": 3,
     "new_samples_with_labels": 0,  // No labels yet
     ...
   }
   ```

### Phase 3: Feedback Collection

1. **Get file hash from scan response** (or calculate it):
   ```python
   import hashlib
   with open("file1.exe", "rb") as f:
       file_hash = hashlib.sha256(f.read()).hexdigest()
   print(file_hash)
   ```

2. **Submit feedback:**
   ```bash
   curl -X POST "http://localhost:8000/api/feedback" \
     -H "Content-Type: application/json" \
     -d '{
       "file_hash": "your_file_hash_here",
       "prediction_was_correct": false,
       "actual_label": "benign",
       "comment": "False positive"
     }'
   ```

3. **Check statistics again:**
   ```bash
   curl http://localhost:8000/api/adaptive/statistics
   ```
   
   Should show:
   ```json
   {
     "new_samples_with_labels": 1,  // Now has label!
     "feedback_unprocessed": 1,
     ...
   }
   ```

### Phase 4: Performance Monitoring

1. **Get performance summary:**
   ```bash
   curl "http://localhost:8000/api/adaptive/performance?days=7"
   ```

2. **Check for drift:**
   ```bash
   curl "http://localhost:8000/api/adaptive/performance?days=30" | jq '.drift_detection'
   ```

### Phase 5: Retraining (Advanced)

1. **Check if retraining is needed:**
   ```bash
   python Model/scheduled_retrain.py --dry-run
   ```

2. **Manually trigger retraining:**
   ```bash
   # Retrain Random Forest
   python Model/adaptive_retrain.py --model rf
   
   # Retrain LSTM
   python Model/adaptive_retrain.py --model lstm
   
   # Retrain both
   python Model/adaptive_retrain.py --model both
   ```

3. **Check retraining history:**
   ```python
   from server.adaptive_learning import AdaptiveLearningManager
   manager = AdaptiveLearningManager()
   stats = manager.get_statistics()
   print(stats['last_retraining'])
   ```

## Testing Scenarios

### Scenario 1: False Positive Feedback

1. Scan a benign file that's misclassified as malware
2. Submit feedback:
   ```json
   {
     "file_hash": "...",
     "prediction_was_correct": false,
     "actual_label": "benign",
     "comment": "False positive - this is a legitimate file"
   }
   ```
3. Check that feedback is recorded
4. Eventually retrain to improve false positive rate

### Scenario 2: False Negative Feedback

1. Scan a malware file that's misclassified as benign
2. Submit feedback:
   ```json
   {
     "file_hash": "...",
     "prediction_was_correct": false,
     "actual_label": "malware",
     "comment": "False negative - this is actually malware"
   }
   ```
3. Check that feedback is recorded
4. Eventually retrain to improve detection rate

### Scenario 3: Concept Drift Detection

1. Scan many files over time
2. Monitor performance trends:
   ```bash
   curl "http://localhost:8000/api/adaptive/performance/trends?days=30"
   ```
3. Check drift detection:
   ```bash
   curl "http://localhost:8000/api/adaptive/performance?days=30"
   ```
4. If drift detected, trigger retraining

### Scenario 4: Automatic Retraining

1. Collect 1000+ samples with labels (or 100+ feedback items)
2. Wait 7+ days since last retraining
3. Run scheduled retraining:
   ```bash
   python Model/scheduled_retrain.py
   ```
4. Verify new model is deployed
5. Check performance improved

## Verification Checklist

After testing, verify:

- [ ] Samples are being collected from scans
- [ ] Feedback can be submitted and stored
- [ ] Statistics endpoint returns correct data
- [ ] Performance metrics are tracked
- [ ] Drift detection works
- [ ] Retraining triggers when thresholds met
- [ ] New models are validated before deployment
- [ ] Old models are backed up
- [ ] Performance trends are tracked over time

## Troubleshooting

### Tests Fail: "Connection refused"

**Problem:** API server not running

**Solution:**
```bash
cd server
python api_server.py
```

### Tests Fail: "MongoDB connection error"

**Problem:** MongoDB not running

**Solution:**
```bash
# Windows
net start MongoDB

# Linux/Mac
sudo systemctl start mongod
```

### Tests Fail: "Adaptive learning not available"

**Problem:** Adaptive learning components not initialized

**Solution:** Check server logs for initialization errors. Ensure:
- MongoDB is running
- Database collections can be created
- No import errors in `adaptive_learning.py` or `performance_monitor.py`

### No Samples Collected

**Problem:** Samples not being added to database

**Solution:**
1. Check server logs for errors
2. Verify `adaptive_manager` is initialized in `api_server.py`
3. Check MongoDB connection
4. Verify file hash calculation works

### Feedback Not Recorded

**Problem:** Feedback endpoint returns error

**Solution:**
1. Verify file hash is correct (SHA256)
2. Check JSON format is correct
3. Verify endpoint is accessible: `GET /api/feedback` (should return 405 Method Not Allowed, not 404)

## Expected Test Results

### Unit Tests
```
✅ All AdaptiveLearningManager tests passed!
✅ All PerformanceMonitor tests passed!
✅ Integration tests passed!
✅ ALL TESTS PASSED!
```

### API Tests
```
✅ API is online
✅ File scanned successfully
✅ Feedback submitted successfully
✅ Statistics retrieved
✅ Performance metrics retrieved
✅ Performance trends retrieved
🎉 All tests passed!
```

### Statistics After Testing
```json
{
  "new_samples_total": 10+,
  "new_samples_with_labels": 5+,
  "feedback_total": 5+,
  "should_retrain": false,  // Until thresholds met
  "retraining_attempts": 0   // Until first retraining
}
```

## Next Steps After Testing

1. **Monitor the system** - Check statistics weekly
2. **Collect real feedback** - Encourage users to provide feedback
3. **Set up scheduled retraining** - Automate retraining process
4. **Review performance trends** - Track improvement over time
5. **Tune thresholds** - Adjust retraining triggers if needed

## Additional Resources

- `ADAPTIVE_LEARNING_README.md` - Complete documentation
- `ADAPTIVE_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `test/test_adaptive_learning.py` - Unit test source
- `test/test_adaptive_api.py` - API test source

Happy testing! 🚀

