# Quick Test Guide

## Fastest Way to Test

### 1. Start Everything

```bash
# Terminal 1: Start MongoDB (if not running)
# Windows: net start MongoDB
# Linux/Mac: sudo systemctl start mongod

# Terminal 2: Start API Server
cd server
python api_server.py
```

### 2. Run Tests

```bash
# Terminal 3: Run unit tests
python test/test_adaptive_learning.py

# Terminal 4: Run API tests
python test/test_adaptive_api.py
```

### 3. Manual API Test (cURL)

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Scan a file (replace with actual file)
curl -X POST "http://localhost:8000/scan" -F "file=@test.exe"

# 3. Get statistics
curl http://localhost:8000/api/adaptive/statistics

# 4. Submit feedback (replace file_hash from scan response)
curl -X POST "http://localhost:8000/api/feedback" \
  -H "Content-Type: application/json" \
  -d '{"file_hash": "abc123...", "prediction_was_correct": true, "actual_label": "benign"}'

# 5. Get performance
curl "http://localhost:8000/api/adaptive/performance?days=30"
```

### 4. Check Results

```bash
# Check if samples collected
curl http://localhost:8000/api/adaptive/statistics | python -m json.tool
```

## Expected Results

✅ All tests should pass
✅ Statistics should show samples collected
✅ Feedback should be recorded
✅ Performance metrics should be available

See `TESTING_GUIDE.md` for detailed instructions.

