# Resume Screening LLM API

A FastAPI-based REST API for scoring resumes against job postings using a lightweight Language Learning Model (LLM).

## Features

- **Single Resume Scoring**: Score individual resumes against job postings
- **Batch Scoring**: Score multiple resumes simultaneously for efficient processing
- **Model Information**: Retrieve model metadata and training history
- **Health Monitoring**: Check API and model status
- **Training Endpoint**: Trigger model training (for development)

## API Endpoints

### 1. Health Check
```
GET /health
```
Returns the health status of the API and whether a model is loaded.

### 2. Model Information
```
GET /model_info
```
Returns detailed information about the loaded model including:
- Model metadata and version
- Vocabulary size
- Training history
- Top vocabulary words

### 3. Score Single Resume
```
POST /score
```
Scores a single resume against a job posting.

**Request Body:**
```json
{
  "resume": {
    "id": "res_001",
    "name": "John Doe",
    "email": "john.doe@email.com",
    "current_title": "Senior Software Engineer",
    "years_experience": 5,
    "skills": ["Python", "FastAPI", "Machine Learning"],
    "education": [{
      "degree": "B.S.",
      "field": "Computer Science",
      "institution": "MIT"
    }],
    "experience": [{
      "title": "Senior Software Engineer",
      "company": "Tech Corp",
      "description": "Developed ML models"
    }]
  },
  "job_posting": {
    "id": "job_001",
    "title": "ML Engineer",
    "company": "AI Startup",
    "description": "Looking for ML engineer",
    "requirements": {
      "skills": ["Python", "TensorFlow", "Docker"],
      "experience": "3+ years",
      "education": "Bachelor's in CS"
    }
  }
}
```

**Response:**
```json
{
  "overall_score": 0.75,
  "matching_skills": ["Python"],
  "skills_match_percentage": 33.3,
  "experience_match": true,
  "recommendation": "strong_match",
  "scoring_metadata": {...},
  "resume_id": "res_001"
}
```

### 4. Batch Score Resumes
```
POST /batch_score
```
Scores multiple resumes against a single job posting.

**Request Body:**
```json
{
  "resumes": [
    {...}, {...}, {...}
  ],
  "job_posting": {...}
}
```

**Response:**
```json
{
  "scores": [
    {...}, {...}, {...}
  ],
  "batch_metadata": {
    "total_resumes": 3,
    "job_id": "job_001",
    "average_score": 0.65,
    "processed_at": "2023-..."
  }
}
```

### 5. Train Model
```
POST /train
```
Triggers model training in the background (development use).

**Request Body:**
```json
{
  "vocab_size": 5000,
  "data_dir": "data/synthetic"
}
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements-api.txt
```

### 2. Start the API
```bash
# Using the provided script
./scripts/start_api.sh

# Or manually
cd src/api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API
- API Base URL: `http://localhost:8000`
- Interactive Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 4. Test the API
```bash
python scripts/test_api.py
```

## Configuration

### Environment Variables
- `LOG_LEVEL`: Set logging level (default: INFO)
- `PORT`: API port (default: 8000)
- `HOST`: API host (default: 0.0.0.0)

### Model Loading
The API automatically loads the latest trained model from `models/resume_llm_latest.pkl` on startup. If no model is found, scoring endpoints will return 503 errors.

## Error Handling

The API uses standard HTTP status codes:
- `200`: Success
- `202`: Accepted (for async operations like training)
- `400`: Bad Request (invalid input)
- `503`: Service Unavailable (model not loaded)
- `500`: Internal Server Error

All errors return a JSON response with a `detail` field explaining the error.

## Development

### Running Tests
```bash
python scripts/test_api.py
```

### Training a Model
If no model exists, train one using:
```bash
python src/train.py
```

Or use the API endpoint:
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"vocab_size": 5000}'
```

## Security Considerations

For production deployment:
1. Configure CORS appropriately (currently allows all origins)
2. Add authentication/authorization
3. Use HTTPS
4. Rate limiting
5. Input validation and sanitization

## Performance

- The API uses FastAPI for high performance
- Batch scoring is optimized for processing multiple resumes
- Model is loaded once at startup for efficiency
- Supports async operations for non-blocking requests