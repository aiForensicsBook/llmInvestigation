import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.resume_llm import ResumeScreeningLLM
from src.train import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Resume Screening LLM API",
    description="API for scoring resumes against job postings using a lightweight LLM",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[ResumeScreeningLLM] = None

# Pydantic models for request/response validation
class Education(BaseModel):
    degree: str
    field: str
    institution: str
    graduation_year: Optional[int] = None

class Experience(BaseModel):
    title: str
    company: str
    description: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class Resume(BaseModel):
    id: Optional[str] = Field(default="unknown", description="Resume ID")
    name: str
    email: Optional[str] = None
    current_title: str
    years_experience: int = Field(ge=0, description="Years of experience")
    skills: List[str]
    education: List[Education]
    experience: List[Experience]
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "res_001",
                "name": "John Doe",
                "email": "john.doe@email.com",
                "current_title": "Senior Software Engineer",
                "years_experience": 5,
                "skills": ["Python", "FastAPI", "Machine Learning", "Docker"],
                "education": [{
                    "degree": "B.S.",
                    "field": "Computer Science",
                    "institution": "MIT",
                    "graduation_year": 2018
                }],
                "experience": [{
                    "title": "Senior Software Engineer",
                    "company": "Tech Corp",
                    "description": "Developed ML models for production systems",
                    "start_date": "2020-01",
                    "end_date": "present"
                }]
            }
        }

class JobRequirements(BaseModel):
    skills: List[str]
    experience: str
    education: Optional[str] = None

class JobPosting(BaseModel):
    id: Optional[str] = Field(default="unknown", description="Job posting ID")
    title: str
    company: str
    description: str
    requirements: JobRequirements
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "job_001",
                "title": "Machine Learning Engineer",
                "company": "AI Startup",
                "description": "Looking for an ML engineer to build scalable AI systems",
                "requirements": {
                    "skills": ["Python", "TensorFlow", "Docker", "AWS"],
                    "experience": "3+ years of ML experience",
                    "education": "Bachelor's in Computer Science or related field"
                }
            }
        }

class ScoreRequest(BaseModel):
    resume: Resume
    job_posting: JobPosting

class BatchScoreRequest(BaseModel):
    resumes: List[Resume]
    job_posting: JobPosting

class ScoreResponse(BaseModel):
    overall_score: float = Field(ge=0, le=1, description="Overall match score between 0 and 1")
    matching_skills: List[str]
    skills_match_percentage: float
    experience_match: bool
    recommendation: str = Field(description="One of: strong_match, possible_match, weak_match")
    scoring_metadata: Dict
    resume_id: Optional[str] = None

class BatchScoreResponse(BaseModel):
    scores: List[ScoreResponse]
    batch_metadata: Dict

class ModelInfo(BaseModel):
    model_metadata: Dict
    vocabulary_size: int
    trained: bool
    training_history: List[Dict]
    top_vocabulary_words: List[str]
    model_file: Optional[str] = None

class TrainRequest(BaseModel):
    vocab_size: int = Field(default=5000, ge=100, le=50000)
    data_dir: str = Field(default="data/synthetic")

class TrainResponse(BaseModel):
    status: str
    message: str
    model_path: Optional[str] = None
    training_stats: Optional[Dict] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    model_version: Optional[str] = None


def load_latest_model():
    """Load the latest trained model."""
    global model
    
    model_dir = "models"
    latest_model_path = os.path.join(model_dir, "resume_llm_latest.pkl")
    
    if not os.path.exists(latest_model_path):
        # Try to find any model file
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            if model_files:
                # Sort by modification time and get the most recent
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                latest_model_path = os.path.join(model_dir, model_files[0])
            else:
                logger.warning("No model files found in models directory")
                return False
        else:
            logger.warning("Models directory does not exist")
            return False
    
    try:
        model = ResumeScreeningLLM()
        model.load_model(latest_model_path)
        logger.info(f"Loaded model from: {latest_model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting Resume Screening LLM API...")
    
    if not load_latest_model():
        logger.warning("No trained model found. Please train a model first using the /train endpoint.")
    else:
        logger.info("Model loaded successfully")


@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Root endpoint."""
    return {
        "message": "Resume Screening LLM API",
        "version": "1.0.0",
        "endpoints": [
            "/docs - API documentation",
            "/health - Health check",
            "/model_info - Get model information",
            "/score - Score a single resume",
            "/batch_score - Score multiple resumes",
            "/train - Train the model"
        ]
    }


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None and model.trained,
        model_version=model.model_metadata.get("version") if model else None
    )


@app.get("/model_info", response_model=ModelInfo, status_code=status.HTTP_200_OK)
async def get_model_info():
    """Get model information and metadata."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train a model first."
        )
    
    info = model.get_model_info()
    
    # Find the model file path
    model_file = None
    model_dir = "models"
    if os.path.exists(model_dir):
        latest_path = os.path.join(model_dir, "resume_llm_latest.pkl")
        if os.path.exists(latest_path):
            model_file = latest_path
    
    return ModelInfo(
        model_metadata=info["model_metadata"],
        vocabulary_size=info["vocabulary_size"],
        trained=info["trained"],
        training_history=info["training_history"],
        top_vocabulary_words=info["top_vocabulary_words"],
        model_file=model_file
    )


@app.post("/score", response_model=ScoreResponse, status_code=status.HTTP_200_OK)
async def score_resume(request: ScoreRequest):
    """Score a single resume against a job posting."""
    if model is None or not model.trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded or not trained. Please train a model first."
        )
    
    try:
        # Convert Pydantic models to dicts
        resume_dict = request.resume.model_dump()
        job_dict = request.job_posting.model_dump()
        
        # Score the resume
        logger.info(f"Scoring resume {resume_dict.get('id')} against job {job_dict.get('id')}")
        result = model.score_resume(resume_dict, job_dict)
        
        # Add resume ID to result
        result["resume_id"] = resume_dict.get("id", "unknown")
        
        return ScoreResponse(**result)
        
    except Exception as e:
        logger.error(f"Error scoring resume: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error scoring resume: {str(e)}"
        )


@app.post("/batch_score", response_model=BatchScoreResponse, status_code=status.HTTP_200_OK)
async def batch_score_resumes(request: BatchScoreRequest):
    """Score multiple resumes against a job posting."""
    if model is None or not model.trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded or not trained. Please train a model first."
        )
    
    try:
        # Convert Pydantic models to dicts
        resumes_dict = [resume.model_dump() for resume in request.resumes]
        job_dict = request.job_posting.model_dump()
        
        # Score all resumes
        logger.info(f"Batch scoring {len(resumes_dict)} resumes against job {job_dict.get('id')}")
        results = model.batch_score(resumes_dict, job_dict)
        
        # Convert results to ScoreResponse objects
        scores = [ScoreResponse(**result) for result in results]
        
        batch_metadata = {
            "total_resumes": len(resumes_dict),
            "job_id": job_dict.get("id", "unknown"),
            "job_title": job_dict.get("title", ""),
            "processed_at": datetime.now().isoformat(),
            "average_score": sum(s.overall_score for s in scores) / len(scores) if scores else 0
        }
        
        return BatchScoreResponse(
            scores=scores,
            batch_metadata=batch_metadata
        )
        
    except Exception as e:
        logger.error(f"Error in batch scoring: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch scoring: {str(e)}"
        )


@app.post("/train", response_model=TrainResponse, status_code=status.HTTP_202_ACCEPTED)
async def train_model_endpoint(
    request: TrainRequest,
    background_tasks: BackgroundTasks
):
    """Trigger model training (for development use)."""
    
    def train_in_background(vocab_size: int, data_dir: str):
        """Background training task."""
        global model
        
        try:
            logger.info(f"Starting model training with vocab_size={vocab_size}")
            
            # Train the model
            trained_model = train_model(vocab_size=vocab_size, data_dir=data_dir)
            
            # Update the global model
            model = trained_model
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
    
    # Check if training data exists
    data_path = Path(request.data_dir)
    if not data_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Training data directory not found: {request.data_dir}"
        )
    
    # Add training task to background
    background_tasks.add_task(
        train_in_background,
        vocab_size=request.vocab_size,
        data_dir=request.data_dir
    )
    
    return TrainResponse(
        status="training_started",
        message=f"Model training started in background with vocab_size={request.vocab_size}",
        model_path=None,
        training_stats=None
    )


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )