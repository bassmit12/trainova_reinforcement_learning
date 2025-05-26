from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import sys
import pandas as pd
from datetime import datetime

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.feedback_prediction_model import FeedbackBasedPredictionModel

app = FastAPI(
    title="Workout Weight Prediction API",
    description="API for predicting workout weights using a feedback-based system",
    version="1.0.0"
)

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
prediction_model = FeedbackBasedPredictionModel(model_dir=model_dir)

# Pydantic models for request/response validation
class WorkoutData(BaseModel):
    exercise: str
    weight: float
    reps: int
    date: Optional[str] = None
    success: Optional[bool] = True
    rir: Optional[int] = None  # Reps In Reserve - how many more reps could have been done
    
class PredictionRequest(BaseModel):
    exercise: str
    previous_workouts: List[WorkoutData]
    
class PredictionResponse(BaseModel):
    weight: float
    confidence: float
    message: str
    suggested_reps: Optional[List[int]] = None
    
class FeedbackRequest(BaseModel):
    exercise: str
    predicted_weight: float
    actual_weight: float
    success: bool = True
    reps: Optional[int] = None
    rir: Optional[int] = None  # Adding RIR to feedback
    
class FeedbackResponse(BaseModel):
    feedback_recorded: bool
    score: float
    message: str
    
class StatsResponse(BaseModel):
    total_predictions: int
    accuracy: float
    success_rate: Optional[float] = None
    message: str

# Helper function to get the prediction model
def get_model():
    return prediction_model

@app.get("/")
async def root():
    return {"message": "Welcome to the Workout Weight Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_weight(
    request: PredictionRequest,
    model: FeedbackBasedPredictionModel = Depends(get_model)
):
    """
    Predict the weight for the next workout based on previous workout data.
    """
    if not request.previous_workouts:
        raise HTTPException(status_code=400, detail="No previous workout data provided")
    
    # Convert Pydantic models to dictionaries
    previous_workouts = [workout.dict() for workout in request.previous_workouts]
    
    # Make prediction
    prediction = model.predict(request.exercise, previous_workouts)
    
    return prediction

@app.post("/feedback", response_model=FeedbackResponse)
async def provide_feedback(
    request: FeedbackRequest,
    model: FeedbackBasedPredictionModel = Depends(get_model)
):
    """
    Provide feedback on a prediction to improve future predictions.
    """
    feedback_result = model.provide_feedback(
        exercise=request.exercise,
        predicted_weight=request.predicted_weight,
        actual_weight=request.actual_weight,
        success=request.success,
        reps=request.reps,
        rir=request.rir
    )
    
    return feedback_result

@app.get("/stats", response_model=StatsResponse)
async def get_stats(
    exercise: Optional[str] = None,
    model: FeedbackBasedPredictionModel = Depends(get_model)
):
    """
    Get performance statistics about the prediction model.
    """
    stats = model.get_performance_stats(exercise)
    return stats

@app.post("/train")
async def train_model(
    model: FeedbackBasedPredictionModel = Depends(get_model)
):
    """
    Retrain the model using feedback history and available workout data.
    No longer using CSV files for data input.
    """
    try:
        # Use the existing feedback history for training
        if not model.feedback_history:
            return {"message": "No feedback history available for training. Add workout data through the API first."}
            
        # Convert feedback history to a dataframe for model training
        workout_data = []
        
        for feedback in model.feedback_history:
            # Extract relevant workout data from feedback entries
            if all(k in feedback for k in ['exercise', 'actual_weight']):
                workout_entry = {
                    'exercise': feedback['exercise'],
                    'weight': feedback['actual_weight'],
                    'reps': feedback.get('reps', 8),  # Default to 8 if not specified
                    'date': datetime.now().strftime('%Y-%m-%d'),  # Use current date if not available
                    'success': feedback.get('success', True)
                }
                workout_data.append(workout_entry)
        
        # Create a DataFrame
        if not workout_data:
            return {"message": "No valid workout data found in feedback history"}
            
        df = pd.DataFrame(workout_data)
        
        # Retrain the model
        model.fit(df)
        
        return {"message": f"Model trained successfully on {len(df)} records from feedback history"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.get("/compare")
async def compare_with_neural_network():
    """
    Compare the feedback-based model with the neural network model
    """
    return {
        "comparison": [
            {
                "feature": "Learning approach", 
                "feedback_model": "Uses past predictions and user feedback to improve",
                "neural_network": "Uses complex neural patterns to learn from data"
            },
            {
                "feature": "Adaptation", 
                "feedback_model": "Continually adapts with each workout feedback",
                "neural_network": "Requires full retraining to adapt to new patterns"
            },
            {
                "feature": "Explainability", 
                "feedback_model": "Decision factors are visible and understandable",
                "neural_network": "Functions as a black box, decisions harder to interpret"
            },
            {
                "feature": "Data requirements", 
                "feedback_model": "Works well with smaller datasets and individual users",
                "neural_network": "Requires large datasets for optimal performance"
            },
            {
                "feature": "Personalization", 
                "feedback_model": "Highly personalized to individual workout patterns",
                "neural_network": "More generalized based on population patterns"
            }
        ]
    }