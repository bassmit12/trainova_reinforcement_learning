from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import sys
import pandas as pd
from datetime import datetime

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.feedback_prediction_model import FeedbackBasedPredictionModel
from data.visualization import plot_prediction_vs_actual, plot_score_distribution, plot_improvement_over_time

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

class VisualizationResponse(BaseModel):
    """Response model for visualization endpoints"""
    image_data: str
    exercise: Optional[str] = None
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

@app.get("/visualize/prediction", response_model=VisualizationResponse)
async def visualize_prediction(
    exercise: Optional[str] = None,
    last_n: Optional[int] = None,
    model: FeedbackBasedPredictionModel = Depends(get_model)
):
    """
    Generate a visualization comparing predicted weights to actual weights over time.
    
    Args:
        exercise: Optional filter for a specific exercise
        last_n: Optional parameter to limit to the last N predictions
    
    Returns:
        JSON response with base64-encoded image and metadata
    """
    try:
        # Generate the visualization
        _, img_str = plot_prediction_vs_actual(model.feedback_history, exercise, last_n)
        
        # Prepare response
        return {
            "image_data": img_str,
            "exercise": exercise,
            "message": "Prediction vs actual weight comparison visualization"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

@app.get("/visualize/scores", response_model=VisualizationResponse)
async def visualize_scores(
    exercise: Optional[str] = None,
    model: FeedbackBasedPredictionModel = Depends(get_model)
):
    """
    Generate a visualization showing the distribution of prediction scores.
    
    Args:
        exercise: Optional filter for a specific exercise
    
    Returns:
        JSON response with base64-encoded image and metadata
    """
    try:
        # Generate the visualization
        _, img_str = plot_score_distribution(model.feedback_history, exercise)
        
        # Prepare response
        return {
            "image_data": img_str,
            "exercise": exercise,
            "message": "Score distribution visualization"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

@app.get("/visualize/improvement", response_model=VisualizationResponse)
async def visualize_improvement(
    exercise: Optional[str] = None,
    window_size: int = 5,
    model: FeedbackBasedPredictionModel = Depends(get_model)
):
    """
    Generate a visualization showing prediction error improvement over time.
    
    Args:
        exercise: Optional filter for a specific exercise
        window_size: Size of the moving average window (default: 5)
    
    Returns:
        JSON response with base64-encoded image and metadata
    """
    try:
        # Generate the visualization
        _, img_str = plot_improvement_over_time(model.feedback_history, exercise, window_size)
        
        # Prepare response
        return {
            "image_data": img_str,
            "exercise": exercise,
            "message": "Prediction error improvement visualization"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")

@app.get("/visualize/html", response_class=HTMLResponse)
async def visualize_html(
    exercise: Optional[str] = None,
    model: FeedbackBasedPredictionModel = Depends(get_model)
):
    """
    Generate an HTML page with all visualizations for easy viewing.
    
    Args:
        exercise: Optional filter for a specific exercise
    
    Returns:
        HTML page with embedded visualizations
    """
    try:
        # Generate all visualizations
        _, img_pred = plot_prediction_vs_actual(model.feedback_history, exercise)
        _, img_score = plot_score_distribution(model.feedback_history, exercise)
        _, img_imp = plot_improvement_over_time(model.feedback_history, exercise)
        
        # Create HTML response
        exercise_title = f"for {exercise}" if exercise else "for all exercises"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Model Visualizations {exercise_title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .viz-container {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Prediction Model Visualizations {exercise_title}</h1>
            
            <div class="viz-container">
                <h2>Prediction vs Actual Weight</h2>
                <img src="data:image/png;base64,{img_pred}" alt="Prediction vs Actual">
            </div>
            
            <div class="viz-container">
                <h2>Score Distribution</h2>
                <img src="data:image/png;base64,{img_score}" alt="Score Distribution">
            </div>
            
            <div class="viz-container">
                <h2>Prediction Error Improvement</h2>
                <img src="data:image/png;base64,{img_imp}" alt="Error Improvement">
            </div>
        </body>
        </html>
        """
        
        return html_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")