import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add parent directory to path to import from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.feedback_prediction_model import FeedbackBasedPredictionModel

def prepare_data_for_model(df):
    """
    Prepare workout data for the feedback-based model
    
    Args:
        df: DataFrame with raw workout data
        
    Returns:
        Processed DataFrame ready for the model
    """
    if df.empty:
        return df
        
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Fill missing values
    if 'weight' in processed_df.columns:
        processed_df['weight'] = processed_df['weight'].fillna(0)
    
    if 'reps' in processed_df.columns:
        processed_df['reps'] = processed_df['reps'].fillna(0)
    
    # Ensure numeric columns are numeric
    numeric_cols = ['weight', 'reps']
    for col in numeric_cols:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    
    # Add derived features
    if 'weight' in processed_df.columns and 'reps' in processed_df.columns:
        processed_df['volume'] = processed_df['weight'] * processed_df['reps']
    
    # Add date-based features if date exists
    if 'date' in processed_df.columns and isinstance(processed_df['date'].iloc[0], pd.Timestamp):
        processed_df['day_of_week'] = processed_df['date'].dt.dayofweek
        
        # Calculate days since previous workout for each exercise
        def calc_days_since_prev(group):
            group = group.sort_values('date')
            group['days_since_prev'] = group['date'].diff().dt.days
            return group
            
        if len(processed_df) > 1:
            processed_df = processed_df.groupby('exercise').apply(calc_days_since_prev)
            processed_df['days_since_prev'] = processed_df['days_since_prev'].fillna(0)
    
    return processed_df

def create_workout_dataframe_from_feedback(feedback_history):
    """
    Create a DataFrame from feedback history for model training
    
    Args:
        feedback_history: List of feedback entries from the model
        
    Returns:
        DataFrame containing workout data extracted from feedback
    """
    if not feedback_history:
        return pd.DataFrame()
        
    workout_data = []
    
    for feedback in feedback_history:
        # Extract relevant workout data from feedback entries
        if all(k in feedback for k in ['exercise', 'actual_weight']):
            workout_entry = {
                'exercise': feedback['exercise'],
                'weight': feedback['actual_weight'],
                'reps': feedback.get('reps', 8),  # Default to 8 if not specified
                'date': datetime.now().strftime('%Y-%m-%d'),  # Use current date if not available
                'success': feedback.get('success', True),
                'rir': feedback.get('rir', None)  # Include RIR data if available
            }
            workout_data.append(workout_entry)
    
    # Create a DataFrame
    if not workout_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(workout_data)
    
    # Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

def train_feedback_model(feedback_history, save_dir='../models'):
    """
    Train the feedback-based prediction model using feedback history
    
    Args:
        feedback_history: List of feedback entries from the model
        save_dir: Directory to save the model to
        
    Returns:
        Trained model instance
    """
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create DataFrame from feedback history
    df = create_workout_dataframe_from_feedback(feedback_history)
    if df.empty:
        print("No valid data found in feedback history. Cannot train model.")
        return None
        
    # Prepare the data
    processed_df = prepare_data_for_model(df)
    
    # Initialize and train the model
    model = FeedbackBasedPredictionModel(model_dir=save_dir)
    model.fit(processed_df)
    
    print(f"Feedback-based model trained on {len(processed_df)} workout records")
    
    return model

if __name__ == "__main__":
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Define paths
    models_dir = os.path.join(project_dir, "models")
    
    # Initialize model to get feedback history
    model = FeedbackBasedPredictionModel(model_dir=models_dir)
    
    if model.feedback_history:
        # Train model with existing feedback history
        trained_model = train_feedback_model(model.feedback_history, models_dir)
        
        if trained_model:
            print("Model training complete. Saved to models directory.")
            
            # Test a prediction
            test_data = [
                {'exercise': 'Bench Press', 'weight': 100, 'reps': 8, 'date': '2023-01-01'},
                {'exercise': 'Bench Press', 'weight': 105, 'reps': 8, 'date': '2023-01-08'},
                {'exercise': 'Bench Press', 'weight': 110, 'reps': 7, 'date': '2023-01-15'}
            ]
            prediction = trained_model.predict('Bench Press', test_data)
            print(f"\nTest Prediction for Bench Press:")
            print(f"Predicted weight: {prediction['weight']} lbs")
            print(f"Confidence: {prediction['confidence']}")
            print(f"Message: {prediction['message']}")
    else:
        print("No feedback history available for training. Add workout data through the API first.")