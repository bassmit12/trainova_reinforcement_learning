import os
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import glob

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.feedback_prediction_model import FeedbackBasedPredictionModel

def load_trainer_recommendations(recommendations_dir: str) -> List[Dict[str, Any]]:
    """
    Load all trainer recommendations from JSON files in the specified directory.
    
    Args:
        recommendations_dir: Directory containing trainer recommendation files
        
    Returns:
        List of recommendation dictionaries
    """
    all_recommendations = []
    
    # Find all JSON files with recommendations
    json_files = glob.glob(os.path.join(recommendations_dir, "recommendations_*.json"))
    
    if not json_files:
        print(f"No recommendation files found in {recommendations_dir}")
        return all_recommendations
    
    # Load each file
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                
            # Extract recommendations and add the trainer info
            recommendations = data.get("recommendations", [])
            trainer_info = data.get("trainer_info", {})
            
            # Add trainer info to each recommendation
            for rec in recommendations:
                rec["trainer_name"] = trainer_info.get("name", "Unknown")
                rec["trainer_certification"] = trainer_info.get("certification", "Unknown")
                rec["trainer_experience"] = trainer_info.get("years_experience", "Unknown")
                
            all_recommendations.extend(recommendations)
            print(f"Loaded {len(recommendations)} recommendations from {os.path.basename(json_file)}")
        except Exception as e:
            print(f"Error loading {json_file}: {str(e)}")
    
    return all_recommendations

def create_pretraining_dataset(recommendations: List[Dict[str, Any]], output_file: str) -> pd.DataFrame:
    """
    Convert trainer recommendations into a dataset for model pretraining.
    
    Args:
        recommendations: List of recommendation dictionaries
        output_file: Path to save the pretraining dataset
        
    Returns:
        DataFrame containing the pretraining dataset
    """
    if not recommendations:
        print("No recommendations to process")
        return pd.DataFrame()
    
    # Create a list to store training examples
    pretraining_data = []
    
    for rec in recommendations:
        # Create a training example from the recommendation
        training_example = {
            "exercise": "Bench Press",  # All our examples are bench press for now
            "weight": rec["recommended_weight"],
            "reps": rec["target_reps"],
            "rir": rec["target_rir"],
            "success": True,  # Assume recommendations would lead to success
            "source": f"Trainer: {rec.get('trainer_name', 'Unknown')}"
        }
        
        pretraining_data.append(training_example)
        
        # Also include the validation workout if available
        if "actual_weight" in rec and "actual_reps" in rec:
            validation_example = {
                "exercise": "Bench Press",
                "weight": rec["actual_weight"],
                "reps": rec["actual_reps"],
                "rir": rec.get("actual_rir", 2),
                "success": True,  # Assume real workout data was successful
                "source": "Validation data"
            }
            
            pretraining_data.append(validation_example)
    
    # Create DataFrame
    df = pd.DataFrame(pretraining_data)
    
    # Save to CSV
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Pretraining dataset saved to {output_file}")
    
    return df

def create_feedback_entries(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert trainer recommendations into feedback entries for the model.
    
    Args:
        recommendations: List of recommendation dictionaries
        
    Returns:
        List of feedback entries
    """
    feedback_entries = []
    
    for rec in recommendations:
        # Create a feedback entry where the model prediction is imaginary,
        # but the actual weight is the trainer's recommendation
        feedback_entry = {
            "exercise": "Bench Press",
            "predicted_weight": rec["actual_weight"],  # Using validation weight as "prediction"
            "actual_weight": rec["recommended_weight"],  # Trainer recommendation as "actual"
            "score": (rec["recommended_weight"] - rec["actual_weight"]) / rec["actual_weight"],
            "reps": rec["target_reps"],
            "rir": rec["target_rir"],
            "success": True,
            "timestamp": rec.get("date", "2025-06-02"),  # Today's date as default
            "source": f"Trainer: {rec.get('trainer_name', 'Unknown')}"
        }
        
        feedback_entries.append(feedback_entry)
    
    return feedback_entries

def pretrain_model(model: FeedbackBasedPredictionModel, recommendations: List[Dict[str, Any]]) -> None:
    """
    Pretrain the prediction model using trainer recommendations.
    
    Args:
        model: The prediction model to pretrain
        recommendations: List of recommendation dictionaries
    """
    # Convert recommendations to a training dataset
    print("Creating pretraining dataset...")
    df = create_pretraining_dataset(
        recommendations, 
        os.path.join("data", "datasets", "trainer_pretraining.csv")
    )
    
    if df.empty:
        print("No pretraining data available")
        return
    
    # Create feedback entries
    print("Creating feedback entries...")
    feedback_entries = create_feedback_entries(recommendations)
    
    # Add entries to model's feedback history
    model.feedback_history.extend(feedback_entries)
    print(f"Added {len(feedback_entries)} feedback entries to model")
    
    # Train the model using the dataset
    print("Pretraining model with trainer recommendations...")
    model.fit(df)
    
    # Save the pretrained model
    model.save()  # Changed from save_model() to save()
    print("Model pretrained and saved successfully")

def main():
    """Main function to run the model pretraining system."""
    print("\nTrainova Feedback Network - Model Pretraining System")
    print("="*60)
    print("This system pretrains the feedback prediction model using")
    print("recommendations from licensed trainers.")
    print("="*60)
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    recommendations_dir = os.path.join(base_dir, "data", "datasets", "trainer_recommendations")
    
    # Check if recommendations directory exists and contains files
    if not os.path.exists(recommendations_dir) or not os.listdir(recommendations_dir):
        print("No trainer recommendations found. Please run trainer_recommendation.py first.")
        return
    
    # Load trainer recommendations
    print("Loading trainer recommendations...")
    recommendations = load_trainer_recommendations(recommendations_dir)
    
    if not recommendations:
        print("No valid recommendations found. Please run trainer_recommendation.py first.")
        return
    
    # Initialize the model
    print("Initializing prediction model...")
    model = FeedbackBasedPredictionModel()
    
    # Pretrain the model
    pretrain_model(model, recommendations)
    
    # Exit
    print("\nModel pretraining completed!")
    print("You can now run the regular training script or server with the pretrained model.")

if __name__ == "__main__":
    main()