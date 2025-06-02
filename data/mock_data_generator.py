import random
import json
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any, Optional

def generate_mock_user_profile(user_id: int) -> Dict[str, Any]:
    """
    Generate a mock user profile with realistic attributes for weight training.
    
    Args:
        user_id: Unique identifier for the user
        
    Returns:
        Dictionary containing user profile data
    """
    # Generate age between 18-65
    age = random.randint(18, 65)
    
    # Generate weight in kg (50-120kg)
    weight_kg = round(random.uniform(50, 120), 1)
    
    # Generate height in cm (155-200cm)
    height_cm = round(random.uniform(155, 200), 1)
    
    # Generate training experience in years (0-20 years)
    experience_years = round(random.uniform(0, 20), 1)
    
    # Randomly assign gender
    gender = random.choice(["male", "female"])
    
    # Calculate BMI
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
    
    # Generate starting bench press strength based on body weight and experience
    # Using a simplified formula that accounts for gender differences in average strength
    strength_multiplier = 0.8 if gender == "female" else 1.2
    experience_factor = 0.6 + (0.4 * min(experience_years / 10, 1))  # Experience caps at 10 years for this factor
    
    # Calculate starting bench press 1RM (one-rep max)
    # This is a simplified model - real strength would have more variables
    starting_bench_1rm = round(weight_kg * strength_multiplier * experience_factor, 1)
    
    # Adjust for age - peak strength around age 30, declining after
    age_factor = 1 - max(0, abs(age - 30) / 100)
    starting_bench_1rm = round(starting_bench_1rm * age_factor, 1)
    
    # Generate a user profile
    user_profile = {
        "user_id": user_id,
        "age": age,
        "gender": gender,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "bmi": bmi,
        "experience_years": experience_years,
        "starting_bench_1rm_kg": starting_bench_1rm
    }
    
    return user_profile

def generate_mock_bench_press_history(
    user_profile: Dict[str, Any], 
    num_workouts: int = 10,
    start_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Generate a realistic bench press workout history for a mock user.
    
    Args:
        user_profile: Dictionary containing user profile data
        num_workouts: Number of workouts to generate
        start_date: Starting date for the workout history
        
    Returns:
        List of dictionaries containing workout data
    """
    if start_date is None:
        # Default to 3 months ago
        start_date = datetime.now() - timedelta(days=90)
    
    # Extract user attributes
    starting_1rm = user_profile["starting_bench_1rm_kg"]
    experience_years = user_profile["experience_years"]
    
    # Calculate progression rate based on experience (beginners progress faster)
    progression_rate = 0.01 * (1 + max(0, 3 - experience_years) / 3)
    
    # Generate random workout days (between 2-7 days apart)
    workout_dates = [start_date]
    for _ in range(num_workouts - 1):
        days_between = random.randint(2, 7)
        next_date = workout_dates[-1] + timedelta(days=days_between)
        workout_dates.append(next_date)
    
    # Generate workout data
    workouts = []
    current_1rm = starting_1rm
    
    for i, workout_date in enumerate(workout_dates):
        # Simulate progress over time with some random variation
        if i > 0:
            # Add some randomness to progression
            progress_factor = random.uniform(0.9, 1.1)
            current_1rm = current_1rm * (1 + progression_rate * progress_factor)
        
        # Choose a rep range for this workout (typically 1-12 for bench press)
        reps = random.choice([1, 3, 5, 8, 10, 12])
        
        # Calculate working weight based on rep range (percentage of 1RM)
        # Using Brzycki formula approximation
        rep_factor = 1.0278 - 0.0278 * reps
        working_weight = round(current_1rm * rep_factor, 1)
        
        # Add a small random variation to make data more realistic
        working_weight = round(working_weight * random.uniform(0.95, 1.05), 1)
        
        # Generate Reps In Reserve (RIR) - how many more reps could have been done
        rir = random.choice([0, 1, 2, 3])
        
        # Determine if workout was successful (most are, but some aren't)
        success = random.random() < 0.9  # 90% success rate
        
        # Create workout entry
        workout = {
            "user_id": user_profile["user_id"],
            "exercise": "Bench Press",
            "date": workout_date.strftime("%Y-%m-%d"),
            "weight": working_weight,
            "reps": reps,
            "rir": rir,
            "success": success,
            "notes": ""
        }
        
        workouts.append(workout)
    
    return workouts

def generate_mock_dataset(num_users: int = 20, workouts_per_user: int = 10) -> Dict[str, Any]:
    """
    Generate a complete mock dataset with multiple users and their bench press histories.
    
    Args:
        num_users: Number of mock users to generate
        workouts_per_user: Number of workouts per user
        
    Returns:
        Dictionary containing user profiles and workout histories
    """
    dataset = {
        "users": [],
        "workouts": []
    }
    
    for user_id in range(1, num_users + 1):
        # Generate user profile
        user_profile = generate_mock_user_profile(user_id)
        dataset["users"].append(user_profile)
        
        # Generate workout history
        workouts = generate_mock_bench_press_history(
            user_profile, 
            num_workouts=workouts_per_user
        )
        dataset["workouts"].extend(workouts)
    
    return dataset

def save_mock_dataset(dataset: Dict[str, Any], output_dir: str) -> None:
    """
    Save the mock dataset to disk in multiple formats.
    
    Args:
        dataset: Dictionary containing mock data
        output_dir: Directory to save the data to
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    with open(os.path.join(output_dir, "mock_bench_press_data.json"), "w") as f:
        json.dump(dataset, f, indent=2)
    
    # Save users as CSV
    users_df = pd.DataFrame(dataset["users"])
    users_df.to_csv(os.path.join(output_dir, "mock_users.csv"), index=False)
    
    # Save workouts as CSV
    workouts_df = pd.DataFrame(dataset["workouts"])
    workouts_df.to_csv(os.path.join(output_dir, "mock_workouts.csv"), index=False)
    
    print(f"Mock dataset saved to {output_dir}")

def prepare_trainer_recommendation_cases(
    dataset: Dict[str, Any], 
    num_cases: int = 10,
    case_dir: str = None
) -> List[Dict[str, Any]]:
    """
    Prepare cases for trainer recommendations by selecting a user's
    workout history and asking for the next workout weight.
    
    Args:
        dataset: Dictionary containing mock data
        num_cases: Number of recommendation cases to generate
        case_dir: Directory to save the cases to (optional)
        
    Returns:
        List of trainer recommendation cases
    """
    # Group workouts by user
    user_workouts = {}
    for workout in dataset["workouts"]:
        user_id = workout["user_id"]
        if user_id not in user_workouts:
            user_workouts[user_id] = []
        user_workouts[user_id].append(workout)
    
    # Sort each user's workouts by date
    for user_id in user_workouts:
        user_workouts[user_id].sort(key=lambda w: w["date"])
    
    # Only include users with enough workouts
    eligible_users = [uid for uid, workouts in user_workouts.items() if len(workouts) >= 5]
    
    if not eligible_users:
        raise ValueError("No users with enough workout history found in the dataset")
    
    # Select random users for cases
    selected_users = random.sample(eligible_users, min(num_cases, len(eligible_users)))
    
    # Generate recommendation cases
    recommendation_cases = []
    
    for i, user_id in enumerate(selected_users):
        # Get user profile
        user_profile = next(u for u in dataset["users"] if u["user_id"] == user_id)
        
        # Get user's workout history
        workouts = user_workouts[user_id]
        
        # Use all but the last workout for history
        history = workouts[:-1]
        
        # The last workout will be for validation
        validation_workout = workouts[-1]
        
        # Create case
        case = {
            "case_id": i + 1,
            "user_profile": user_profile,
            "workout_history": history,
            "validation_workout": validation_workout,
            "trainer_recommendation": None  # To be filled by trainer
        }
        
        recommendation_cases.append(case)
    
    # Save cases if directory provided
    if case_dir:
        os.makedirs(case_dir, exist_ok=True)
        
        # Save all cases in one file
        with open(os.path.join(case_dir, "trainer_recommendation_cases.json"), "w") as f:
            json.dump(recommendation_cases, f, indent=2)
        
        # Save individual case files
        for case in recommendation_cases:
            case_file = os.path.join(case_dir, f"case_{case['case_id']}.json")
            with open(case_file, "w") as f:
                json.dump(case, f, indent=2)
        
        print(f"Trainer recommendation cases saved to {case_dir}")
    
    return recommendation_cases

if __name__ == "__main__":
    # Generate mock dataset
    print("Generating mock bench press dataset...")
    mock_dataset = generate_mock_dataset(num_users=30, workouts_per_user=15)
    
    # Save the dataset
    save_mock_dataset(mock_dataset, os.path.join("data", "datasets"))
    
    # Prepare trainer recommendation cases
    print("Preparing trainer recommendation cases...")
    recommendation_cases = prepare_trainer_recommendation_cases(
        mock_dataset, 
        num_cases=15,
        case_dir=os.path.join("data", "datasets", "trainer_cases")
    )
    
    print(f"Generated {len(recommendation_cases)} trainer recommendation cases")