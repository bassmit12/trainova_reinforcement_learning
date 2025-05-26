import os
import sys
import json
import numpy as np
from datetime import datetime
from models.feedback_prediction_model import FeedbackBasedPredictionModel

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_exercise_history(exercise):
    """Get previous workouts for an exercise from the feedback history"""
    workouts = []
    for entry in model.feedback_history:
        if entry.get('exercise') == exercise:
            workout = {
                'exercise': entry.get('exercise'),
                'weight': entry.get('actual_weight'),
                'reps': entry.get('reps', 8),
                'rir': entry.get('rir', '-'),  # Use '-' as default for missing RIR values
                'date': entry.get('date', datetime.now().strftime('%Y-%m-%d')),
            }
            workouts.append(workout)
    return workouts

def print_previous_workouts(exercise):
    """Print previous workouts for an exercise"""
    workouts = get_exercise_history(exercise)
    if not workouts:
        print(f"No previous workouts found for {exercise}")
        return
    print(f"\nPrevious workouts for {exercise}:")
    print("-" * 50)
    print(f"{'Weight':^10} | {'Reps':^8} | {'RIR':^6}")
    print("-" * 50)
    for workout in workouts:
        # Handle possibly missing values with safe defaults
        weight = workout.get('weight', '-')
        reps = workout.get('reps', '-')
        rir = workout.get('rir', '-')
        
        # Convert to strings for formatting
        weight_str = str(weight) if weight is not None else '-'
        reps_str = str(reps) if reps is not None else '-'
        rir_str = str(rir) if rir is not None else '-'
        
        print(f"{weight_str:^10} | {reps_str:^8} | {rir_str:^6}")
    print("-" * 50)

# Initialize model
print("Initializing prediction model...")
model = FeedbackBasedPredictionModel()

clear_screen()

print("Trainova Feedback Network - Interactive Training Mode")
print("====================================================")
print("This mode allows you to train the model by providing feedback on predictions.")
print("You can add new exercises or continue training existing ones.")

# Interactive training loop
while True:
    # Show unique exercises in feedback history
    exercises = set()
    for entry in model.feedback_history:
        if 'exercise' in entry:
            exercises.add(entry['exercise'])

    if exercises:
        print("\nExisting exercises:")
        for i, ex in enumerate(sorted(exercises), 1):
            print(f"{i}. {ex}")

    print("\nOptions:")
    print("1. Train with existing exercise")
    print("2. Add a new exercise")
    print("3. Exit")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == '1':
        if not exercises:
            print("No existing exercises found. Please add a new exercise first.")
            continue

        exercise_choice = input("Enter exercise number or name: ").strip()
        try:
            # Check if input is a number
            idx = int(exercise_choice) - 1
            if 0 <= idx < len(exercises):
                exercise = sorted(exercises)[idx]
            else:
                print("Invalid exercise number. Please try again.")
                continue
        except ValueError:
            # Input is a name
            exercise = exercise_choice
            if exercise not in exercises:
                print(f"Exercise '{exercise}' not found. Please try again.")
                continue

    elif choice == '2':
        exercise = input("Enter new exercise name: ").strip()
        if not exercise:
            print("Exercise name cannot be empty. Please try again.")
            continue
    elif choice == '3':
        print("\nExiting. Thank you for training the model!")
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
        continue

    clear_screen()
    print(f"Training with exercise: {exercise}\n")

    # Print previous workouts for this exercise
    print_previous_workouts(exercise)

    # Get workout parameters
    try:
        reps = int(input("\nEnter target reps for this workout: ").strip() or "8")
    except ValueError:
        print("Invalid input. Using default value of 8 reps.")
        reps = 8

    # Get prediction from model
    previous_workouts = get_exercise_history(exercise)
    prediction = model.predict(exercise, previous_workouts)

    print(f"\nMODEL PREDICTION:")
    print(f"  Weight: {prediction['weight']} kg")
    print(f"  Confidence: {prediction['confidence']}")
    if 'suggested_reps' in prediction:
        suggested_reps = prediction['suggested_reps']
        print(f"  Suggested rep scheme: {suggested_reps}")
    if 'message' in prediction:
        print(f"  {prediction['message']}")
    if 'scaling_factor' in prediction:
        print(f"  Responsiveness scaling: {prediction['scaling_factor']}x")
    if 'limited_data_mode' in prediction and prediction['limited_data_mode']:
        print(f"  LIMITED DATA MODE: Model is more responsive to feedback")

    # Get actual weight used
    try:
        actual_weight = float(input("\nEnter the ACTUAL weight used (or should be used): ").strip())
    except ValueError:
        print("Invalid input. Using predicted weight.")
        actual_weight = prediction['weight']

    # Get RIR
    try:
        rir = int(input("Enter Reps In Reserve (RIR) - how many more reps you could have done: ").strip() or "2")
    except ValueError:
        print("Invalid input. Using default value of 2 RIR.")
        rir = 2

    # Get workout success
    success_input = input("Was the workout successful? (y/n, default: y): ").strip().lower() or "y"
    success = success_input == "y"

    # Provide feedback to model
    feedback = model.provide_feedback(
        exercise=exercise,
        predicted_weight=prediction['weight'],
        actual_weight=actual_weight,
        success=success,
        reps=reps,
        rir=rir
    )

    print("\nFeedback recorded successfully!")
    print(f"Score: {feedback['score']}")
    print(f"Message: {feedback['message']}")

    input("\nPress Enter to continue...")
    clear_screen()