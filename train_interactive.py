import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from models.feedback_prediction_model import FeedbackBasedPredictionModel
from data.visualization import plot_prediction_vs_actual, plot_score_distribution, plot_improvement_over_time

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

def parse_weight_and_reps(input_str):
    """
    Parse weight and reps from a string input like "70x5" or "70 x 5".
    
    Args:
        input_str: String containing weight and reps
        
    Returns:
        Tuple of (weight, reps)
    """
    # Handle different possible formats
    input_str = input_str.strip().lower()
    
    if 'x' in input_str:
        # Split by 'x' and clean up spaces
        parts = [part.strip() for part in input_str.split('x')]
        if len(parts) == 2 and parts[0] and parts[1]:
            try:
                weight = float(parts[0])
                reps = int(parts[1])
                return weight, reps
            except ValueError:
                pass
    
    # If we couldn't parse the input, raise an error
    raise ValueError("Invalid format. Please use 'weight x reps' format (e.g., '70x5' or '70 x 5')")

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

def show_visualization_menu(exercise=None):
    """Show visualization options and display graphs"""
    while True:
        clear_screen()
        print("\nTrainova Feedback Network - Visualization Menu")
        print("===============================================")
        
        if exercise:
            print(f"Current exercise: {exercise}")
        else:
            print("Showing data for all exercises")
        
        print("\nVisualization Options:")
        print("1. Prediction vs Actual Weight")
        print("2. Score Distribution")
        print("3. Error Improvement Over Time")
        print("4. Show All Visualizations")
        print("5. Back to Main Menu")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Show prediction vs actual visualization
            fig, _ = plot_prediction_vs_actual(model.feedback_history, exercise)
            plt.show()
            input("\nPress Enter to continue...")
        elif choice == '2':
            # Show score distribution
            fig, _ = plot_score_distribution(model.feedback_history, exercise)
            plt.show()
            input("\nPress Enter to continue...")
        elif choice == '3':
            # Show error improvement over time
            fig, _ = plot_improvement_over_time(model.feedback_history, exercise)
            plt.show()
            input("\nPress Enter to continue...")
        elif choice == '4':
            # Show all visualizations
            fig1, _ = plot_prediction_vs_actual(model.feedback_history, exercise)
            plt.figure(1)
            plt.show(block=False)
            
            fig2, _ = plot_score_distribution(model.feedback_history, exercise)
            plt.figure(2)
            plt.show(block=False)
            
            fig3, _ = plot_improvement_over_time(model.feedback_history, exercise)
            plt.figure(3)
            plt.show()
            
            input("\nPress Enter to continue...")
            plt.close('all')
        elif choice == '5':
            return
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")
            input("\nPress Enter to continue...")

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
    print("3. View Visualizations")
    print("4. Exit")

    choice = input("\nEnter your choice (1-4): ").strip()

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
        # Show visualization menu
        exercise_for_viz = None
        if exercises:
            viz_choice = input("Show visualizations for a specific exercise? (y/n, default: n): ").strip().lower() or "n"
            if viz_choice == 'y':
                exercise_choice = input("Enter exercise number or name (or press Enter for all): ").strip()
                if exercise_choice:
                    try:
                        # Check if input is a number
                        idx = int(exercise_choice) - 1
                        if 0 <= idx < len(exercises):
                            exercise_for_viz = sorted(exercises)[idx]
                        else:
                            print("Invalid exercise number. Showing all exercises.")
                    except ValueError:
                        # Input is a name
                        if exercise_choice in exercises:
                            exercise_for_viz = exercise_choice
                        else:
                            print(f"Exercise '{exercise_choice}' not found. Showing all exercises.")
        
        show_visualization_menu(exercise_for_viz)
        clear_screen()
        continue
    elif choice == '4':
        print("\nExiting. Thank you for training the model!")
        break
    else:
        print("Invalid choice. Please enter 1, 2, 3, or 4.")
        continue

    clear_screen()
    print(f"Training with exercise: {exercise}\n")

    # Print previous workouts for this exercise
    print_previous_workouts(exercise)

    # Get workout parameters
    print("\nTIP: You can enter reps and weight together using format '70x5' or '70 x 5'")
    
    reps = None
    weight_reps_input = input("\nEnter target weight x reps for this workout (e.g. '70x5'): ").strip()
    
    # Try to parse combined format first
    if 'x' in weight_reps_input:
        try:
            actual_weight, reps = parse_weight_and_reps(weight_reps_input)
        except ValueError as e:
            print(f"Error: {str(e)}")
            try:
                reps = int(input("Enter target reps for this workout: ").strip() or "8")
            except ValueError:
                print("Invalid input. Using default value of 8 reps.")
                reps = 8
    else:
        try:
            reps = int(weight_reps_input or "8")
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
    if 'x' in weight_reps_input and 'actual_weight' in locals():
        # We already have the weight from the combined input
        print(f"\nUsing weight from combined input: {actual_weight} kg")
    else:
        try:
            weight_input = input("\nEnter the ACTUAL weight used (or should be used): ").strip()
            
            # Check if user entered in combined format here too
            if 'x' in weight_input:
                actual_weight, new_reps = parse_weight_and_reps(weight_input)
                print(f"Parsed weight: {actual_weight} kg and reps: {new_reps}")
                reps = new_reps  # Update reps with the new value
            else:
                actual_weight = float(weight_input)
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

    # Ask if user wants to see visualization after providing feedback
    view_viz = input("\nWould you like to see visualizations for this exercise? (y/n, default: n): ").strip().lower() or "n"
    if view_viz == 'y':
        show_visualization_menu(exercise)

    input("\nPress Enter to continue...")
    clear_screen()