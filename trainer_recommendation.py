import os
import sys
import json
import pandas as pd
import datetime
from typing import List, Dict, Any, Optional
from prettytable import PrettyTable

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainerRecommendationSystem:
    """
    System for collecting weight recommendations from licensed trainers
    based on mock bench press workout data.
    """
    
    def __init__(self, cases_dir: str, output_dir: str):
        """
        Initialize the trainer recommendation system.
        
        Args:
            cases_dir: Directory containing the trainer recommendation cases
            output_dir: Directory to save trainer recommendations
        """
        self.cases_dir = cases_dir
        self.output_dir = output_dir
        self.cases = []
        self.trainer_info = {}
        self.recommendations = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load cases
        self._load_cases()
    
    def _load_cases(self) -> None:
        """Load all trainer recommendation cases from the cases directory."""
        cases_file = os.path.join(self.cases_dir, "trainer_recommendation_cases.json")
        
        if not os.path.exists(cases_file):
            print(f"Error: Cases file not found at {cases_file}")
            return
        
        try:
            with open(cases_file, "r") as f:
                self.cases = json.load(f)
            print(f"Loaded {len(self.cases)} trainer recommendation cases")
        except Exception as e:
            print(f"Error loading cases: {str(e)}")
    
    def collect_trainer_info(self) -> None:
        """Collect information about the trainer providing recommendations."""
        print("\n" + "=" * 60)
        print("TRAINER INFORMATION")
        print("=" * 60)
        
        self.trainer_info = {
            "name": input("Trainer Name: ").strip(),
            "years_experience": input("Years of Experience: ").strip(),
            "specialization": input("Specialization (if any): ").strip(),
            "date": datetime.datetime.now().strftime("%Y-%m-%d")
        }
        
        # Validate required fields
        if not self.trainer_info["name"]:
            print("\nError: Name is a required field.")
            self.collect_trainer_info()
        
        print("\nThank you for providing your information.")
    
    def print_workout_history(self, workouts: List[Dict[str, Any]]) -> None:
        """
        Print workout history in a formatted table.
        
        Args:
            workouts: List of workout dictionaries to display
        """
        if not workouts:
            print("No workout history available.")
            return
        
        # Create a table
        table = PrettyTable()
        table.field_names = ["Date", "Weight (kg)", "Reps", "RIR", "Success"]
        
        # Sort workouts by date
        sorted_workouts = sorted(workouts, key=lambda w: w["date"])
        
        # Add rows to table
        for workout in sorted_workouts:
            table.add_row([
                workout["date"],
                workout["weight"],
                workout["reps"],
                workout.get("rir", "-"),
                "Yes" if workout.get("success", True) else "No"
            ])
        
        print(table)
    
    def print_user_profile(self, profile: Dict[str, Any]) -> None:
        """
        Print user profile information in a formatted way.
        
        Args:
            profile: Dictionary containing user profile data
        """
        print("\nUSER PROFILE")
        print("-" * 40)
        print(f"Age: {profile['age']} years")
        print(f"Gender: {profile['gender'].capitalize()}")
        print(f"Weight: {profile['weight_kg']} kg")
        print(f"Height: {profile['height_cm']} cm")
        print(f"BMI: {profile['bmi']}")
        print(f"Training Experience: {profile['experience_years']} years")
        print(f"Starting Bench 1RM: {profile['starting_bench_1rm_kg']} kg")
        print("-" * 40)
    
    def parse_weight_and_reps(self, input_str: str) -> tuple:
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
    
    def collect_recommendations(self) -> None:
        """Collect weight recommendations from the trainer for each case."""
        if not self.cases:
            print("No cases available. Please generate cases first.")
            return
        
        if not self.trainer_info:
            self.collect_trainer_info()
        
        print("\n" + "=" * 60)
        print("WEIGHT RECOMMENDATION SESSION")
        print("=" * 60)
        print(f"Trainer: {self.trainer_info['name']}")
        print(f"Date: {self.trainer_info['date']}")
        print("=" * 60)
        print("\nTIP: You can enter weight and reps together using format '70x5' or '70 x 5'")
        print("\nIMPORTANT: You can quit at any time by typing 'quit', 'exit', or pressing Ctrl+C.")
        print(f"Your progress will be saved to: {self.output_dir}")
        
        self.recommendations = []
        
        try:
            for i, case in enumerate(self.cases):
                case_id = case["case_id"]
                print(f"\n\nCASE #{case_id} (Case {i+1} of {len(self.cases)})")
                print("=" * 60)
                
                # Display user profile
                self.print_user_profile(case["user_profile"])
                
                # Display workout history
                print("\nWORKOUT HISTORY (Bench Press)")
                self.print_workout_history(case["workout_history"])
                
                # Get trainer recommendation
                print("\nBased on this user's profile and workout history, please provide your recommendation:")
                
                weight = None
                reps = None
                
                # Try to get weight and reps in combined format first
                while True:
                    try:
                        weight_reps_input = input("Recommended weight x reps for next workout (e.g., '70x5'): ").strip()
                        
                        # Check for quit commands
                        if weight_reps_input.lower() in ['quit', 'exit', 'q']:
                            print("\nQuitting recommendation session...")
                            # Save progress before exiting
                            if self.recommendations:
                                self._save_recommendations(self.recommendations)
                                print(f"\nProgress saved. You completed {len(self.recommendations)} of {len(self.cases)} cases.")
                            return
                        
                        # If user entered in combined format
                        if 'x' in weight_reps_input:
                            weight, reps = self.parse_weight_and_reps(weight_reps_input)
                        else:
                            # Fall back to separate inputs if no 'x' is found
                            weight = float(weight_reps_input)
                            reps_input = input("Target reps for this weight: ").strip()
                            
                            # Check for quit commands
                            if reps_input.lower() in ['quit', 'exit', 'q']:
                                print("\nQuitting recommendation session...")
                                # Save progress before exiting
                                if self.recommendations:
                                    self._save_recommendations(self.recommendations)
                                    print(f"\nProgress saved. You completed {len(self.recommendations)} of {len(self.cases)} cases.")
                                return
                                
                            reps = int(reps_input)
                        
                        # Validate weight is reasonable (5-500 kg)
                        if weight < 5 or weight > 500:
                            print("Please enter a reasonable weight between 5 and 500 kg.")
                            continue
                        
                        # Validate reps (1-20)
                        if reps < 1 or reps > 20:
                            print("Please enter reps between 1 and 20.")
                            continue
                        
                        break
                    except ValueError as e:
                        print(f"Error: {str(e)}. Please try again.")
                
                # Get target RIR (Reps In Reserve)
                while True:
                    try:
                        rir_input = input("Target RIR (Reps In Reserve, 0-5): ").strip() or "2"
                        
                        # Check for quit commands
                        if rir_input.lower() in ['quit', 'exit', 'q']:
                            print("\nQuitting recommendation session...")
                            # Save progress before exiting
                            if self.recommendations:
                                self._save_recommendations(self.recommendations)
                                print(f"\nProgress saved. You completed {len(self.recommendations)} of {len(self.cases)} cases.")
                            return
                            
                        rir = int(rir_input)
                        if rir < 0 or rir > 5:
                            print("Please enter an RIR between 0 and 5.")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid number.")
                
                # Record recommendation
                recommendation = {
                    "case_id": case_id,
                    "user_id": case["user_profile"]["user_id"],
                    "trainer_id": self.trainer_info["name"],
                    "recommended_weight": weight,
                    "target_reps": reps,
                    "target_rir": rir,
                    "reasoning": "",  # Removed reasoning requirement
                    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "actual_weight": case["validation_workout"]["weight"],
                    "actual_reps": case["validation_workout"]["reps"],
                    "actual_rir": case["validation_workout"].get("rir", 2)
                }
                
                self.recommendations.append(recommendation)
                
                # Store recommendation in the case
                case["trainer_recommendation"] = {
                    "weight": weight,
                    "reps": reps,
                    "rir": rir,
                    "reasoning": ""  # Removed reasoning requirement
                }
                
                print("\nRecommendation recorded. Moving to next case...")
                
                # Save progress after each recommendation
                self._save_recommendations(self.recommendations, partial=True)
                print(f"Progress saved ({len(self.recommendations)} of {len(self.cases)} cases completed)")
                
                continue_input = input("Press Enter to continue or type 'quit' to exit: ").strip().lower()
                if continue_input in ['quit', 'exit', 'q']:
                    print("\nQuitting recommendation session...")
                    print(f"\nProgress saved. You completed {len(self.recommendations)} of {len(self.cases)} cases.")
                    return
            
            # Final save at the end of all cases
            self._save_recommendations(self.recommendations)
            
            print("\n" + "=" * 60)
            print("RECOMMENDATIONS COMPLETED")
            print("=" * 60)
            print(f"Thank you {self.trainer_info['name']} for providing your expertise!")
            print(f"Your recommendations have been saved to {self.output_dir}")
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\nSession interrupted. Saving progress...")
            if self.recommendations:
                self._save_recommendations(self.recommendations)
                print(f"\nProgress saved. You completed {len(self.recommendations)} of {len(self.cases)} cases.")
            print("You can continue later by running the program again.")
    
    def _save_recommendations(self, recommendations: List[Dict[str, Any]], partial: bool = False) -> None:
        """
        Save trainer recommendations to files.
        
        Args:
            recommendations: List of recommendation dictionaries
            partial: Whether this is a partial save (in progress) or final
        """
        # Create a timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trainer_id = self.trainer_info["name"].replace(" ", "_").lower()
        
        # Add suffix for partial saves
        status = "partial" if partial else "complete"
        
        # Save as JSON
        json_file = os.path.join(self.output_dir, f"recommendations_{trainer_id}_{timestamp}_{status}.json")
        with open(json_file, "w") as f:
            json.dump({
                "trainer_info": self.trainer_info,
                "recommendations": recommendations,
                "timestamp": timestamp,
                "status": status,
                "completed": len(recommendations),
                "total_cases": len(self.cases)
            }, f, indent=2)
        
        # Save as CSV
        csv_file = os.path.join(self.output_dir, f"recommendations_{trainer_id}_{timestamp}_{status}.csv")
        df = pd.DataFrame(recommendations)
        df.to_csv(csv_file, index=False)
        
        # Only save full cases file for complete sessions or on explicit request
        if not partial:
            cases_file = os.path.join(self.output_dir, f"cases_with_recommendations_{trainer_id}_{timestamp}.json")
            with open(cases_file, "w") as f:
                json.dump(self.cases, f, indent=2)
        
        if not partial:
            print(f"\nRecommendations saved to:")
            print(f"- {json_file}")
            print(f"- {csv_file}")
            if not partial:
                print(f"- {cases_file}")

def main():
    """Main function to run the trainer recommendation system."""
    print("\nTrainova Feedback Network - Trainer Recommendation System")
    print("="*60)
    print("This system collects weight recommendations from licensed trainers")
    print("based on mock bench press workout data.")
    print("="*60)
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cases_dir = os.path.join(base_dir, "data", "datasets", "trainer_cases")
    output_dir = os.path.join(base_dir, "data", "datasets", "trainer_recommendations")
    
    # Ensure directories exist
    os.makedirs(cases_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Display data paths
    print(f"\nData paths:")
    print(f"- Cases source: {cases_dir}")
    print(f"- Recommendations will be saved to: {output_dir}")
    
    # Check if cases directory is empty, if so generate mock data
    if not os.listdir(cases_dir):
        print("No trainer cases found. Generating mock data...")
        
        # Import the mock data generator
        try:
            from data.mock_data_generator import generate_mock_dataset, prepare_trainer_recommendation_cases
        except ImportError:
            print("Error: Could not import mock data generator. Make sure you're running from the correct directory.")
            return
        
        # Generate mock dataset
        mock_dataset = generate_mock_dataset(num_users=30, workouts_per_user=15)
        
        # Prepare trainer recommendation cases
        prepare_trainer_recommendation_cases(
            mock_dataset, 
            num_cases=15,
            case_dir=cases_dir
        )
    
    # Initialize the recommendation system
    recommendation_system = TrainerRecommendationSystem(cases_dir, output_dir)
    
    try:
        # Collect recommendations
        recommendation_system.collect_recommendations()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Any saved progress is available in the output directory.")
    
    # Exit
    print("\nThank you for using the Trainer Recommendation System!")

if __name__ == "__main__":
    main()