import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

# Optional imports for advanced features - will be checked at runtime
# Import flags to track which optional dependencies are available
HAS_STATSMODELS = False
HAS_MATPLOTLIB = False
HAS_PYTORCH = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    HAS_STATSMODELS = True
except ImportError:
    warnings.warn("statsmodels not installed; time-series forecasting features will be disabled")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    warnings.warn("matplotlib not installed; visualization features will be disabled")
    
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    warnings.warn("PyTorch not installed; deep learning features will be disabled")

class FeedbackBasedPredictionModel:
    """
    A prediction model that uses feedback to adjust and improve its predictions
    for progressive overload in weightlifting.
    """
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.dirname(os.path.abspath(__file__))
        self.metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        self.weights_scaler_path = os.path.join(self.model_dir, "weight_scaler.joblib")
        self.feature_scaler_path = os.path.join(self.model_dir, "feature_scaler.joblib")
        self.exercise_encoder_path = os.path.join(self.model_dir, "exercise_encoder.joblib")
        
        # Added paths for advanced models
        self.arima_models_path = os.path.join(self.model_dir, "arima_models.joblib")
        self.lstm_model_path = os.path.join(self.model_dir, "lstm_model.pt")
        
        # Prediction weights for different factors
        self.prediction_weights = {
            "last_weight": 0.6,  # Weight given to the last workout's weight
            "avg_progress": 0.3,  # Weight given to the average progress rate (increased from 0.2)
            "consistency": 0.05,  # Weight given to consistency factor (decreased from 0.1)
            "volume": 0.05,       # Weight given to total volume factor (decreased from 0.1)
        }
        
        # Constants for rep progression
        self.rep_progression_threshold = 8  # When user hits this many reps, suggest weight increase
        self.consecutive_success_threshold = 2  # Number of successful sessions needed before weight increase
        
        # Add advanced model weights if available
        if HAS_STATSMODELS:
            self.prediction_weights["time_series"] = 0.0  # Will be adjusted when model is available
            
        if HAS_PYTORCH:
            self.prediction_weights["deep_learning"] = 0.0  # Will be adjusted when model is available
        
        # Feature engineering parameters
        self.moving_average_windows = [3, 5, 10]  # Windows for moving averages
        self.recent_max_windows = [5, 10, 20]     # Windows for recent maximums
        
        # Time series model parameters
        self.arima_models = {}
        self.arima_order = (2, 1, 1)  # Default ARIMA model order (p,d,q)
        
        # Deep learning parameters
        self.lstm_model = None
        self.sequence_length = 5  # Number of previous workouts to use for LSTM input
        
        # Feedback adjustments
        self.feedback_history = []
        self.feedback_influence = 0.1  # How much feedback affects future predictions
        
        # Initialize or load models and scalers
        self.weights_scaler = None
        self.feature_scaler = None
        self.exercise_encoder = None
        self._load_or_initialize()
    
    def _load_or_initialize(self):
        """Load existing models and metadata or initialize new ones"""
        try:
            # Try to load the metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.prediction_weights = metadata.get('prediction_weights', self.prediction_weights)
                self.feedback_influence = metadata.get('feedback_influence', self.feedback_influence)
                self.feedback_history = metadata.get('feedback_history', [])
            
            # Try to load the scalers and encoder
            if os.path.exists(self.weights_scaler_path):
                self.weights_scaler = joblib.load(self.weights_scaler_path)
            
            if os.path.exists(self.feature_scaler_path):
                self.feature_scaler = joblib.load(self.feature_scaler_path)
                
            if os.path.exists(self.exercise_encoder_path):
                self.exercise_encoder = joblib.load(self.exercise_encoder_path)
                
            # Load advanced models if dependencies are available
            if HAS_STATSMODELS and os.path.exists(self.arima_models_path):
                try:
                    self.arima_models = joblib.load(self.arima_models_path)
                    # If we've loaded ARIMA models, update prediction weights
                    if self.arima_models:
                        # Adjust weights to include time series predictions
                        self.prediction_weights["time_series"] = 0.2
                        self._normalize_prediction_weights()
                except Exception as e:
                    print(f"Error loading ARIMA models: {e}")
                
            if HAS_PYTORCH and os.path.exists(self.lstm_model_path):
                try:
                    # For PyTorch, we directly load the model file
                    self.lstm_model = torch.load(self.lstm_model_path)
                    # If we've loaded LSTM model, update prediction weights
                    if self.lstm_model:
                        self.prediction_weights["deep_learning"] = 0.1
                        self._normalize_prediction_weights()
                except Exception as e:
                    print(f"Error loading LSTM model: {e}")
                    
        except Exception as e:
            print(f"Error loading model components: {e}")
            # Continue with initialization of new components
    
    def _normalize_prediction_weights(self):
        """Ensure prediction weights sum to 1.0"""
        total = sum(self.prediction_weights.values())
        if total > 0:
            for key in self.prediction_weights:
                self.prediction_weights[key] /= total
    
    def save(self):
        """Save model metadata and components"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'prediction_weights': self.prediction_weights,
            'feedback_influence': self.feedback_influence,
            'feedback_history': self.feedback_history[-100:],  # Keep only the last 100 feedback entries
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Save scalers if they exist
        if self.weights_scaler:
            joblib.dump(self.weights_scaler, self.weights_scaler_path)
        
        if self.feature_scaler:
            joblib.dump(self.feature_scaler, self.feature_scaler_path)
            
        if self.exercise_encoder:
            joblib.dump(self.exercise_encoder, self.exercise_encoder_path)
    
    def fit(self, workout_data: pd.DataFrame):
        """
        Train the prediction model using historical workout data
        
        Args:
            workout_data: DataFrame with columns like 'exercise', 'weight', 'reps', 'date', etc.
        """
        if workout_data.empty:
            print("No data provided for training")
            return
        
        # Initialize scalers and encoders if not already loaded
        if self.weights_scaler is None:
            self.weights_scaler = StandardScaler()
            self.weights_scaler.fit(workout_data[['weight']].values)
        
        # Feature scaler for numerical features (apart from weight)
        if self.feature_scaler is None and 'reps' in workout_data.columns:
            numerical_features = workout_data[['reps']].values
            if numerical_features.size > 0:
                self.feature_scaler = StandardScaler()
                self.feature_scaler.fit(numerical_features)
        
        # Exercise encoder for categorical features
        if self.exercise_encoder is None and 'exercise' in workout_data.columns:
            from sklearn.preprocessing import LabelEncoder
            self.exercise_encoder = LabelEncoder()
            self.exercise_encoder.fit(workout_data['exercise'])
        
        # Analysis to determine average progress rates per exercise
        exercise_progress = self._analyze_progress_rates(workout_data)
        
        # Enhanced: Fit advanced time-series models if dependencies are available
        if HAS_STATSMODELS and len(workout_data) >= 8:
            try:
                print("Fitting time-series ARIMA models...")
                self._fit_arima_models(workout_data)
            except Exception as e:
                print(f"Error fitting ARIMA models: {e}")
                
        # Enhanced: Fit deep learning model if PyTorch is available
        if HAS_PYTORCH and len(workout_data) >= 20:
            try:
                print("Fitting LSTM deep learning model...")
                self._fit_lstm_model(workout_data)
            except Exception as e:
                print(f"Error fitting LSTM model: {e}")
                
        # Set advanced model weights if models exist
        self._update_prediction_weights_for_advanced_models()
        
        # Save the fitted model and encoders
        self.save()
        
        return self
    
    def _update_prediction_weights_for_advanced_models(self):
        """Update prediction weights based on available advanced models"""
        # If ARIMA models are available, add weight for time series predictions
        if HAS_STATSMODELS and self.arima_models:
            self.prediction_weights["time_series"] = 0.2
            
        # If LSTM model is available, add weight for deep learning predictions
        if HAS_PYTORCH and self.lstm_model is not None:
            self.prediction_weights["deep_learning"] = 0.1
            
        # Normalize weights to ensure they sum to 1.0
        self._normalize_prediction_weights()
    
    def _fit_arima_models(self, workout_data: pd.DataFrame):
        """
        Fit ARIMA models for each exercise in the workout data
        
        Args:
            workout_data: DataFrame with workout data
        """
        if 'exercise' not in workout_data.columns or 'weight' not in workout_data.columns:
            print("Insufficient data to fit ARIMA models")
            return
        
        # Group data by exercise
        grouped = workout_data.groupby('exercise')
        
        arima_models = {}
        
        for exercise, group in grouped:
            # Ensure data is sorted by date
            group = group.sort_values('date')
            
            # Fit ARIMA model
            try:
                model = ARIMA(group['weight'], order=self.arima_order)
                model_fit = model.fit()
                
                # Save the model
                arima_models[exercise] = model_fit
                
                print(f"Fitted ARIMA model for {exercise}")
            except Exception as e:
                print(f"Error fitting ARIMA model for {exercise}: {e}")
        
        # Save all ARIMA models
        joblib.dump(arima_models, self.arima_models_path)
        self.arima_models = arima_models
    
    def _fit_lstm_model(self, workout_data: pd.DataFrame):
        """
        Fit LSTM model for predicting workout weights
        
        Args:
            workout_data: DataFrame with workout data
        """
        if 'exercise' not in workout_data.columns or 'weight' not in workout_data.columns:
            print("Insufficient data to fit LSTM model")
            return
        
        # Prepare data for LSTM
        # We need to create sequences of previous workouts for each exercise
        workout_data = workout_data.sort_values('date')
        
        # Create a mapping of exercise to its sequential data
        exercise_sequences = {}
        
        for exercise, group in workout_data.groupby('exercise'):
            # Ensure group is sorted by date
            group = group.sort_values('date')
            
            # Create sequences
            sequences = []
            weights = group['weight'].values
            for i in range(self.sequence_length, len(weights)):
                seq = weights[i-self.sequence_length:i]
                sequences.append(seq)
            
            exercise_sequences[exercise] = np.array(sequences)
        
        # Define LSTM model architecture
        model = LSTMModel(input_size=1, hidden_size=50, output_size=1)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model on each exercise's data
        for exercise, sequences in exercise_sequences.items():
            # Reshape input to be 3D [samples, time steps, features]
            X = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
            
            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(sequences, dtype=torch.float32).view(-1, 1)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Train the model
            model.train()
            for epoch in range(200):
                for x_batch, y_batch in loader:
                    optimizer.zero_grad()
                    y_pred = model(x_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
            
            print(f"Trained LSTM model for {exercise}")
        
        # Save the LSTM model
        torch.save(model, self.lstm_model_path)
        self.lstm_model = model
    
    def _analyze_progress_rates(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze historical data to determine average progress rates per exercise
        
        Returns:
            Dictionary mapping exercise names to their average progress rates
        """
        progress_rates = {}
        
        if 'exercise' not in data.columns or 'weight' not in data.columns:
            return progress_rates
            
        # Group by exercise and sort by date for each group
        if 'date' in data.columns:
            data = data.sort_values('date')
        
        for exercise in data['exercise'].unique():
            exercise_data = data[data['exercise'] == exercise]
            
            if len(exercise_data) < 2:
                progress_rates[exercise] = 0.0  # Not enough data for progress calculation
                continue
                
            # Calculate average progress rate
            weights = exercise_data['weight'].values
            weight_changes = np.diff(weights)
            avg_change = np.mean(weight_changes) if weight_changes.size > 0 else 0
            
            # Normalize by the average weight to get a percentage
            avg_weight = np.mean(weights)
            progress_rate = (avg_change / avg_weight) if avg_weight > 0 else 0
            
            progress_rates[exercise] = progress_rate
            
        return progress_rates
    
    def predict(self, exercise: str, previous_workouts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict weight for the next workout based on previous workouts
        
        Args:
            exercise: Name of the exercise
            previous_workouts: List of dicts containing previous workout data
                Each dict should have 'weight', 'reps', etc.
                
        Returns:
            Dictionary with predicted weight and confidence
        """
        if not previous_workouts:
            return {"weight": 0, "confidence": 0, "message": "No previous workout data provided"}
        
        # Sort workouts by date if available
        if 'date' in previous_workouts[0]:
            previous_workouts = sorted(previous_workouts, key=lambda x: x['date'])
        
        # Extract the last workout's weight and reps
        last_workout = previous_workouts[-1]
        last_weight = last_workout.get('weight', 0)
        last_reps = last_workout.get('reps', 8)  # Default to 8 if not specified
        
        # Calculate consistency factor (std deviation of weights)
        weights = [w.get('weight', 0) for w in previous_workouts]
        consistency = 1.0 / (1.0 + np.std(weights)) if len(weights) > 1 else 0.5
        
        # Calculate volume factor with more emphasis on reps
        volumes = [w.get('weight', 0) * w.get('reps', 0) for w in previous_workouts]
        avg_volume = np.mean(volumes) if volumes else 0
        volume_factor = min(avg_volume / 100, 1.0)  # Normalize to 0-1
        
        # Calculate progress rate (weight change per workout)
        if len(weights) > 1:
            weight_changes = np.diff(weights)
            avg_progress = np.mean(weight_changes) if len(weight_changes) > 0 else 0
        else:
            avg_progress = 0
        
        # Calculate rep-based adjustment
        rep_adjustment = self._calculate_rep_adjustment(previous_workouts, last_reps)
        
        # ENHANCED: Compute engineered features
        engineered_features = self._compute_engineered_features(previous_workouts)
        
        # Basic prediction with original factors
        weighted_prediction = (
            self.prediction_weights["last_weight"] * last_weight +
            self.prediction_weights["avg_progress"] * (last_weight + avg_progress) +
            self.prediction_weights["consistency"] * consistency * last_weight +
            self.prediction_weights["volume"] * volume_factor * last_weight
        )
        
        # Apply engineered features adjustments
        if engineered_features:
            # Use moving averages for smoothing if available
            if 'weight_ma_3' in engineered_features:
                # Blend prediction with 3-workout moving average (80/20 blend)
                weighted_prediction = 0.8 * weighted_prediction + 0.2 * engineered_features['weight_ma_3']
            
            # Adjust for fatigue if detected
            if 'fatigue_index_short' in engineered_features:
                fatigue = engineered_features['fatigue_index_short']
                # If fatigue is detected (index < 0.95), reduce the prediction slightly
                if fatigue < 0.95:
                    weighted_prediction = weighted_prediction * (0.95 + (fatigue - 0.95) * 2)
                # If recovery is good (index > 1.05), increase prediction slightly
                elif fatigue > 1.05:
                    weighted_prediction = weighted_prediction * (1.05 + (fatigue - 1.05) * 0.5)
            
            # Consider trend slope for additional adjustment
            if 'weight_trend_slope' in engineered_features:
                slope = engineered_features['weight_trend_slope']
                # Normalize the slope to a small adjustment factor
                slope_factor = max(min(slope / (last_weight * 0.1), 0.05), -0.05) if last_weight > 0 else 0
                weighted_prediction = weighted_prediction * (1 + slope_factor)
                
            # Use recent maximums as a ceiling reference
            if 'weight_max_5' in engineered_features:
                recent_max = engineered_features['weight_max_5']
                # If prediction is significantly higher than recent max, temper it
                if weighted_prediction > recent_max * 1.1:
                    weighted_prediction = (weighted_prediction + recent_max * 1.1) / 2
        
        # Apply feedback adjustment
        feedback_adjustment = self._calculate_feedback_adjustment(exercise)
        adjusted_prediction = weighted_prediction * (1 + feedback_adjustment) * (1 + rep_adjustment)
        
        # Round to nearest 2.5kg increment
        rounded_weight = self._round_to_increment(adjusted_prediction)
        
        # Calculate enhanced confidence based on amount of data and prediction quality
        data_points_factor = min(len(previous_workouts) / 10, 1.0)
        rep_consistency = self._calculate_rep_consistency(previous_workouts)
        
        # Base confidence calculation
        confidence = 0.5 + (0.3 * data_points_factor) + (0.1 * consistency) + (0.1 * rep_consistency)
        
        # Generate intensity-appropriate rep scheme based on the predicted weight
        suggested_reps = self._generate_intensity_based_reps(last_weight, last_reps, rounded_weight)
        
        # Build enhanced response with additional analysis
        response = {
            "weight": rounded_weight,
            "confidence": round(min(confidence, 1.0), 2),  # Cap at 1.0
            "message": f"Prediction based on {len(previous_workouts)} previous workouts",
            "suggested_reps": suggested_reps
        }
        
        # Add advanced analysis if we have enough data
        if len(previous_workouts) >= 5:
            analysis = {
                "trend": "increasing" if avg_progress > 0 else "decreasing",
                "consistency": "high" if consistency > 0.8 else "medium" if consistency > 0.5 else "low"
            }
            
            # Add fatigue analysis if available
            if 'fatigue_index_short' in engineered_features:
                fatigue = engineered_features['fatigue_index_short']
                if fatigue < 0.95:
                    analysis["fatigue"] = "high"
                elif fatigue > 1.05:
                    analysis["fatigue"] = "recovered"
                else:
                    analysis["fatigue"] = "normal"
                    
            # Add volume analysis
            if 'volume_trend_slope' in engineered_features:
                volume_trend = engineered_features['volume_trend_slope']
                analysis["volume_trend"] = "increasing" if volume_trend > 0 else "decreasing"
                
            response["analysis"] = analysis
        
        return response
    
    def _calculate_rep_adjustment(self, previous_workouts: List[Dict[str, Any]], target_reps: int) -> float:
        """
        Calculate weight adjustment based on rep count differences
        
        Args:
            previous_workouts: List of previous workout data
            target_reps: Target rep count for the next workout
            
        Returns:
            Adjustment factor for weight prediction
        """
        # Get average reps from previous workouts
        all_reps = [w.get('reps', 0) for w in previous_workouts if w.get('reps', 0) > 0]
        if not all_reps:
            return 0.0
            
        avg_reps = np.mean(all_reps)
        
        # Calculate adjustment based on target reps vs average reps
        # Using a general strength training principle: ~2.5% weight change per rep
        rep_diff = avg_reps - target_reps
        adjustment = rep_diff * 0.025  # 2.5% per rep difference
        
        # Limit the adjustment to prevent extreme predictions
        adjustment = max(min(adjustment, 0.2), -0.2)
        
        return adjustment
    
    def _calculate_rep_consistency(self, previous_workouts: List[Dict[str, Any]]) -> float:
        """
        Calculate consistency factor based on rep counts
        
        Args:
            previous_workouts: List of previous workout data
            
        Returns:
            Rep consistency factor (0-1)
        """
        # Get reps from previous workouts
        reps = [w.get('reps', 0) for w in previous_workouts if w.get('reps', 0) > 0]
        if len(reps) < 2:
            return 0.5  # Default consistency if not enough data
            
        # Calculate consistency as inverse of coefficient of variation
        std_dev = np.std(reps)
        mean_reps = np.mean(reps)
        
        if mean_reps > 0:
            coef_var = std_dev / mean_reps
            # Convert to a 0-1 scale where 1 is perfectly consistent
            consistency = 1.0 / (1.0 + coef_var)
            return consistency
        else:
            return 0.5
    
    def _generate_suggested_reps(self, base_reps: int) -> List[int]:
        """
        Generate suggested reps for a workout based on base rep count
        
        Args:
            base_reps: Base rep count to build suggestions from
            
        Returns:
            List of suggested rep counts for each set
        """
        # Default to 3 sets with a descending pattern
        return [base_reps, max(base_reps - 1, 1), max(base_reps - 2, 1)]
    
    def provide_feedback(self, 
                        exercise: str, 
                        predicted_weight: float, 
                        actual_weight: float, 
                        success: bool,
                        reps: int = None) -> Dict[str, Any]:
        """
        Provide feedback on a prediction to improve future predictions
        
        Args:
            exercise: Name of the exercise
            predicted_weight: The weight that was predicted
            actual_weight: The weight that was actually used
            success: Whether the workout was completed successfully
            reps: Number of reps completed (optional)
            
        Returns:
            Dict with feedback results
        """
        # Calculate score based on prediction accuracy
        weight_diff = actual_weight - predicted_weight
        
        # Normalize the difference relative to the weight
        if predicted_weight > 0:
            relative_diff = weight_diff / predicted_weight
        else:
            relative_diff = 0
            
        # Calculate score: positive if prediction was too low, negative if too high
        # Scale is from -1 to 1 where 0 is perfect
        score = max(min(relative_diff, 1.0), -1.0)
        
        # If the workout was unsuccessful, adjust the score
        if not success:
            # If prediction was too high and workout failed, strengthen the negative feedback
            if score < 0:
                score = score * 1.5
        
        # Record this feedback for future predictions
        feedback_entry = {
            'exercise': exercise,
            'predicted_weight': predicted_weight,
            'actual_weight': actual_weight,
            'success': success,
            'score': score,
            'reps': reps
        }
        self.feedback_history.append(feedback_entry)
        
        # Adjust prediction weights based on this feedback
        self._update_prediction_weights(score)
        
        # Save updated model
        self.save()
        
        return {
            'feedback_recorded': True,
            'score': round(score, 3),
            'message': self._generate_feedback_message(score)
        }
    
    def _update_prediction_weights(self, score: float):
        """
        Update the prediction weights based on feedback score
        
        Args:
            score: Feedback score from -1 to 1
        """
        # The magnitude of the score determines how much to adjust weights
        adjustment_factor = abs(score) * self.feedback_influence
        
        # If score is negative (prediction too high), increase weight of consistency
        # If score is positive (prediction too low), increase weight of progress
        if score < 0:
            # Prediction was too high, increase consistency weight
            self.prediction_weights["consistency"] += adjustment_factor
            self.prediction_weights["avg_progress"] -= adjustment_factor
        else:
            # Prediction was too low, increase progress weight
            self.prediction_weights["avg_progress"] += adjustment_factor
            self.prediction_weights["consistency"] -= adjustment_factor
        
        # Normalize weights to ensure they sum to 1
        total = sum(self.prediction_weights.values())
        for key in self.prediction_weights:
            self.prediction_weights[key] /= total
    
    def _generate_feedback_message(self, score: float) -> str:
        """
        Generate a human-readable feedback message based on feedback score
        
        Args:
            score: Feedback score from -1 to 1
            
        Returns:
            Human-readable message about the prediction accuracy
        """
        if abs(score) < 0.05:
            return "The prediction was very accurate!"
        elif score > 0.15:
            return "The prediction was too low. We'll adjust future predictions upward."
        elif score > 0.05:
            return "The prediction was slightly conservative. Minor adjustments will be made."
        elif score < -0.15:
            return "The prediction was too high. We'll be more conservative next time."
        elif score < -0.05:
            return "The prediction was slightly aggressive. Minor adjustments will be made."
        else:
            return "The prediction was reasonably accurate."
    
    def _calculate_weight_for_reps(self, one_rep_max: float, target_reps: int, formula: str = 'brzycki') -> float:
        """
        Calculate the appropriate weight to use for a specific rep target
        
        Args:
            one_rep_max: Estimated 1RM
            target_reps: Desired number of reps
            formula: Which formula to use ('epley', 'brzycki', etc.)
            
        Returns:
            Appropriate weight for the target rep count
        """
        if target_reps <= 1:
            return one_rep_max
            
        if formula.lower() == 'epley':
            # Rearranged Epley formula: w = 1RM / (1 + r/30)
            return one_rep_max / (1 + target_reps/30)
        elif formula.lower() == 'brzycki':
            # Rearranged Brzycki formula: w = 1RM × (1.0278 - 0.0278 × r)
            return one_rep_max * (1.0278 - 0.0278 * target_reps)
        elif formula.lower() == 'lombardi':
            # Rearranged Lombardi formula: w = 1RM / r^0.10
            return one_rep_max / (target_reps ** 0.10)
        elif formula.lower() == 'mayhew':
            # Mayhew formula is difficult to rearrange, so we'll use an approximation
            # based on common percentage tables
            percentages = {
                1: 1.00, 2: 0.95, 3: 0.93, 4: 0.90, 5: 0.87, 
                6: 0.85, 7: 0.83, 8: 0.80, 9: 0.77, 10: 0.75,
                11: 0.73, 12: 0.70, 15: 0.65, 20: 0.60
            }
            
            # Find the closest key in the percentages dict
            closest_rep = min(percentages.keys(), key=lambda x: abs(x - target_reps))
            percentage = percentages[closest_rep]
            
            return one_rep_max * percentage
        else:
            # Default to averaged prediction
            epley = one_rep_max / (1 + target_reps/30)
            brzycki = one_rep_max * (1.0278 - 0.0278 * target_reps)
            return (epley + brzycki) / 2
    
    def _generate_intensity_based_reps(self, previous_weight: float, previous_reps: int, predicted_weight: float) -> List[int]:
        """
        Generate rep scheme based on intensity relationship 
        
        Args:
            previous_weight: Weight from previous workout
            previous_reps: Reps from previous workout
            predicted_weight: Predicted weight for next workout
            
        Returns:
            List of suggested rep counts for sets
        """
        # Calculate estimated 1RM based on previous performance
        # We'll use Brzycki formula as it's more accurate in the 2-10 rep range
        estimated_1rm = self._calculate_one_rep_max(previous_weight, previous_reps, formula='brzycki')
        
        # Calculate intensity percentage of the predicted weight relative to 1RM
        intensity = (predicted_weight / estimated_1rm) if estimated_1rm > 0 else 0.75
        
        # Map intensity to appropriate rep ranges
        # Standard intensity-rep mapping based on strength training principles
        intensity_mapping = {
            # Intensity: Appropriate rep range
            1.00: 1,    # 100% 1RM = 1 rep
            0.95: 2,    # 95% 1RM = 2 reps
            0.93: 3,    # 93% 1RM = 3 reps
            0.90: 4,    # 90% 1RM = 4 reps
            0.87: 5,    # 87% 1RM = 5 reps
            0.85: 6,    # 85% 1RM = 6 reps
            0.83: 7,    # 83% 1RM = 7 reps
            0.80: 8,    # 80% 1RM = 8 reps
            0.77: 9,    # 77% 1RM = 9 reps
            0.75: 10,   # 75% 1RM = 10 reps
            0.73: 11,   # 73% 1RM = 11 reps
            0.70: 12,   # 70% 1RM = 12 reps
            0.65: 15,   # 65% 1RM = 15 reps
            0.60: 20,   # 60% 1RM = 20 reps
        }
        
        # Find the closest intensity level in our mapping
        closest_intensity = min(intensity_mapping.keys(), key=lambda x: abs(x - intensity))
        suggested_base_reps = intensity_mapping[closest_intensity]
        
        # Create a rep scheme based on the base rep count
        # For intermediate and advanced lifters, a slight rep drop is common across sets
        if suggested_base_reps <= 5:
            # For low-rep/high-intensity work, keep reps consistent to maintain intensity
            return [suggested_base_reps] * 3
        elif suggested_base_reps <= 8:
            # For moderate rep ranges, use a slight drop-off
            return [suggested_base_reps, suggested_base_reps - 1, suggested_base_reps - 1]
        else:
            # For higher rep ranges, use a more pronounced drop-off
            return [suggested_base_reps, suggested_base_reps - 2, suggested_base_reps - 3]
    
    def _compute_engineered_features(self, previous_workouts: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute advanced engineered features from workout history
        
        Args:
            previous_workouts: List of previous workout data
            
        Returns:
            Dictionary of engineered features
        """
        features = {}
        
        if not previous_workouts or len(previous_workouts) < 3:
            return features
        
        # Ensure workouts are sorted by date
        if 'date' in previous_workouts[0]:
            previous_workouts = sorted(previous_workouts, key=lambda x: x['date'])
        
        # Extract weights, reps, and calculate volumes
        weights = np.array([w.get('weight', 0) for w in previous_workouts])
        reps = np.array([w.get('reps', 0) for w in previous_workouts])
        sets = np.array([w.get('sets', 1) for w in previous_workouts])
        volumes = weights * reps * sets
        
        # Moving averages for weights
        for window in self.moving_average_windows:
            if len(weights) >= window:
                features[f'weight_ma_{window}'] = np.mean(weights[-window:])
        
        # Moving averages for volume (crucial for progressive overload tracking)
        for window in self.moving_average_windows:
            if len(volumes) >= window:
                features[f'volume_ma_{window}'] = np.mean(volumes[-window:])
        
        # Recent maximums
        for window in self.recent_max_windows:
            if len(weights) >= window:
                features[f'weight_max_{window}'] = np.max(weights[-window:])
                features[f'volume_max_{window}'] = np.max(volumes[-window:])
        
        # Calculate fatigue indices
        if len(weights) >= 10:
            # Short-term fatigue: ratio of recent weights to slightly older weights
            recent_avg = np.mean(weights[-3:])
            older_avg = np.mean(weights[-10:-3])
            features['fatigue_index_short'] = recent_avg / older_avg if older_avg > 0 else 1.0
            
            # Volume fatigue: ratio of recent volume to older volume
            recent_vol_avg = np.mean(volumes[-3:])
            older_vol_avg = np.mean(volumes[-10:-3])
            features['fatigue_index_volume'] = recent_vol_avg / older_vol_avg if older_vol_avg > 0 else 1.0
        
        # Calculate training density and frequency features
        if 'date' in previous_workouts[0] and len(previous_workouts) >= 2:
            try:
                import datetime
                dates = [w.get('date') for w in previous_workouts if w.get('date')]
                
                # Convert to datetime objects if they're strings
                if isinstance(dates[0], str):
                    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') 
                            if isinstance(date, str) else date for date in dates]
                
                # Calculate days between workouts
                day_diffs = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
                if day_diffs:
                    features['avg_days_between_workouts'] = np.mean(day_diffs)
                    if len(day_diffs) >= 3:
                        features['recent_training_frequency'] = 7 / np.mean(day_diffs[-3:]) if np.mean(day_diffs[-3:]) > 0 else 0
            except (ValueError, TypeError, AttributeError) as e:
                # Continue if date parsing fails
                print(f"Error processing dates: {e}")
        
        # Calculate intensity features (percentage of 1RM)
        if len(weights) > 0 and len(reps) > 0:
            # Get estimated 1RM for the last workout
            last_weight = weights[-1]
            last_reps = reps[-1]
            estimated_1rm = self._calculate_one_rep_max(last_weight, last_reps, formula='brzycki')
            
            # Calculate intensity for each workout
            intensities = []
            for w, r in zip(weights, reps):
                if r > 0 and estimated_1rm > 0:
                    workout_1rm = self._calculate_one_rep_max(w, r, formula='brzycki')
                    intensities.append(w / workout_1rm if workout_1rm > 0 else 0)
            
            if intensities:
                features['avg_intensity'] = np.mean(intensities)
                if len(intensities) >= 6:
                    features['recent_intensity_trend'] = np.mean(intensities[-3:]) - np.mean(intensities[:-3])
        
        # Calculate progress metrics
        if len(weights) >= 5:
            # Linear regression on recent weights to detect trends
            try:
                from sklearn.linear_model import LinearRegression
                x = np.arange(len(weights[-5:])).reshape(-1, 1)
                y = weights[-5:].reshape(-1, 1)
                model = LinearRegression().fit(x, y)
                features['weight_trend_slope'] = model.coef_[0][0]
                
                # Do the same for volume
                y_vol = volumes[-5:].reshape(-1, 1)
                model_vol = LinearRegression().fit(x, y_vol)
                features['volume_trend_slope'] = model_vol.coef_[0][0]
            except Exception as e:
                # Fall back to simple difference if regression fails
                features['weight_trend_slope'] = (weights[-1] - weights[-5]) / 4
                features['volume_trend_slope'] = (volumes[-1] - volumes[-5]) / 4
        
        return features
    
    def _calculate_one_rep_max(self, weight: float, reps: int, formula: str = 'brzycki') -> float:
        """
        Calculate the one rep max based on weight and reps
        
        Args:
            weight: Weight used
            reps: Number of reps completed
            formula: Which formula to use ('epley', 'brzycki', etc.)
            
        Returns:
            Estimated one rep max
        """
        if reps <= 0 or weight <= 0:
            return 0
            
        if formula.lower() == 'epley':
            return weight * (1 + reps/30)
        elif formula.lower() == 'brzycki':
            return weight / (1.0278 - 0.0278 * reps) if reps < 37 else weight * (1 + reps/30)
        elif formula.lower() == 'lombardi':
            return weight * (reps ** 0.10)
        elif formula.lower() == 'mayhew':
            return (100 * weight) / (52.2 + 41.9 * np.exp(-0.055 * reps))
        else:
            # Default to averaged prediction
            epley = weight * (1 + reps/30)
            brzycki = weight / (1.0278 - 0.0278 * reps) if reps < 37 else epley  # Brzycki has a limit
            return (epley + brzycki) / 2
    
    def _calculate_feedback_adjustment(self, exercise: str) -> float:
        """
        Calculate adjustment factor based on previous feedback for this exercise
        
        Args:
            exercise: Name of the exercise
            
        Returns:
            Adjustment factor to apply to prediction
        """
        # Get relevant feedback for this exercise
        relevant_feedback = [f for f in self.feedback_history if f.get('exercise') == exercise]
        
        if not relevant_feedback:
            return 0.0
        
        # Use more recent feedback with higher weight
        recency_weights = [1 / (i + 1) for i in range(len(relevant_feedback))]
        total_weight = sum(recency_weights)
        
        if total_weight <= 0:
            return 0.0
            
        # Calculate weighted average feedback score
        weighted_scores = [f.get('score', 0) * w for f, w in zip(relevant_feedback, recency_weights)]
        avg_score = sum(weighted_scores) / total_weight
        
        # Convert score to adjustment factor
        # Positive score means prediction was too low, so increase
        # Negative score means prediction was too high, so decrease
        adjustment = avg_score * self.feedback_influence
        
        return adjustment
        
    def _round_to_increment(self, weight: float, increment: float = 2.5) -> float:
        """
        Round the weight to the nearest increment (usually 2.5kg/5lb)
        
        Args:
            weight: Weight to round
            increment: Increment to round to
            
        Returns:
            Rounded weight
        """
        return round(weight / increment) * increment

class LSTMModel(nn.Module):
    """
    PyTorch LSTM model for weight prediction
    """
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        x = self.dropout(lstm_out)
        x = self.fc(x)
        return x