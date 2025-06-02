import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import io
import base64
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path to import from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.feedback_prediction_model import FeedbackBasedPredictionModel

def create_feedback_dataframe(feedback_history: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a DataFrame from feedback history for visualization
    
    Args:
        feedback_history: List of feedback entries from the model
        
    Returns:
        DataFrame containing feedback data with timestamps
    """
    if not feedback_history:
        return pd.DataFrame()
        
    # Convert feedback history to DataFrame
    df = pd.DataFrame(feedback_history)
    
    # Add index as a surrogate for time if no timestamp is available
    if 'timestamp' not in df.columns:
        df['timestamp'] = range(len(df))
        
    return df

def plot_prediction_vs_actual(feedback_history: List[Dict[str, Any]], 
                             exercise: Optional[str] = None,
                             last_n: Optional[int] = None) -> Tuple[plt.Figure, str]:
    """
    Generate a plot comparing predicted weights vs actual weights over time
    
    Args:
        feedback_history: List of feedback entries from the model
        exercise: Optional filter for a specific exercise
        last_n: Optional number of most recent records to include
        
    Returns:
        Tuple of (matplotlib figure, base64 encoded image string)
    """
    df = create_feedback_dataframe(feedback_history)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No feedback data available for visualization", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return fig, img_str
    
    # Filter by exercise if specified
    if exercise:
        df = df[df['exercise'] == exercise]
        if df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"No feedback data available for exercise: {exercise}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return fig, img_str
    
    # Filter to last N records if specified
    if last_n and last_n < len(df):
        df = df.iloc[-last_n:]
    
    # Create figure and axis for the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot predicted and actual weights
    ax.plot(df.index, df['predicted_weight'], 'o-', label='Predicted Weight', color='blue', alpha=0.7)
    ax.plot(df.index, df['actual_weight'], 'o-', label='Actual Weight', color='green', alpha=0.7)
    
    # Calculate error (difference between predicted and actual)
    df['error'] = df['predicted_weight'] - df['actual_weight']
    
    # Plot error margin as a shaded area
    ax.fill_between(df.index, 
                    df['predicted_weight'] - abs(df['error']), 
                    df['predicted_weight'] + abs(df['error']), 
                    color='blue', alpha=0.1)
    
    # Calculate and plot trend lines to show improvement over time
    if len(df) > 2:
        x = np.array(df.index)
        
        # Predicted weight trend
        pred_trend = np.polyfit(x, df['predicted_weight'], 1)
        pred_line = np.poly1d(pred_trend)
        ax.plot(x, pred_line(x), '--', color='blue', alpha=0.5, label='Prediction Trend')
        
        # Actual weight trend
        actual_trend = np.polyfit(x, df['actual_weight'], 1)
        actual_line = np.poly1d(actual_trend)
        ax.plot(x, actual_line(x), '--', color='green', alpha=0.5, label='Actual Trend')
        
        # Error trend
        error_trend = np.polyfit(x, abs(df['error']), 1)
        error_line = np.poly1d(error_trend)
        
        # Calculate error reduction percentage (negative slope means improvement)
        error_reduction = 0
        if np.mean(abs(df['error'])) > 0:
            error_reduction = -error_trend[0] / np.mean(abs(df['error'])) * 100
        
        if error_reduction > 0:
            trend_msg = f"Model is improving: Error reduced by {error_reduction:.1f}% over time"
        else:
            trend_msg = "Model needs more training data to improve"
            
        ax.text(0.02, 0.02, trend_msg, transform=ax.transAxes, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add average prediction score
    avg_score = df['score'].mean() if 'score' in df.columns else 0
    ax.text(0.02, 0.06, f"Average prediction score: {avg_score:.3f}", 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Add styling
    title = f"Prediction vs Actual Weight Over Time"
    if exercise:
        title += f" for {exercise}"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Workout Session', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Add x-axis labels with workout numbers
    ax.set_xticks(df.index)
    ax.set_xticklabels([f"{i+1}" for i in range(len(df))], rotation=45)
    
    plt.tight_layout()
    
    # Convert plot to base64 string for API response
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    
    return fig, img_str

def plot_score_distribution(feedback_history: List[Dict[str, Any]], 
                           exercise: Optional[str] = None) -> Tuple[plt.Figure, str]:
    """
    Generate a histogram of prediction scores to show model accuracy distribution
    
    Args:
        feedback_history: List of feedback entries from the model
        exercise: Optional filter for a specific exercise
        
    Returns:
        Tuple of (matplotlib figure, base64 encoded image string)
    """
    df = create_feedback_dataframe(feedback_history)
    
    if df.empty or 'score' not in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No score data available for visualization", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return fig, img_str
    
    # Filter by exercise if specified
    if exercise:
        df = df[df['exercise'] == exercise]
        if df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"No score data available for exercise: {exercise}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return fig, img_str
    
    # Create figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram of scores
    bins = np.linspace(-1, 1, 21)  # 20 bins from -1 to 1
    ax.hist(df['score'], bins=bins, alpha=0.7, color='blue', edgecolor='black')
    
    # Add a vertical line at zero (perfect prediction)
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='Perfect Prediction')
    
    # Calculate mean score and add vertical line
    mean_score = df['score'].mean()
    ax.axvline(x=mean_score, color='red', linestyle='-', alpha=0.7, label=f'Mean Score: {mean_score:.3f}')
    
    # Add styling
    title = "Distribution of Prediction Scores"
    if exercise:
        title += f" for {exercise}"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Score (negative = over-prediction, positive = under-prediction)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Convert plot to base64 string for API response
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    
    return fig, img_str

def plot_improvement_over_time(feedback_history: List[Dict[str, Any]], 
                             exercise: Optional[str] = None,
                             window_size: int = 5) -> Tuple[plt.Figure, str]:
    """
    Generate a plot showing the model's prediction error over time with a moving average
    
    Args:
        feedback_history: List of feedback entries from the model
        exercise: Optional filter for a specific exercise
        window_size: Size of the moving average window
        
    Returns:
        Tuple of (matplotlib figure, base64 encoded image string)
    """
    df = create_feedback_dataframe(feedback_history)
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No feedback data available for visualization", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return fig, img_str
    
    # Filter by exercise if specified
    if exercise:
        df = df[df['exercise'] == exercise]
        if df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"No feedback data available for exercise: {exercise}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            plt.close(fig)
            
            return fig, img_str
    
    # Calculate absolute error for each prediction
    df['abs_error'] = abs(df['predicted_weight'] - df['actual_weight'])
    
    # Calculate percent error
    df['percent_error'] = 100 * df['abs_error'] / df['actual_weight']
    
    # Calculate moving average of absolute error
    if len(df) >= window_size:
        df['error_ma'] = df['abs_error'].rolling(window=window_size).mean()
    else:
        df['error_ma'] = df['abs_error']
    
    # Create figure and axis for the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot absolute error
    ax.plot(df.index, df['abs_error'], 'o-', label='Absolute Error', color='red', alpha=0.5)
    
    # Plot moving average
    if len(df) >= window_size:
        ax.plot(df.index, df['error_ma'], 'o-', label=f'{window_size}-point Moving Average', 
                color='blue', linewidth=2)
    
    # Calculate and plot trend line to show improvement over time
    if len(df) > 2:
        x = np.array(df.index)
        y = df['abs_error'].values
        
        # Error trend
        error_trend = np.polyfit(x, y, 1)
        error_line = np.poly1d(error_trend)
        ax.plot(x, error_line(x), '--', color='green', alpha=0.7, label='Error Trend')
        
        # Calculate error reduction percentage (negative slope means improvement)
        error_reduction = 0
        if np.mean(y) > 0:
            error_reduction = -error_trend[0] / np.mean(y) * 100
        
        if error_reduction > 0:
            trend_msg = f"Model is improving: Error reduced by {error_reduction:.1f}% over time"
        else:
            trend_msg = "Model needs more training data to improve"
            
        ax.text(0.02, 0.02, trend_msg, transform=ax.transAxes, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add styling
    title = "Prediction Error Over Time"
    if exercise:
        title += f" for {exercise}"
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Workout Session', fontsize=12)
    ax.set_ylabel('Absolute Error (Weight Units)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add x-axis labels with workout numbers
    ax.set_xticks(df.index)
    ax.set_xticklabels([f"{i+1}" for i in range(len(df))], rotation=45)
    
    plt.tight_layout()
    
    # Convert plot to base64 string for API response
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    
    return fig, img_str

if __name__ == "__main__":
    # Test visualization with sample data
    sample_data = [
        {
            'exercise': 'Bench Press',
            'predicted_weight': 100,
            'actual_weight': 95,
            'score': -0.05,
            'timestamp': '2023-01-01'
        },
        {
            'exercise': 'Bench Press',
            'predicted_weight': 97,
            'actual_weight': 95,
            'score': -0.02,
            'timestamp': '2023-01-08'
        },
        {
            'exercise': 'Bench Press',
            'predicted_weight': 96,
            'actual_weight': 97.5,
            'score': 0.015,
            'timestamp': '2023-01-15'
        },
        {
            'exercise': 'Bench Press',
            'predicted_weight': 98,
            'actual_weight': 100,
            'score': 0.02,
            'timestamp': '2023-01-22'
        },
        {
            'exercise': 'Bench Press',
            'predicted_weight': 100,
            'actual_weight': 100,
            'score': 0,
            'timestamp': '2023-01-29'
        }
    ]
    
    # Generate and save test plots
    fig1, _ = plot_prediction_vs_actual(sample_data)
    fig1.savefig('test_prediction_vs_actual.png')
    
    fig2, _ = plot_score_distribution(sample_data)
    fig2.savefig('test_score_distribution.png')
    
    fig3, _ = plot_improvement_over_time(sample_data)
    fig3.savefig('test_improvement.png')
    
    print("Test plots generated and saved.")