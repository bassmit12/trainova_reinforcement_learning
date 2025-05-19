@echo off
echo Training Feedback-Based Prediction Model...
echo This will use feedback history to train the feedback model.
echo.

python -c "from data.preprocess_data import train_feedback_model; from models.feedback_prediction_model import FeedbackBasedPredictionModel; model = FeedbackBasedPredictionModel(); train_feedback_model(model.feedback_history)"

echo.
echo Training complete! The model is ready to use.
pause