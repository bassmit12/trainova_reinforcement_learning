import uvicorn
import os
import sys

if __name__ == "__main__":
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Define host and port
    host = "0.0.0.0"
    port = 8001  # Using port 8001 to avoid conflict with neural network API
    
    print(f"Starting Feedback-Based Prediction API on http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    # Run the server
    uvicorn.run("api.main:app", host=host, port=port, reload=True)