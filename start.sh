#!/bin/bash
# start.sh
# Boot script for Render to run both FastAPI (backend) and Streamlit (frontend) on a single Free Tier Web Service.

# Start the FastAPI backend in the background
# Make sure it uses a separate port internally (e.g., 8001) so it doesn't conflict with Render's exposed port.
echo "Starting FastAPI Backend on internal port 8001..."
uvicorn app:app --host 0.0.0.0 --port 8001 &

# Wait a moment to ensure the backend starts up
sleep 3

# Start the Streamlit frontend in the foreground.
# Render requires the main web service process to bind to the $PORT environment variable.
echo "Starting Streamlit Frontend on Render's external port ($PORT)..."
export API_BASE_URL="http://localhost:8001"
streamlit run ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
