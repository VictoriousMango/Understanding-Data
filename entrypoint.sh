#!/bin/bash

# Start the Flask app in the background
python app.py &

# Wait for the Flask app to start
sleep 3

# Start ngrok and expose the Flask app on port 5000
ngrok http 5000
