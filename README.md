# Understanding-Data

## Virtual Environment
python -m venv venv   ### To Create a Virutual Environment
venv\Scripts\activate    ### To activate virtual environment

## Docker Run Commands
docker build -t flask-ngrok-app .
docker run -p 5000:5000 flask-ngrok-app
