# Use a lightweight Python image as the base
FROM python:latest

# Set the working directory
WORKDIR /app

# Copy the Flask app files to the container
COPY app.py /app/

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Download ngrok binary
RUN apt-get update && apt-get install -y curl && \
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list && \
    apt-get update && apt-get install ngrok && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Expose the Flask application port
EXPOSE 5000

# Create an entrypoint script for starting both Flask and ngrok
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
