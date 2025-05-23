# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
# Google Cloud Run expects the container to listen on port 8080
EXPOSE 8080

# Define environment variable
ENV PORT=8080

# Create a small wrapper script to run the app
RUN echo 'import Ultron\napp = Ultron.create_dash_app()\nserver = app.server' > wsgi.py

# Run the app when the container launches
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 wsgi:server
