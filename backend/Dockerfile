# Base image with Miniconda
FROM continuumio/miniconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the environment.yml file to the current working directory
COPY environment.yml /app/environment.yml

# Verify that environment.yml is copied (useful for debugging)
RUN ls -l /app

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the environment
SHELL ["conda", "run", "-n", "face_recog", "/bin/bash", "-c"]

# Copy the application code to the container
COPY . /app

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["conda", "run", "-n", "face_recog", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

