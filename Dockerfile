# Use a base image with PyTorch and MONAI pre-installed
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy your project files into the container
COPY src /app/src

# Install dependencies from requirements.txt
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt -v

# Specify the command to run when the container starts
ENTRYPOINT ["python", "/app/src/codes/inference.py"]
