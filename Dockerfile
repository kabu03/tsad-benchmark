# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
# build-essential is often required for compiling Python packages (e.g. stumpy/numba dependencies)
# python3-tk is required for the GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-tk \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*


# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create the results directory just in case
RUN mkdir -p results

# Entrypoint to run the GUI by default
# Uses -m to ensure imports work correctly
ENTRYPOINT ["python", "-m", "src.gui"]

