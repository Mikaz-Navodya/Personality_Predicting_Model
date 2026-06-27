# Use slim Python 3.11 to match modern Scikit-Learn C-compiler wheels
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Keep Python from buffering stdout/stderr (crucial for cloud Docker logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install required system dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server script AND both machine learning binaries
COPY main.py .
COPY preprocessor.pkl .
COPY personality_model.pkl .

# Expose standard container port
EXPOSE 8000

# Fire up Uvicorn tied to the container's internal network gateway
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]