# Use Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install streamlit requests

# Copy the frontend code
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
