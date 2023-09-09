FROM python:3.8

# Install Poetry
RUN pip install --upgrade pip && \
    pip install poetry

# Copy project files and install dependencies
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# Copy only the necessary files for application execution
COPY st_multi_batch.py .
COPY model.pkl .
COPY sads_logging.py .
COPY utils/ ./utils/
# Copy Streamlit config file
COPY .streamlit/config.toml .streamlit/config.toml


# Set environment variables
ENV PYTHONPATH=/app
EXPOSE 8501

# Set the entry point for the Docker container
ENTRYPOINT ["streamlit", "run", "--server.headless=true", "--server.port=8501", "st_multi_batch.py"]
