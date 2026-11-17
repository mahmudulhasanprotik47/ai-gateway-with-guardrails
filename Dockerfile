# Small official Python image, same minor version as the local .venv.
FROM python:3.14-slim

# Don't write .pyc files (useless in a throwaway container); don't buffer stdout,
# so the one-JSON-line-per-/chat request log appears immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# All following paths are relative to /app inside the image.
WORKDIR /app

# Copy only the dependency list first, so this layer and the install below
# stay cached until requirements.txt itself changes.
COPY requirements.txt .

# Install runtime deps only; --no-cache-dir keeps pip's download cache out of the image.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code last: editing it re-runs only this cheap step.
COPY app/ ./app/

# Unprivileged account, so a compromised process doesn't run as root.
RUN useradd --create-home appuser

# Everything from here on, including CMD, runs as appuser.
USER appuser

# Documents the port the server listens on (publishing still needs -p).
EXPOSE 8000

# sh expands ${PORT:-8000} (Render sets PORT); exec makes uvicorn PID 1 so it gets
# SIGTERM directly. No --workers: the rate limiter is in-memory, single process only.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
