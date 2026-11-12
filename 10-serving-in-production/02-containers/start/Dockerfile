# The gateway from lesson 10.4, as an image. Not run on the author's Mac (no Docker installed there);
# checked against the Dockerfile reference.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "fastapi==0.*" "uvicorn==0.*" "httpx==0.*"
COPY policy.py gateway.py ./
# A non-root user: the gateway needs no privileges.
RUN useradd --system app
USER app
EXPOSE 8090
CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8090"]
