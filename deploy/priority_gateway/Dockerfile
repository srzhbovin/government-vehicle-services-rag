FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY deploy/priority_gateway/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/priority_gateway src/priority_gateway
COPY deploy/priority_gateway/priorities.yaml deploy/priority_gateway/priorities.yaml

EXPOSE 8080
CMD ["uvicorn", "priority_gateway.app:app", "--host", "0.0.0.0", "--port", "8080"]
