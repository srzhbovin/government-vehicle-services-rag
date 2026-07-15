FROM python:3.13-slim@sha256:bffeb7bd6a85767587059c6ba23e1e9122078e3aa3fa836099171b9bb5a9bb00

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY src/priority_gateway src/priority_gateway

ENTRYPOINT ["python", "-m", "priority_gateway.token_routing_benchmark"]
CMD ["--output-dir", "/app/output"]
