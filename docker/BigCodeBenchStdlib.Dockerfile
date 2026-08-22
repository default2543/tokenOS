FROM python:3.12.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /usr/sbin/nologin evaluator

COPY --chown=evaluator:evaluator \
    docker/bigcodebench_evaluator.py /opt/tokenos/bigcodebench_evaluator.py

USER evaluator
ENTRYPOINT ["python", "/opt/tokenos/bigcodebench_evaluator.py"]
