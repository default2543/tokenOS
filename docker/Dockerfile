FROM python:3.12.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EVALPLUS_MAX_MEMORY_BYTES=402653184

RUN pip install --no-cache-dir --disable-pip-version-check evalplus==0.3.1

COPY docker/evaluator.py /opt/tokenos/evaluator.py

# Pin and precompute HumanEval+ data so runtime containers need no network or writes.
RUN python - <<'PY'
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth

problems = get_human_eval_plus(version="v0.1.10")
assert len(problems) == 164
get_groundtruth(problems, get_human_eval_plus_hash(version="v0.1.10"), [])
PY

ENTRYPOINT ["python", "/opt/tokenos/evaluator.py"]
