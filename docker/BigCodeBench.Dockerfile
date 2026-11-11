# The digest pins the official linux/amd64 evaluator environment used by the
# archived BigCodeBench project. Docker Desktop runs it under emulation on ARM Macs.
FROM --platform=linux/amd64 bigcodebench/bigcodebench-evaluate@sha256:1327bddf60be9bc241648c59e6060cac4ca50248a0588ab735cd0200b17cc8c2

COPY --chown=bigcodebenchuser:bigcodebenchuser \
    docker/bigcodebench_evaluator.py /opt/tokenos/bigcodebench_evaluator.py

ENTRYPOINT ["python3", "/opt/tokenos/bigcodebench_evaluator.py"]
CMD []
