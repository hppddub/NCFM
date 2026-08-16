FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --no-cache-dir -e ".[dev,plots]"

CMD ["python", "-m", "pytest"]
