# CUDA image for workers and router. Build:  docker build -f deploy/Dockerfile -t llmserve .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt bitsandbytes pynvml
COPY llmserve ./llmserve
COPY bench ./bench
COPY configs ./configs
ENV TOKENIZERS_PARALLELISM=false HF_HOME=/models
EXPOSE 8000 8001
# default: a worker; override the command to run the router
CMD ["python", "-m", "llmserve.worker", "--host", "0.0.0.0", "--port", "8001", "--precision", "fp16"]
