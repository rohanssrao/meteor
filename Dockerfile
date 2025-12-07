FROM python:3.12-slim

WORKDIR /app

# Install dependencies (CPU-only PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy script
COPY meteor.py .

# Pre-download the model
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)"

ENTRYPOINT ["python", "meteor.py"]
