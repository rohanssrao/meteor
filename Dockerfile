FROM python:3.12-slim

WORKDIR /app

# Install dependencies (CPU-only PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy script
COPY meteor.py .

# Pre-download the model
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    revision='9d2be5519834990d30996f878b6771cccbd24f2c'; \
    AutoTokenizer.from_pretrained('LiquidAI/LFM2.5-230M-Base', revision=revision); \
    AutoModelForCausalLM.from_pretrained('LiquidAI/LFM2.5-230M-Base', revision=revision)"

ENTRYPOINT ["python", "meteor.py"]
