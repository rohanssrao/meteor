FROM python:3.7

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN python -c "from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel; GPT2Tokenizer.from_pretrained('gpt2-medium'); GPT2LMHeadModel.from_pretrained('gpt2-medium')"

COPY *.py .

ENTRYPOINT ["python3", "meteor.py"]
