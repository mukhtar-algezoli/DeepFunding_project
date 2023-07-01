FROM python:3.7-slim-buster

RUN apt-get update && apt-get install -y build-essential
RUN pip3 install sagemaker-training
RUN pip3 install transformers sentence_transformers torch scikit-learn pandas numpy peft wandb datasets[s3]
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
