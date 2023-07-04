FROM pytorch/pytorch:latest

# FROM nvidia/cuda:11.8.0-base-ubuntu22.04

#Download an open source TensorFlow Docker image
# FROM tensorflow/tensorflow:latest-gpu-jupyter
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.5.0-gpu-py36-cu101-ubuntu16.04

# RUN apt-get update && apt-get install -y \
#     curl \
#     ca-certificates \
#     sudo \
#     git \
#     bzip2 \
#     libx11-6

RUN apt-get update && apt-get install -y build-essential


# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip3 install sagemaker-training

# # Copies the training code inside the container
# COPY train.py /opt/ml/code/train.py

# # Defines train.py as script entrypoint
# ENV SAGEMAKER_PROGRAM train.py

# RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl --upgrade && \
#     pip3 install torchvision --upgrade

# install libs
RUN pip3 install transformers sentence_transformers torch scikit-learn pandas numpy peft wandb datasets[s3]

ENV PYTHONUNBUFFERED=TRUE
# ENV PYTHONDONTWRITEBYTECODE=TRUE
# ENV PATH="/opt/program:${PATH}"

# # Set up the program in the image
# COPY decision_trees /opt/program
# WORKDIR /opt/program

ENTRYPOINT ["python3"]