### https://dev.to/et813/install-docker-and-nvidia-container-m0j
## According to the site,
## Make sure you have installed the NVIDIA driver and Docker 19.03 for your Linux distribution Note that you do not need to install the CUDA toolkit on the host, but the driver needs to be installed.
## ^^ See this for running GUIs

# FROM ubuntu:latest
# FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
# FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04
# ^^^ Didn't work when running iLQR.exe

# FROM nvidia/cuda:11.4.3-devel-ubuntu18.04
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 --no-cache-dir install --upgrade pip \
  && rm -rf /var/lib/apt/lists/*

#TMUX, git, net-tools, nano
RUN apt-get update
RUN apt-get install -y tmux git net-tools nano
#install 
RUN apt-get update

# Drake dependencies
RUN apt-get install -y --no-install-recommends \
      libpython3.8 libx11-6 libsm6 libxt6 libglib2.0-0

RUN apt-get install -y python3.8-venv python3-tk

#This installation supports gui in matplotlib
# https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so
# RUN apt-get install -y python3-tk

# Video recording
# RUN apt-get update
# RUN apt-get install -y ffmpeg

#install requirements
# COPY ./requirements.txt /tmp/requirements.txt
# RUN pip3 install -r /tmp/requirements.txt

# # Google Cloud CLI - https://cloud.google.com/sdk/docs/quickstart#deb
# RUN apt-get install -y gcc curl
# RUN apt-get install -y apt-transport-https ca-certificates gnupg
# RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y
# COPY ./potent-arcade-341204-013d447e4564.json /tmp/potent-arcade-341204-013d447e4564.json
# RUN gcloud auth activate-service-account --key-file=/tmp/potent-arcade-341204-013d447e4564.json
# # https://stackoverflow.com/questions/37428287/not-able-to-perform-gcloud-init-inside-dockerfile
# RUN gcloud config set project potent-arcade-341204
# RUN pip3 install --upgrade grpcio==1.43

# #google cli install
# # Google Cloud CLI - https://cloud.google.com/sdk/docs/quickstart#deb
# RUN apt-get install -y gcc curl
# RUN apt-get install -y apt-transport-https ca-certificates gnupg
# RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y
# COPY ./potent-arcade-341204-013d447e4564.json /tmp/potent-arcade-341204-013d447e4564.json
# RUN gcloud auth activate-service-account --key-file=/tmp/potent-arcade-341204-013d447e4564.json
# # https://stackoverflow.com/questions/37428287/not-able-to-perform-gcloud-init-inside-dockerfile
# RUN gcloud config set project potent-arcade-341204

# RUN pip3 install google-api-python-client
# RUN pip3 install cryptography
# ENV GOOGLE_APPLICATION_CREDENTIALS="/tmp/potent-arcade-341204-013d447e4564.json"

#Set Work dir
WORKDIR /root/Python_ws

CMD ["tail", "-f", "/dev/null"]
#ENTRYPOINT ["python3"]