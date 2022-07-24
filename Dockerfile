# Dockerfile
FROM debian:stable-20220711-slim
MAINTAINER Hiroaki Sano <abcanswers@gmail.com>
RUN apt-get update && apt-get install -y vim curl git
RUN curl -o "/opt/Miniconda3-py39_4.12.0-Linux-x86_64.sh" "https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh"
RUN bash "/opt/Miniconda3-py39_4.12.0-Linux-x86_64.sh" -b -p /opt/miniconda3
ENV PATH /opt/miniconda3/bin:$PATH
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
RUN cd /root/ && git clone --depth 1 "https://github.com/pongues/digeon1.git"
