FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

ENV LANGUAGE C.UTF-8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

#RUN pip install --upgrade setuptools wheel
#RUN python -m pip install --no-use-pep517 bcrypt
RUN apt-get update
RUN apt-get -y install gcc
ADD src/requirements.txt /root/recommendation_system/src/requirements.txt
RUN pip install -r /root/recommendation_system/src/requirements.txt

RUN pip install mlplatform-lib==8.4.0.0.16.0.0

ADD src /root/recommendation_system/src

WORKDIR /root/recommendation_system/src
