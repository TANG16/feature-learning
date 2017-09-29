FROM tensorflow/tensorflow:0.7.1

RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get -y install python2.7-dev
RUN apt-get -y install python-pip
RUN apt-get -y install python-tk
RUN pip install numpy

RUN apt-get update && \
        apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        libavformat-dev \
        libpq-dev

RUN pip install pandas
RUN apt-get -y install python-matplotlib
RUN pip install matplotlib

ADD . /app
WORKDIR /app

RUN pip install .
