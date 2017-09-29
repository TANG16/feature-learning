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

RUN wget https://github.com/Itseez/opencv/archive/3.1.0.zip -O /opencv.zip\
&& unzip /opencv.zip -d / \
&& mkdir /opencv-3.1.0/cmake_binary \
&& cd /opencv-3.1.0/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DENABLE_AVX=ON \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
&& make install \
&& rm /opencv.zip \
&& rm -r /opencv-3.1.0

RUN pip install pandas
RUN apt-get -y install python-matplotlib
RUN pip install matplotlib

ADD . /app
WORKDIR /app

RUN pip install .
