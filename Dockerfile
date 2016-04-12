# To run a container based on the image, allowing GUI access (unsecured way):
# xhost +
# docker run --rm -ti -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY opencv3

FROM ubuntu:14.04

# Update packages
ENV LAST_OS_UPDATE 2015-10-15
RUN apt-get -y update
RUN apt-get -y upgrade

# Install x11-utils to get xdpyinfo, for X11 display
RUN apt-get -y install x11-utils mesa-utils

# Install OpenCV dependencies
RUN apt-get -y install build-essential cmake git pkg-config
RUN apt-get -y install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
RUN apt-get -y install libgtk2.0-dev
RUN apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get -y install libatlas-base-dev gfortran
RUN apt-get -y install python2.7-dev
RUN apt-get -y install wget

# PIP & Numpy
WORKDIR /usr/local/src
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
pip install numpy

# Checkout OpenCV contrib
WORKDIR /usr/local/src
RUN git clone --branch 3.0.0 https://github.com/Itseez/opencv_contrib
WORKDIR opencv_contrib
RUN rm -rf .git

# Compile OpenCV master branch from sources
WORKDIR /usr/local/src
RUN git clone --branch 3.0.0 https://github.com/Itseez/opencv.git
WORKDIR opencv
RUN rm -rf .git
RUN mkdir build
WORKDIR build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/usr/local/src/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..

RUN make -j2
RUN make install
RUN ldconfig

# Default command
WORKDIR /imsearch
CMD ["bash"]
