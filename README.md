# Image Search Samples

## Installing

#### Docker

Build the OpenCV Docker Image:

    git clone https://github.com/feroult/imsearch.git
    docker build -t opencv3 .

#### For Linux Hosts

    xhost +
    docker run --rm -ti -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/imsearch -e DISPLAY=$DISPLAY opencv3

#### For OSX Hosts

    brew install socat
    socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"
    docker run --rm -ti -v $(pwd):/imsearch -e DISPLAY={HOST_IP}:0 opencv3

## Building and running a sample

    ./build.sh src/cpp/surf_flann_matcher.cpp
    ./bin/surf_flann_matcher data/query-image.jpg data/image1.jpg

## TODO

* Some samples were writen when OpenCV 3.0 was still beta. They may not compile/run.
  We have to update them.

## Some Links

* http://answers.opencv.org/question/10459/surf-matching-against-a-database-of-images/
* http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
