#### Running Sample

```bash
cd lib
g++ `pkg-config --cflags --libs opencv` surf_flann_matcher.cpp -o surf_flann_matcher
./surf_flann_matcher ../data/query-image.jpg ../data/image1.jpg
```
## Docker

    docker build -t opencv3 .

### Linux

    docker run --rm -ti -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/imsearch -e DISPLAY=$DISPLAY opencv3

### OSX

    brew install socat
    socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"
    docker run --rm -ti -v $(pwd):/imsearch -e DISPLAY={HOST_IP}:0 opencv3

## Links

* INSTALL OSX:
  * Install ffmpeg following this guide: https://tech.enekochan.com/en/2012/07/27/install-opencv-2-4-2-with-ffmpeg-support-in-mac-os-x-10-8/
  * Then install opencv: http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/
* http://answers.opencv.org/question/10459/surf-matching-against-a-database-of-images/
* http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
