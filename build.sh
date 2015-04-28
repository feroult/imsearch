#!/bin/bash

(cd lib; g++ `pkg-config --cflags --libs opencv` surf_flann_matcher.cpp -o ../bin/surf_flann_matcher)
(cd lib; g++ `pkg-config --cflags --libs opencv` surf_descriptor.cpp -o ../bin/surf_descriptor)
