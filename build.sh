#!/bin/bash

FILE=$1
FILENAME=$(basename $FILE)
NAME="${FILENAME%.*}"
EXTENSION="${FILENAME##*.}"

g++ `pkg-config --cflags --libs opencv` $FILE -o bin/$NAME
