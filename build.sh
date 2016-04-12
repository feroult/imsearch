#!/bin/bash

FILE=$1
FILENAME=$(basename $FILE)
NAME="${FILENAME%.*}"
EXTENSION="${FILENAME##*.}"

g++ $FILE -o bin/$NAME `pkg-config --cflags --libs opencv`
