#!/bin/bash

FILE=$1
FILENAME=$(basename $FILE)
NAME="${FILENAME%.*}"
EXTENSION="${FILENAME##*.}"

clang $FILE -o bin/$NAME -L"/usr/local/src/ccv/lib" -I"/usr/local/src/ccv/lib" -lccv `cat /usr/local/src/ccv/lib/.deps`
