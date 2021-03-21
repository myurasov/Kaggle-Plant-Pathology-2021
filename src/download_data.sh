#!/bin/bash

DATA_DIR=`python -c "from src.config import c; print(c['DATA_DIR'])"`
DEST_DIR="${DATA_DIR}/src"

rm -rfv $DEST_DIR
mkdir -pv $DEST_DIR
cd $DEST_DIR

kaggle competitions download -c plant-pathology-2021-fgvc8

unzip plant-pathology-2021-fgvc8.zip
rm plant-pathology-2021-fgvc8.zip