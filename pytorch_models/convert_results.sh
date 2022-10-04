#!/bin/bash

# Converts test results to csv through Pandas
# Arguments:
#   $1 Architecture: (Faster-RCNN, MobileNet, Resnet, YOLO)
#   $2 Dataset: (b-59-850, SEILS, Mus-Tradicional/...)
#   $3 Data augmentation: (0 1 2) 0=>no-aug; 1=>random-auto; 2=>random-auto-uniform-sampling
#   $4 Input file (with .txt extension)
#   $4 Temporary file (with .txt extension)
#   $5 Output file (with .csv extension)

ARCH=$1
DATASET=$2
DATA_AUG=$3
FILE_IN=$4
FILE_TMP=$5
FILE_OUT=$6

cat $FILE_IN | grep "MAP" > $FILE_TMP
python results_to_csv.py $ARCH $DATASET $DATA_AUG $FILE_TMP $FILE_OUT
# rm $FILE_TMP


