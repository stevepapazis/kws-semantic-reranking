#!/usr/bin/env bash


TIME=$(date)
echo -e "Job started at" $TIME "\n"
echo -e "Job started at" $TIME "\n" 1>&2 

echo -e "Using" $(python -V) "from" $(which python) "\n"
echo -e "Using" $(python -V) "from" $(which python) "\n" 1>&2


DATASET="IAM"
OUTPUT="../../output/WordRetrievalNet/"
EPOCHS=100

python "../../WordRetrievalNet/train.py" -d $DATASET -e $EPOCHS -o $OUTPUT
