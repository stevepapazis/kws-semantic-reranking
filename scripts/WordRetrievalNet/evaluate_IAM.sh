#!/usr/bin/env bash


TIME=$(date)
echo -e "Job started at" $TIME "\n"
echo -e "Job started at" $TIME "\n" 1>&2 

echo -e "Using" $(python -V) "from" $(which python) "\n"
echo -e "Using" $(python -V) "from" $(which python) "\n" 1>&2


DATASET="IAM"
OUTPUT="../../output/WordRetrievalNet"
QUERIES="labour meeting should peers"
TOPN=30

python "../../WordRetrievalNet/predict.py" -d $DATASET -o $OUTPUT $QUERIES $TOPN
