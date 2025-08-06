#!/usr/bin/env bash


TIME=$(date)
echo -e "Job started at" $TIME "\n"
echo -e "Job started at" $TIME "\n" 1>&2 

echo -e "Using" $(python -V) "from" $(which python) "\n"
echo -e "Using" $(python -V) "from" $(which python) "\n" 1>&2


for f in {0..3};
do
    echo -e "~~~~~~~~~~~~~~CV ${f}~~~~~~~~~~~~~~"
    echo -e "~~~~~~~~~~~~~~CV ${f}~~~~~~~~~~~~~~" 1>&2
        
    DATASET="GW${f}"
    OUTPUT="../../output/WordRetrievalNet"
    QUERIES="orders recruits december stores instructions and"
    TOPN=30
    python "../../WordRetrievalNet/predict.py" -d $DATASET -o $OUTPUT $QUERIES $TOPN
done
