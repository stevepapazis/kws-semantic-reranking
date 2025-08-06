#!/usr/bin/env bash


TIME=$(date)
echo -e "Job started at" $TIME "\n"
echo -e "Job started at" $TIME "\n" 1>&2 

echo -e "Using" $(python -V) "from" $(which python) "\n"
echo -e "Using" $(python -V) "from" $(which python) "\n" 1>&2


for FOLD in {0..0};
do
    echo -e "~~~~~~~~~~~~~~CV ${FOLD}~~~~~~~~~~~~~~"
    echo -e "~~~~~~~~~~~~~~CV ${FOLD}~~~~~~~~~~~~~~" 1>&2  
    
    python "../../SemanticallyRerankedKWS/finetune_trocr.py" --path-to-gw "../../datasets/GW" --test-fold $FOLD --epochs 10 --output "../../output/trocr"
done
