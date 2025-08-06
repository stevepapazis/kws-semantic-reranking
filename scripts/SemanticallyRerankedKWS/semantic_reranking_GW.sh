#!/usr/bin/env bash


TIME=$(date)
echo -e "Job started at" $TIME "\n"
echo -e "Job started at" $TIME "\n" 1>&2 

echo -e "Using" $(python -V) "from" $(which python) "\n"
echo -e "Using" $(python -V) "from" $(which python) "\n" 1>&2


DATASET="GW"
DATASET_PATH="../../datasets"
GROUND_TRUTH="${DATASET_PATH}/GW/ground_truth/"

SEGFREEKWS_OUTPUT="../../output/SegFreeKWS"
WRN_OUTPUT="../../output/WordRetrievalNet"
KWS_RESULTS="../../output/SemanticallyRerankedKWS"


### Parses the KWS results and saves them as dataframes at $KWS_RESULTS 
python "../../SemanticallyRerankedKWS/semantically_reranked_kws.py" --ground-truth-location $GROUND_TRUTH --wrn-result-location $WRN_OUTPUT --segfreekws-result-location $SEGFREEKWS_OUTPUT --save-results-location $KWS_RESULTS --avoid-cache --dataset $DATASET --skip-mAP-computation ##--skip-SegFreeKWS-mAP-computation ##--skip-WRN-mAP-computation


SEGFREEKWS_MODULE="../../SegFreeKWS/"   

### Compute the semantic similarity 
for FOLD in {0..3};
do
    echo -e "~~~~~~~~~~~~~~CV ${FOLD}~~~~~~~~~~~~~~"
    echo -e "~~~~~~~~~~~~~~CV ${FOLD}~~~~~~~~~~~~~~" 1>&2  
     
    TROCR_MODEL="../../pretrained_models/trocr_finetuned_gw_cv$FOLD"
    SEGFREEKWS_MODEL="../../pretrained_models/segfreekws_gw_cv${FOLD+1}_best.pt"
    
    python "../../SemanticallyRerankedKWS/produce_semantic_embeddings.py" --path-to-segfreekws-module $SEGFREEKWS_MODULE --path-to-kws-results $KWS_RESULTS --path-to-gw $DATASET_PATH --test-fold $FOLD --path-to-trocr $TROCR_MODEL --path-to-segfreekws-decoder $SEGFREEKWS_MODEL --avoid-cache
done


### Compute mAP for each case
python "../../SemanticallyRerankedKWS/semantically_reranked_kws.py" --ground-truth-location $GROUND_TRUTH --wrn-result-location $WRN_OUTPUT --segfreekws-result-location $SEGFREEKWS_OUTPUT --save-results-location $KWS_RESULTS --keep-topk 30 --dataset $DATASET
