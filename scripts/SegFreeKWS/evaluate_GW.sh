#!/usr/bin/env bash


TIME=$(date)
echo -e "Job started at" $TIME "\n"
echo -e "Job started at" $TIME "\n" 1>&2 

echo -e "Using" $(python -V) "from" $(which python) "\n"
echo -e "Using" $(python -V) "from" $(which python) "\n" 1>&2


for FOLD in {1..4};
do
    echo -e "~~~~~~~~~~~~~~CV ${FOLD}~~~~~~~~~~~~~~"
    echo -e "~~~~~~~~~~~~~~CV ${FOLD}~~~~~~~~~~~~~~" 1>&2
    DATASET="gw"
    DATASET_PATH="../../datasets"
    MODEL="../../pretrained_models/segfreekws_gw_cv${FOLD}_best.pt"
    RESULT_OUTPUT="../../output/SegFreeKWS/predict_result_segfreekws_GW${FOLD}.npy"
    
    python "../../SegFreeKWS/evaluate_form_kws.py" --gpu_id 0 --test_fold $FOLD --iou_mode 0 --dataset_path $DATASET_PATH --dataset $DATASET --model_path $MODEL --result_output_path $RESULT_OUTPUT
done
