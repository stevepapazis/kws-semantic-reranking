#!/usr/bin/env bash


TIME=$(date)
echo -e "Job started at" $TIME "\n"
echo -e "Job started at" $TIME "\n" 1>&2 

echo -e "Using" $(python -V) "from" $(which python) "\n"
echo -e "Using" $(python -V) "from" $(which python) "\n" 1>&2


DATASET="iam"
DATASET_PATH="../../datasets"
MODEL="../../pretrained_models/segfreekws_iam_best.pt"
RESULT_OUTPUT="../../output/SegFreeKWS/predict_result_segfreekws_IAM.npy"

python "../../SegFreeKWS/evaluate_form_kws.py" --gpu_id 0 --iou_mode 0 --dataset_path $DATASET_PATH --dataset $DATASET --model_path $MODEL --result_output_path $RESULT_OUTPUT
