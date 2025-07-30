#!/usr/bin/env bash


TIME=$(date)
echo -e "Job started at" $TIME "\n"
echo -e "Job started at" $TIME "\n" 1>&2 

echo -e "Using" $(python -V) "from" $(which python) "\n"
echo -e "Using" $(python -V) "from" $(which python) "\n" 1>&2


DATASET="iam"
DATASET_PATH="../../datasets"
MODEL="../../pretrained_models/segfreekws_iam.pt"
RESULT_OUTPUT="../../output/SegFreeKWS/predict_result_segfreekws_IAM.npy"
EPOCHS=100

python "../../SegFreeKWS/train_words.py" --gpu_id 0 --dataset_path $DATASET_PATH --dataset $DATASET --model_save_path $MODEL --max_epochs $EPOCHS --learning_rate 1e-3 --batch_size 64 --result_output_path $RESULT_OUTPUT
