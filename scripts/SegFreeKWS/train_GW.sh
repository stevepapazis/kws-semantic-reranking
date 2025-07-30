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
    MODEL="../../pretrained_models/segfreekws_gw_cv${FOLD}.pt"
    BEST_MODEL="../../pretrained_models/segfreekws_gw_cv${FOLD}_best.pt"
    RESULT_OUTPUT="../../output/SegFreeKWS/predict_result_segfreekws_GW${FOLD}.npy"
    EPOCHS=100
    
    python "../../SegFreeKWS/train_words.py" --gpu_id 0 --test_fold $FOLD --dataset_path $DATASET_PATH --dataset $DATASET --model_save_path $MODEL --max_epochs $EPOCHS --learning_rate 1e-3 --batch_size 64 --result_output_path $RESULT_OUTPUT
    python "../../SegFreeKWS/evaluate_form_kws.py" --gpu_id 0 --test_fold $FOLD --iou_mode 0 --dataset_path $DATASET_PATH --dataset $DATASET --model_path $BEST_MODEL --result_output_path $RESULT_OUTPUT
done
