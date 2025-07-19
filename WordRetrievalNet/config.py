#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import pathlib


## TODO point this pathlib.Path where your datasets are
seg_free_datasets = pathlib.Path.home()/"datasets"/"seg-free"

BH2M = seg_free_datasets/"BH2M"
GW = seg_free_datasets/"GW"
IAM = seg_free_datasets/"IAM"

data_cfg = [
    # {
    #     "name": "Konzilsprotokolle",
    #     "train_img_path": "/home/zhaopeng/WordSpottingDatasets/Konzilsprotokolle/gen/images/",
    #     "train_gt_path": "/home/zhaopeng/WordSpottingDatasets/Konzilsprotokolle/gen/labels/",
    #     "test_img_path": "/home/zhaopeng/WordSpottingDatasets/Konzilsprotokolle/test/images/",
    #     "test_gt_path": "/home/zhaopeng/WordSpottingDatasets/Konzilsprotokolle/test/labels/",
    # },
    {
        "name": "BH2M",
        "train_img_path": f"{BH2M}/gen/images/",
        "train_gt_path": f"{BH2M}/gen/labels/",
        "test_img_path": f"{BH2M}/test/images/",
        "test_gt_path": f"{BH2M}/test/labels/",
    },
    {
        "name": "IAM",
        "train_img_path": f"{IAM}/gen/images/",
        "train_gt_path": f"{IAM}/gen/labels/",
        "test_img_path": f"{IAM}/test/images/",
        "test_gt_path": f"{IAM}/test/labels/",
        "val_img_path": f"{IAM}/test/images/",
        "val_gt_path": f"{IAM}/test/labels/",
    },
]

data_cfg.extend([
    {
        "name": f"GW{i}",
        "train_img_path": f"{GW}/cv{i}/gen/images/",
        "train_gt_path": f"{GW}/cv{i}/gen/labels/",
        "val_img_path": f"{GW}/cv{i}/validation/images/",
        "val_gt_path": f"{GW}/cv{i}/validation/labels/",
        "test_img_path": f"{GW}/cv{i}/test/images/",
        "test_gt_path": f"{GW}/cv{i}/test/labels/",
    } for i in range(4)
])

OVERWRITE_THIS="TO BE OVERWRITTEN!"

global_cfg = {
    "data_cfg": OVERWRITE_THIS,
    "arch": {
        "backbone": "resnet50",
        "pre_trained": True,
    },
    "loss": {
        "weight_cls": 1.0,
        "weight_angle": 10.0,
        "weight_diou": 1.0,
        "weight_embed": 1.0,
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 1e-3,
            "weight_decay": 0.00005,
            "amsgrad": True,
        },
    },
    "lr_scheduler": {
        "type": "MultiStepLR",
        "args": {
            "milestones": [200],
            "gamma": 0.1,
        }
    },
    "trainer": {
        "seed": 2,
        "gpus": [0],
        "img_channel": 3,
        "input_size": 512,
        "long_size": 2048,
        "batch_size": 4,#32
        "num_workers": 3,#0,#4,
        "epochs": OVERWRITE_THIS,#50,#120,#50,
        "lr_step": [80, 100],
        "save_interval": 10,
        "display_interval": 100,
        "show_images_interval": 10,
        "resume_checkpoint": OVERWRITE_THIS,
        "finetune_checkpoint": OVERWRITE_THIS,
        "output_dir": OVERWRITE_THIS,
        "tensorboard": False,
        "metrics": "map",
    },
    "tester": {
        "img_channel": 3,
        "long_size": 2048,
        "output_dir": OVERWRITE_THIS,
        "cls_score_thresh": 0.9,
        "bbox_nms_overlap": 0.4,
        "query_nms_overlap": 0.9,
        "overlap_thresh": [0.25, 0.5],
        "distance_metric": "cosine",
    },
}


def pick_configuration(
    dataset_name,
    *,
    trainer_output_dir=None,
    trainer_resume_checkpoint=None,
    trainer_finetune_checkpoint=None,
    tester_output_dir=None,
    epochs=None,
):
    if trainer_output_dir is None:
        trainer_output_dir="output"
    if trainer_resume_checkpoint is None:
        trainer_resume_checkpoint=""
    else:
        trainer_resume_checkpoint=pathlib.Path(trainer_resume_checkpoint).expanduser()
    if trainer_finetune_checkpoint is None:
        trainer_finetune_checkpoint=""
    else:
        trainer_finetune_checkpoint=pathlib.Path(trainer_finetune_checkpoint).expanduser()
    if tester_output_dir is None:
        tester_output_dir="output"
    if epochs is None:
        epochs=50

    for data in data_cfg:
        if data["name"] == dataset_name:
            global_cfg["data_cfg"] = data
            break
    else:
        raise ValueError(f"The given dataset name {dataset_name} is an invalid choice")

    global_cfg["trainer"]["output_dir"] = trainer_output_dir
    global_cfg["trainer"]["resume_checkpoint"] = trainer_resume_checkpoint
    global_cfg["trainer"]["finetune_checkpoint"] = trainer_finetune_checkpoint
    global_cfg["trainer"]["epochs"] = epochs

    global_cfg["tester"]["output_dir"] = tester_output_dir

    return global_cfg
