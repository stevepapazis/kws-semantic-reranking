#!/usr/bin/env bash

for f in {0..3};
do
    echo "Downloading WordRetrievalNet pretrained on GW cv${f}..."
    wget -nc "https://github.com/stevepapazis/kws-semantic-reranking/releases/download/WordRetrievalNet_pretrained_on_GW_cv${f}/wordretrievalnet_gw_cv${f}.zip"
    
    unzip -o "wordretrievalnet_gw_cv${f}.zip"
    mkdir -p "../output/WordRetrievalNet/resnet50_GW${f}/checkpoint/"
    mv "WordRetrievalNet_best.pth" "../output/WordRetrievalNet/resnet50_GW${f}/checkpoint/WordRetrievalNet_best.pth"
done;


echo "Downloading WordRetrievalNet pretrained on IAM..."
wget -nc "https://github.com/stevepapazis/kws-semantic-reranking/releases/download/WordRetrievalNet_pretrained_on_IAM/wordretrievalnet_iam.zip"

unzip -o "wordretrievalnet_iam.zip"
mkdir -p "../output/WordRetrievalNet/resnet50_IAM/checkpoint/"
mv "WordRetrievalNet_best.pth" "../output/WordRetrievalNet/resnet50_IAM/checkpoint/WordRetrievalNet_best.pth"
