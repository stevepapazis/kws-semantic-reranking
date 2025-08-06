#!/usr/bin/env bash

for f in {0..3};
do
    echo "Downloading WordRetrievalNet pretrained on GW cv${f}..."
    wget "https://github.com/stevepapazis/kws-semantic-reranking/releases/download/wordretrievalnet_gw_cv${f}/wordretrievalnet_gw_cv${f}.zip"
    
    unzip "wordretrievalnet_gw_cv${f}.zip"
    mkdir -p "../../output/WordRetrievalNet/resnet50_GW${f}/checkpoint/"
    mv "wordretrievalnet_gw_cv${f}/WordRetrievalNet_best.pth" "../../output/WordRetrievalNet/resnet50_GW${f}/checkpoint/WordRetrievalNet_best.pth"
done;


echo "Downloading WordRetrievalNet pretrained on IAM..."
wget "https://github.com/stevepapazis/kws-semantic-reranking/releases/download/wordretrievalnet_iam/wordretrievalnet_iam.zip"

unzip "wordretrievalnet_iam.zip"
mkdir -p "../../output/WordRetrievalNet/resnet50_IAM/checkpoint/"
mv "wordretrievalnet_iam/WordRetrievalNet_best.pth" "../../output/WordRetrievalNet/resnet50_IAM/checkpoint/WordRetrievalNet_best.pth"
