#!/usr/bin/env bash

for f in {0..3};
do
    echo "Downloading TrOCR finetuned on GW cv${f}..."
    wget -nc "https://github.com/stevepapazis/kws-semantic-reranking/releases/download/TrOCR_finetuned_on_GW_cv${f}/trocr_finetuned_on_GW_cv${f}.zip"
    unzip -n "trocr_finetuned_on_GW_cv${f}.zip"
done;
