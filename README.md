# Semantic Re-Ranking for Handwritten Keyword Spotting

Official repository for the code and datasets accompanying the paper:\
*[Enhancing Keyword Spotting via NLP-Based Re-Ranking: Leveraging Semantic Relevance Feedback in the Handwritten Domain](https://doi.org/10.3390/electronics14142900)*

Overview of the proposed *semantic relevance feedback* mechanism
![The proposed relevance feedback mechanism.](/relevance-feedback-mechanism.png "The proposed relevance feedback mechanism")


## 📁 Repository Structure

```
SegFreeKWS/                 # Fork of the baseline Segmentation-Free KWS-Simplified implementation, adapted for the GW and IAM datasets
SemanticallyRerankedKWS/    # Implementation of the semantic re-ranking mechanism
WordRetrievalNet/           # Fork of the baseline WordRetrievalNet implementation, adapted for the GW and IAM datasets
datasets/                   # Template directory layout for the GW and IAM datasets
output/                     # Destination folder for experimental outputs
pretrained_models/          # Pretrained models provided for reproducibility
scripts/                    # Helper scripts to train and evaluate the baselines, as well as to apply the re-ranking and evaluate it
```

## 🛠️ Usage

1. Install the dependencies using:
```
conda create --name rerankKWS --file requirements.txt
conda activate rerankKWS
```

2. Download the datasets from:
    - [George Washington (GW) Database](https://fki.tic.heia-fr.ch/databases/washington-database)
    - [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)

    After downloading, extract them into the `datasets/` folder, following the directory layout indicated by the placeholder files.

3. The `pretrained_models/` directory contains the Segmentation-Free KWS-Simplified pretrained models. Due to GitHub’s file size limitations, the remaining pretrained models are provided as part of the repository’s releases.

    For convenience, we provide scripts to automate the downloading of these models:
    - `pretrained_models/get_WordRetrievalNet_pretrained_models.sh`
    - `pretrained_models/get_TrOCR_models_finetuned_on_GW.sh`

4. The `scripts/` directory provides bash scripts to train and evaluate the two baseline models, as well as to run and evaluate the re-ranking scheme.

    If you plan to train WordRetrievalNet, make sure to augment the data beforehand using `scripts/setup_GW_folds.sh`.
    

## 📜 License
- This project is licensed under the [Creative Commons BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).


## 📖 Citation

- If you use this codebase or build upon the ideas presented in our work, please consider citing the following paper:
```
Papazis, S.; Giotis, A.P.; Nikou, C.
Enhancing Keyword Spotting via NLP-Based Re-Ranking: Leveraging Semantic Relevance Feedback in the Handwritten Domain.
Electronics 2025, 14, 2900. https://doi.org/10.3390/electronics14142900
```

- BibTeX entry:
```bibtex
@article{Papazis2025NLPReRanking,
  author  = {Papazis, S. and Giotis, A.P. and Nikou, C.},
  title   = {Enhancing Keyword Spotting via NLP-Based Re-Ranking: Leveraging Semantic Relevance Feedback in the Handwritten Domain},
  journal = {Electronics},
  volume  = {14},
  year    = {2025},
  number  = {14},
  pages   = {2900},
  doi     = {10.3390/electronics14142900}
}
```
