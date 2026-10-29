# EMCC

This repository provides the reference implementation of **EMCC (Evidence-Maturity Cascaded Classification)** for six-class encrypted-traffic classification on the **VPN subset of ISCXVPN2016**, referred to here as **ISCXVPN**. EMCC performs encrypted-traffic classification from progressively observed packets and integrates packet-, burst-, and flow-level evidence through a staged cascade with causal early exit.

The repository contains the model, leakage-aware preprocessing, training, and evaluation code. Raw traffic captures, processed data, pretrained weights, and experimental results are not included.

## Repository structure

```text
EMCC/
├── config/       # Experiment configuration and command-line arguments
├── data/         # ISCXVPN subset preprocessing and dataset loaders
├── loss/         # Classification and progressive-learning objectives
├── models/       # EMCC and its packet/burst/flow components
├── progressive/  # Progressive training, evaluation, and early exit
├── utils/        # Reproducibility, checkpoint, and metric utilities
└── train.py      # Main training and evaluation entry point
```

## Environment

The code was tested with Python 3.10 and the package versions listed in `requirements.txt`. A CUDA-enabled PyTorch installation is recommended for training, while CPU execution is supported for small-scale checks.

```bash
python -m pip install -r requirements.txt
```

## Dataset

The [ISCXVPN2016 dataset](https://www.unb.ca/cic/datasets/vpn.html) contains both VPN and non-VPN traffic. This repository's experiment uses **only its VPN subset (ISCXVPN)**, with six application categories: chat, email, file transfer, P2P, streaming, and VoIP. The experiment does not use the full VPN/non-VPN dataset or include browsing as a target class.

Download the raw PCAP files from the official dataset page and place

## Preprocessing

Preprocessing uses flow-group splitting, header anonymization, and causal features from the first 1–20 packets.

```bash
python data/preprocess_iscxvpn.py
```

By default, processed arrays and metadata are written to:

```text
datasets/ISCXVPN2016/processed/
```

Use `python data/preprocess_iscxvpn.py --help` to inspect alternative input/output paths and preprocessing options.

## Training

After preprocessing, run the default ISCXVPN subset experiment with:

```bash
python train.py
```

## Evaluation

To evaluate a trained checkpoint without retraining:

```bash
python train.py --eval-only --eval-checkpoint results/iscxvpn2016/best_model.pt
```

Evaluation reports accuracy, macro-F1, per-prefix metrics, the mean exit step, and causal online early-exit statistics. Evaluation files are generated locally under `results/` and are intentionally excluded from version control.


## Citation

Citation information for the EMCC paper will be added after publication. When using the dataset, please also cite the ISCXVPN2016 paper listed on the official dataset page.
