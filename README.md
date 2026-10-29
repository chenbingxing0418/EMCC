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

Download the raw PCAP files from the official dataset page and place **only the VPN captures for these six categories** under:

```text
datasets/ISCXVPN2016/raw/
```

The preprocessing script recursively discovers `.pcap` files and infers application labels from their filenames. **It does not automatically filter VPN versus non-VPN captures**: the `vpn_` filename marker is removed during label inference, and both types can receive the same application label. Keep non-VPN captures out of the input directory to reproduce the ISCXVPN subset experiment.

The `ISCXVPN2016` directory name is retained in the code for compatibility; it identifies the parent dataset and does not mean that the full dataset is used.

Neither the original dataset nor generated arrays are tracked by this repository. Please follow the dataset provider's terms and cite the original dataset paper when using ISCXVPN2016.

## Preprocessing

The default preprocessing configuration uses leakage-aware flow-group splitting, anonymizes environment-specific header fields, and constructs causal views for packet prefixes 1 through 20. Each prefix view is recomputed using only the packets available at that prefix.

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

The default configuration uses deterministic execution with seed 42, trains for 30 epochs, and writes checkpoints and evaluation artifacts to `results/iscxvpn2016/`. The main hyperparameters are defined in `config/emcc_config.py` and can be overridden with command-line arguments.

Use `python train.py --help` to list all available options.

## Evaluation

To evaluate a trained checkpoint without retraining:

```bash
python train.py --eval-only --eval-checkpoint results/iscxvpn2016/best_model.pt
```

Evaluation reports accuracy, macro-F1, per-prefix metrics, the mean exit step, and causal online early-exit statistics. Evaluation files are generated locally under `results/` and are intentionally excluded from version control.

## Reproducibility notes

- The default data split seed is 42.
- PCAP- or flow-group splitting should be used for reported results; random window splitting is retained only for controlled comparisons.
- Prefix features are recomputed from the first `K` packets to prevent future-packet leakage.
- Header anonymization is enabled by default.
- Results may vary slightly across hardware and PyTorch/CUDA versions.

## Citation

Citation information for the EMCC paper will be added after publication. When using the dataset, please also cite the ISCXVPN2016 paper listed on the official dataset page.

## License

No open-source license has been selected yet. Add the intended license before publicly releasing the repository.
