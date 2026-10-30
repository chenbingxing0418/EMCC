# EMCC

This repository provides the reference implementation of EMCC (Evidence-Maturity Cascaded Classification) for six-class encrypted-traffic classification on the VPN subset of ISCXVPN2016, referred to here as ISCXVPN.

## Dataset

Experiments use the **VPN subset of ISCXVPN2016 (ISCXVPN)** with six classes: chat, email, file transfer, P2P, streaming, and VoIP.

Download the corresponding VPN PCAP files from the [official dataset page](https://www.unb.ca/cic/datasets/vpn.html).

Preprocessing uses flow-group splitting, clears only source and destination IPv4 addresses, and computes causal features for packet prefixes 1 through 20.

## Usage

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python data/preprocess_iscxvpn.py
python train.py
```

Evaluate a trained checkpoint:

```bash
python train.py --eval-only --eval-checkpoint results/iscxvpn2016/best_model.pt
```

Configuration: `config/emcc_config.py`. Results: `results/iscxvpn2016/`.

Data, model weights, and experimental results are not included.
