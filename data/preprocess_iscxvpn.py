"""
Leakage-aware ISCXVPN preprocessing with byte/packet/flow24/burst features.

The old window split is kept as --split-mode window for ablation only.
For paper results, prefer:
  - --split-mode pcap: split pcap files first, then windowize each split
  - --split-mode flow: split by pcap + five-tuple flow group after windowizing

Early-classification views are written to prefix_K subdirectories. Every
byte/packet/flow/burst feature in prefix_K is recomputed from only the first K
packets, so aggregate features do not leak later packets.

The raw byte view keeps the first N bytes from the IP packet. A companion
byte_header_payload view marks each byte position as padding/header/payload, so
models can use header-payload structure without replacing the original byte64
signal.

"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scapy.all import IP, TCP, UDP
from sklearn.model_selection import train_test_split
from tqdm import tqdm


MAX_PACKETS_PER_FLOW = 20
BYTE_FEATURE_SIZE = 64
PACKET_FEATURE_DIM = 12
FLOW_FEATURE_DIM = 24
MAX_BURSTS_PER_FLOW = 10
BURST_FEATURE_DIM = 10
DEFAULT_PREFIX_PACKETS = tuple(range(1, MAX_PACKETS_PER_FLOW + 1))
VIEW_KEYS = (
    "byte",
    "byte_header_payload",
    "packet",
    "flow",
    "burst",
    "labels",
    "observed_packets",
    "available_packets",
)
BYTE_REGION_PADDING = 0
BYTE_REGION_HEADER = 1
BYTE_REGION_PAYLOAD = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "ISCXVPN2016"

MAX_PKT_LEN = 1500.0
MAX_IAT = 10.0
MAX_BYTES = 30000.0

LABEL_RULES = {
    "streaming": ["youtube", "vimeo", "netflix"],
    "voip": ["voip", "skype", "hangouts"],
    "chat": ["aim", "icq", "facebook", "chat"],
    "filetransfer": ["ftps", "sftp", "scp", "file"],
    "p2p": ["bittorrent", "torrent", "p2p"],
    "email": ["email", "smtp", "pop", "imap"],
}


@dataclass
class PcapEntry:
    path: Path
    rel_path: str
    label: str


@dataclass
class WindowRecord:
    packets: list
    flow_key: tuple
    flow_key_text: str
    window_start: float
    window_index: int


def default_output_dir(byte_feature_size: int, split_mode: str, anonymize_header: bool = True) -> Path:
    """Return the repository-local output directory used by the default experiment."""
    del byte_feature_size, split_mode, anonymize_header
    return DEFAULT_DATASET_ROOT / "processed"


def parse_prefix_packets(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        prefixes = sorted({int(token.strip()) for token in value.split(",") if token.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--prefix-packets must be comma-separated integers") from exc
    invalid = [prefix for prefix in prefixes if prefix < 1 or prefix > MAX_PACKETS_PER_FLOW]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"--prefix-packets values must be between 1 and {MAX_PACKETS_PER_FLOW}: {invalid}"
        )
    return tuple(prefixes)


def parse_args():
    parser = argparse.ArgumentParser(description="Leakage-aware short-flow ISCXVPN preprocessing")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to datasets/ISCXVPN2016/processed "
            "under the repository root."
        ),
    )
    parser.add_argument("--window-size", type=float, default=60.0, help="Window size in seconds")
    parser.add_argument(
        "--byte-feature-size",
        type=int,
        default=BYTE_FEATURE_SIZE,
        help=(
            "Number of per-packet byte positions to store in *_byte.npy. "
            "The ISCXVPN EMCC config uses byte_head=64 by default."
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prefix-packets",
        type=parse_prefix_packets,
        default=DEFAULT_PREFIX_PACKETS,
        help=(
            "Comma-separated early packet budgets. Each prefix recomputes all features without future packets. "
            "Use an empty value to disable prefix output."
        ),
    )
    parser.add_argument(
        "--split-mode",
        choices=["window", "pcap", "flow"],
        default="flow",
        help="window=old random-window split; pcap=split pcaps first; flow=split by pcap+five-tuple flow",
    )
    parser.add_argument(
        "--anonymize-header",
        dest="anonymize_header",
        action="store_true",
        default=True,
        help=(
            "Clear environment-specific header fields before saving byte features. "
            "This keeps protocol, lengths, TCP flags, and payload bytes."
        ),
    )
    parser.add_argument(
        "--no-anonymize-header",
        dest="anonymize_header",
        action="store_false",
        help="Disable header anonymization for ablation only.",
    )
    args = parser.parse_args()
    if args.byte_feature_size <= 0:
        raise ValueError("--byte-feature-size must be positive")
    if args.output_dir is None:
        args.output_dir = default_output_dir(
            args.byte_feature_size,
            args.split_mode,
            anonymize_header=args.anonymize_header,
        )
    return args


def infer_label(filename: str) -> str | None:
    name = filename.lower().replace("vpn_", "")
    if any(kw in name for kw in ("youtube", "vimeo", "netflix", "spotify")):
        return "streaming"
    if any(kw in name for kw in ("bittorrent", "torrent", "p2p")):
        return "p2p"
    if any(kw in name for kw in ("email", "smtp", "pop", "imap")):
        return "email"
    if any(kw in name for kw in ("ftps", "sftp", "scp", "file", "files")):
        return "filetransfer"
    if "chat" in name or any(kw in name for kw in ("aim", "icq")):
        return "chat"
    if "audio" in name or "voip" in name:
        return "voip"
    for label, keywords in LABEL_RULES.items():
        if any(kw in name for kw in keywords):
            return label

def canonical_flow_key(pkt):
    if IP not in pkt:
        return None
    proto = 6 if TCP in pkt else (17 if UDP in pkt else None)
    if proto is None:
        return None
    src, dst = pkt[IP].src, pkt[IP].dst
    sp = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport
    dp = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport
    if (src, sp) > (dst, dp):
        src, dst, sp, dp = dst, src, dp, sp
    return (src, dst, int(sp), int(dp), int(proto))


def flow_key_to_text(key: tuple) -> str:
    return f"{key[0]}:{key[2]}-{key[1]}:{key[3]}-p{key[4]}"


def packet_header_length(pkt, raw_len: int) -> int:
    """Return IP + transport header length inside bytes(pkt[IP])."""
    ip_header_len = int(getattr(pkt[IP], "ihl", 5) or 5) * 4
    if TCP in pkt:
        transport_header_len = int(getattr(pkt[TCP], "dataofs", 5) or 5) * 4
    elif UDP in pkt:
        transport_header_len = 8
    else:
        transport_header_len = 0
    return int(np.clip(ip_header_len + transport_header_len, 0, raw_len))


def anonymize_packet_header(pkt, raw: bytearray) -> None:
    """Clear identifier fields while keeping protocol structure and payload."""
    if len(raw) < 20:
        return
    ip_header_len = int(getattr(pkt[IP], "ihl", 5) or 5) * 4
    ip_header_len = int(np.clip(ip_header_len, 20, len(raw)))

    raw[4:6] = b"\x00" * 2  # IP identification
    raw[8] = 0  # TTL is environment-dependent
    raw[10:12] = b"\x00" * 2  # IP checksum
    raw[12:20] = b"\x00" * 8  # source/destination IP

    if TCP in pkt:
        start = ip_header_len
        tcp_header_len = int(getattr(pkt[TCP], "dataofs", 5) or 5) * 4
        end = min(start + tcp_header_len, len(raw))
        if end - start < 20:
            return
        raw[start : start + 4] = b"\x00" * 4  # source/destination ports
        raw[start + 4 : start + 12] = b"\x00" * 8  # seq/ack numbers
        raw[start + 14 : start + 20] = b"\x00" * 6  # window/checksum/urgent pointer
        if end > start + 20:
            raw[start + 20 : end] = b"\x00" * (end - start - 20)  # TCP options
    elif UDP in pkt:
        start = ip_header_len
        end = min(start + 8, len(raw))
        if end - start < 8:
            return
        raw[start : start + 4] = b"\x00" * 4  # source/destination ports
        raw[start + 6 : start + 8] = b"\x00" * 2  # UDP checksum


def reassemble_flows_windowed(
    pcap_path: Path,
    window_size: float,
    anonymize_header: bool = True,
) -> list[WindowRecord]:
    """Return window records and keep each record's canonical five-tuple flow_key."""
    from scapy.all import PcapReader

    raw_flows: dict[tuple, list] = defaultdict(list)
    try:
        with PcapReader(str(pcap_path)) as reader:
            for pkt in reader:
                key = canonical_flow_key(pkt)
                if key is None:
                    continue
                raw = bytearray(bytes(pkt[IP]))
                if len(raw) < 40:
                    continue
                header_len = packet_header_length(pkt, len(raw))
                if anonymize_header:
                    anonymize_packet_header(pkt, raw)
                src = pkt[IP].src
                sp = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport
                direction = 1 if (src, sp) == (key[0], key[2]) else -1
                raw_flows[key].append((float(pkt.time), direction, bytes(raw), header_len))
    except Exception as exc:
        print(f"[warn] failed to read {pcap_path}: {exc}")

    samples: list[WindowRecord] = []
    for key, pkts in raw_flows.items():
        pkts.sort(key=lambda x: x[0])
        if not pkts:
            continue
        window_start = pkts[0][0]
        window_pkts = []
        window_index = 0
        for pkt in pkts:
            if pkt[0] - window_start > window_size:
                if len(window_pkts) >= 3:
                    samples.append(
                        WindowRecord(
                            packets=window_pkts[:MAX_PACKETS_PER_FLOW],
                            flow_key=key,
                            flow_key_text=flow_key_to_text(key),
                            window_start=window_start,
                            window_index=window_index,
                        )
                    )
                    window_index += 1
                window_pkts = []
                window_start = pkt[0]
            window_pkts.append(pkt)
        if len(window_pkts) >= 3:
            samples.append(
                WindowRecord(
                    packets=window_pkts[:MAX_PACKETS_PER_FLOW],
                    flow_key=key,
                    flow_key_text=flow_key_to_text(key),
                    window_start=window_start,
                    window_index=window_index,
                )
            )
    return samples


def build_bursts(packets: list) -> list[list]:
    pkts = packets[:MAX_PACKETS_PER_FLOW]
    if not pkts:
        return []

    bursts = []
    current_burst = [pkts[0]]
    for pkt in pkts[1:]:
        if pkt[1] == current_burst[-1][1]:
            current_burst.append(pkt)
        else:
            bursts.append(current_burst)
            if len(bursts) >= MAX_BURSTS_PER_FLOW:
                break
            current_burst = [pkt]
    if len(bursts) < MAX_BURSTS_PER_FLOW and current_burst:
        bursts.append(current_burst)
    return bursts[:MAX_BURSTS_PER_FLOW]


def packet_raw_bytes(pkt) -> bytes:
    return pkt[2]


def packet_header_len(pkt) -> int:
    return int(pkt[3]) if len(pkt) > 3 else len(pkt[2])


def extract_byte_feature(packets: list, byte_feature_size: int = BYTE_FEATURE_SIZE) -> np.ndarray:
    feat = np.zeros((MAX_PACKETS_PER_FLOW, byte_feature_size), dtype=np.float32)
    for i, pkt in enumerate(packets[:MAX_PACKETS_PER_FLOW]):
        raw_bytes = packet_raw_bytes(pkt)
        b = np.frombuffer(raw_bytes[:byte_feature_size], dtype=np.uint8).astype(np.float32)
        feat[i, :len(b)] = b / 255.0
    return feat


def extract_byte_header_payload_feature(packets: list, byte_feature_size: int = BYTE_FEATURE_SIZE) -> np.ndarray:
    feat = np.zeros((MAX_PACKETS_PER_FLOW, byte_feature_size), dtype=np.int64)
    for i, pkt in enumerate(packets[:MAX_PACKETS_PER_FLOW]):
        raw_bytes = packet_raw_bytes(pkt)
        valid_len = min(len(raw_bytes), byte_feature_size)
        if valid_len <= 0:
            continue
        header_len = min(max(packet_header_len(pkt), 0), valid_len)
        feat[i, :header_len] = BYTE_REGION_HEADER
        if header_len < valid_len:
            feat[i, header_len:valid_len] = BYTE_REGION_PAYLOAD
    return feat


def extract_packet_feature(packets: list) -> np.ndarray:
    pkts = packets[:MAX_PACKETS_PER_FLOW]
    n = len(pkts)
    timestamps = np.array([p[0] for p in pkts], dtype=np.float64)
    directions = np.array([p[1] for p in pkts], dtype=np.float32)
    pkt_lens = np.array([len(packet_raw_bytes(p)) for p in pkts], dtype=np.float32)
    payload_lens = pkt_lens.copy()
    iats = np.zeros(n, dtype=np.float32)
    if n > 1:
        iats[1:] = np.diff(timestamps).astype(np.float32)
    pkt_len_diff = np.zeros(n, dtype=np.float32)
    iat_diff = np.zeros(n, dtype=np.float32)
    if n > 1:
        pkt_len_diff[1:] = np.diff(pkt_lens)
        iat_diff[1:] = np.diff(iats)
    cum_bytes = np.cumsum(pkt_lens)
    cum_pkts = np.arange(1, n + 1, dtype=np.float32)
    up_cum = np.cumsum(np.where(directions == 1, pkt_lens, 0))
    up_ratio_cum = up_cum / np.maximum(cum_bytes, 1.0)
    iat_p75 = np.full(n, np.percentile(iats, 75) if n > 1 else 0.0, dtype=np.float32)
    feat = np.zeros((MAX_PACKETS_PER_FLOW, PACKET_FEATURE_DIM), dtype=np.float32)
    feat[:n, 0] = pkt_lens / MAX_PKT_LEN
    feat[:n, 1] = np.clip(iats, 0, MAX_IAT) / MAX_IAT
    feat[:n, 2] = directions
    feat[:n, 3] = payload_lens / MAX_PKT_LEN
    feat[:n, 4] = np.clip(pkt_len_diff, -MAX_PKT_LEN, MAX_PKT_LEN) / MAX_PKT_LEN
    feat[:n, 5] = np.clip(iat_diff, -MAX_IAT, MAX_IAT) / MAX_IAT
    feat[:n, 6] = cum_bytes / MAX_BYTES
    feat[:n, 7] = cum_pkts / MAX_PACKETS_PER_FLOW
    feat[:n, 8] = np.clip(iat_p75, 0, MAX_IAT) / MAX_IAT
    feat[:n, 9] = np.clip(payload_lens / np.maximum(pkt_lens, 1.0), 0, 1)
    feat[:n, 10] = up_ratio_cum
    feat[:n, 11] = (directions == 1).astype(np.float32)
    return feat


def extract_flow_feature(packets: list) -> np.ndarray:
    pkts = packets[:MAX_PACKETS_PER_FLOW]
    n = len(pkts)
    feat = np.zeros(FLOW_FEATURE_DIM, dtype=np.float32)
    if n == 0:
        return feat

    timestamps = np.array([p[0] for p in pkts], dtype=np.float64)
    directions = np.array([p[1] for p in pkts], dtype=np.float32)
    pkt_lens = np.array([len(packet_raw_bytes(p)) for p in pkts], dtype=np.float32)
    iats = np.diff(timestamps).astype(np.float32) if n > 1 else np.array([], dtype=np.float32)
    bursts = build_bursts(pkts)

    total_packets = float(n)
    total_bytes = float(pkt_lens.sum())
    up_mask = directions == 1
    down_mask = directions == -1
    up_packets = float(up_mask.sum())
    down_packets = float(down_mask.sum())
    up_bytes = float(pkt_lens[up_mask].sum()) if up_mask.any() else 0.0
    down_bytes = float(pkt_lens[down_mask].sum()) if down_mask.any() else 0.0
    direction_switch_count = float((directions[1:] != directions[:-1]).sum()) if n > 1 else 0.0
    direction_switch_rate = direction_switch_count / max(n - 1, 1)
    small_packet_ratio = float((pkt_lens < 200).mean())
    large_packet_ratio = float((pkt_lens > 1000).mean())

    mean_iat = float(iats.mean()) if iats.size > 0 else 0.0
    std_iat = float(iats.std()) if iats.size > 0 else 0.0
    cv_iat = min(std_iat / (mean_iat + 1e-6), 10.0)

    burst_packet_counts = np.array([len(b) for b in bursts], dtype=np.float32) if bursts else np.array([0.0], dtype=np.float32)
    burst_bytes = (
        np.array([sum(len(packet_raw_bytes(pkt)) for pkt in b) for b in bursts], dtype=np.float32)
        if bursts
        else np.array([0.0], dtype=np.float32)
    )
    burst_count = float(len(bursts))

    feat[0] = total_packets / MAX_PACKETS_PER_FLOW
    feat[1] = total_bytes / MAX_BYTES
    feat[2] = pkt_lens.mean() / MAX_PKT_LEN
    feat[3] = pkt_lens.std() / MAX_PKT_LEN
    feat[4] = pkt_lens.max() / MAX_PKT_LEN
    feat[5] = pkt_lens.min() / MAX_PKT_LEN
    feat[6] = np.percentile(pkt_lens, 25) / MAX_PKT_LEN
    feat[7] = np.percentile(pkt_lens, 75) / MAX_PKT_LEN
    feat[8] = np.clip(mean_iat, 0, MAX_IAT) / MAX_IAT
    feat[9] = np.clip(std_iat, 0, MAX_IAT) / MAX_IAT
    feat[10] = cv_iat / 10.0
    feat[11] = direction_switch_count / MAX_PACKETS_PER_FLOW
    feat[12] = direction_switch_rate
    feat[13] = small_packet_ratio
    feat[14] = large_packet_ratio
    feat[15] = up_packets / total_packets
    feat[16] = down_packets / total_packets
    feat[17] = up_bytes / max(total_bytes, 1.0)
    feat[18] = down_bytes / max(total_bytes, 1.0)
    feat[19] = burst_count / MAX_BURSTS_PER_FLOW
    feat[20] = burst_packet_counts.mean() / MAX_PACKETS_PER_FLOW
    feat[21] = burst_packet_counts.max() / MAX_PACKETS_PER_FLOW
    feat[22] = burst_bytes.mean() / MAX_BYTES
    feat[23] = burst_bytes.max() / MAX_BYTES
    return feat


def extract_burst_feature(packets: list) -> np.ndarray:
    feat = np.zeros((MAX_BURSTS_PER_FLOW, BURST_FEATURE_DIM), dtype=np.float32)
    bursts = build_bursts(packets)
    if not bursts:
        return feat

    for burst_idx, burst_pkts in enumerate(bursts):
        timestamps = np.array([p[0] for p in burst_pkts], dtype=np.float64)
        pkt_lens = np.array([len(packet_raw_bytes(p)) for p in burst_pkts], dtype=np.float32)
        iats = np.diff(timestamps).astype(np.float32) if len(burst_pkts) > 1 else np.array([], dtype=np.float32)
        burst_duration = float(timestamps[-1] - timestamps[0]) if len(burst_pkts) > 1 else 0.0
        burst_total_iat = float(iats.sum()) if iats.size > 0 else 0.0

        feat[burst_idx, 0] = float(burst_pkts[0][1])
        feat[burst_idx, 1] = len(burst_pkts) / MAX_PACKETS_PER_FLOW
        feat[burst_idx, 2] = pkt_lens.sum() / MAX_BYTES
        feat[burst_idx, 3] = np.clip(burst_duration, 0, MAX_IAT) / MAX_IAT
        feat[burst_idx, 4] = pkt_lens.mean() / MAX_PKT_LEN
        feat[burst_idx, 5] = pkt_lens.max() / MAX_PKT_LEN
        feat[burst_idx, 6] = pkt_lens.min() / MAX_PKT_LEN
        feat[burst_idx, 7] = np.clip(iats.mean() if iats.size > 0 else 0.0, 0, MAX_IAT) / MAX_IAT
        feat[burst_idx, 8] = np.clip(iats.std() if iats.size > 0 else 0.0, 0, MAX_IAT) / MAX_IAT
        feat[burst_idx, 9] = np.clip(burst_total_iat, 0, MAX_IAT) / MAX_IAT
    return feat


def label_sort_key(label: str) -> tuple[int, str]:
    labels = list(LABEL_RULES)
    return (labels.index(label) if label in LABEL_RULES else len(labels), label)


def build_label_map(labels: list[str]) -> dict[str, int]:
    return {label: idx for idx, label in enumerate(sorted(set(labels), key=label_sort_key))}


def collect_pcap_entries(input_dir: Path) -> list[PcapEntry]:
    entries = []
    for pcap_path in sorted(input_dir.rglob("*.pcap"), key=lambda p: str(p).lower()):
        label = infer_label(pcap_path.name)
        if label is None:
            continue
        try:
            rel_path = str(pcap_path.relative_to(input_dir))
        except ValueError:
            rel_path = pcap_path.name
        entries.append(PcapEntry(path=pcap_path, rel_path=rel_path, label=label))
    return entries


def split_items_by_label(items: list, labels: list[str], val_size: float, test_size: float, seed: int) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    by_label: dict[str, list] = defaultdict(list)
    for item, label in zip(items, labels):
        by_label[label].append(item)

    splits = {"train": [], "val": [], "test": []}
    for label in sorted(by_label, key=label_sort_key):
        label_items = list(by_label[label])
        rng.shuffle(label_items)
        n = len(label_items)
        if n == 1:
            n_val, n_test = 0, 0
        elif n == 2:
            n_test = 1 if test_size >= val_size else 0
            n_val = 1 - n_test
        else:
            n_test = max(1, int(round(n * test_size))) if test_size > 0 else 0
            n_val = max(1, int(round(n * val_size))) if val_size > 0 else 0
            while n_test + n_val >= n:
                if n_test >= n_val and n_test > 0:
                    n_test -= 1
                elif n_val > 0:
                    n_val -= 1
                else:
                    break
        splits["test"].extend(label_items[:n_test])
        splits["val"].extend(label_items[n_test:n_test + n_val])
        splits["train"].extend(label_items[n_test + n_val:])

    for split_items in splits.values():
        rng.shuffle(split_items)
    return splits


def random_window_split(labels: np.ndarray, val_size: float, test_size: float, seed: int) -> tuple[dict[str, np.ndarray], str]:
    idx = np.arange(len(labels))
    tmp_size = test_size + val_size
    note = "stratified"
    try:
        idx_train, idx_tmp, y_train, y_tmp = train_test_split(
            idx,
            labels,
            test_size=tmp_size,
            stratify=labels,
            random_state=seed,
        )
        idx_val, idx_test, _, _ = train_test_split(
            idx_tmp,
            y_tmp,
            test_size=test_size / tmp_size,
            stratify=y_tmp,
            random_state=seed,
        )
    except ValueError as exc:
        note = f"non_stratified_fallback: {exc}"
        idx_train, idx_tmp = train_test_split(idx, test_size=tmp_size, random_state=seed)
        idx_val, idx_test = train_test_split(idx_tmp, test_size=test_size / tmp_size, random_state=seed)
    return {"train": idx_train, "val": idx_val, "test": idx_test}, note


def empty_view_buffers() -> dict[str, list]:
    return {key: [] for key in VIEW_KEYS}


def empty_buffers(prefix_packets: tuple[int, ...] = ()) -> dict:
    buffers = empty_view_buffers()
    buffers["meta"] = []
    buffers["prefixes"] = {prefix: empty_view_buffers() for prefix in prefix_packets}
    return buffers


def append_view(
    buffers: dict[str, list],
    packets: list,
    label_id: int,
    byte_feature_size: int,
    available_packets: int,
):
    buffers["byte"].append(extract_byte_feature(packets, byte_feature_size))
    buffers["byte_header_payload"].append(extract_byte_header_payload_feature(packets, byte_feature_size))
    buffers["packet"].append(extract_packet_feature(packets))
    buffers["flow"].append(extract_flow_feature(packets))
    buffers["burst"].append(extract_burst_feature(packets))
    buffers["labels"].append(label_id)
    buffers["observed_packets"].append(len(packets))
    buffers["available_packets"].append(available_packets)


def append_feature(buffers: dict, record: WindowRecord, label_id: int, meta: dict, byte_feature_size: int):
    packets = record.packets[:MAX_PACKETS_PER_FLOW]
    available_packets = len(packets)
    append_view(buffers, packets, label_id, byte_feature_size, available_packets)
    buffers["meta"].append(meta)
    for prefix, prefix_buffers in buffers["prefixes"].items():
        append_view(
            prefix_buffers,
            packets[:prefix],
            label_id,
            byte_feature_size,
            available_packets,
        )


def copy_buffer_item(source: dict, target: dict, idx: int) -> None:
    for key in VIEW_KEYS:
        target[key].append(source[key][idx])
    target["meta"].append(source["meta"][idx])
    for prefix in source["prefixes"]:
        for key in VIEW_KEYS:
            target["prefixes"][prefix][key].append(source["prefixes"][prefix][key][idx])


def buffers_to_arrays(buffers: dict[str, list], byte_feature_size: int) -> dict[str, np.ndarray]:
    if buffers["labels"]:
        return {
            "byte": np.asarray(buffers["byte"], dtype=np.float32),
            "byte_header_payload": np.asarray(buffers["byte_header_payload"], dtype=np.int64),
            "packet": np.asarray(buffers["packet"], dtype=np.float32),
            "flow": np.asarray(buffers["flow"], dtype=np.float32),
            "burst": np.asarray(buffers["burst"], dtype=np.float32),
            "labels": np.asarray(buffers["labels"], dtype=np.int64),
            "observed_packets": np.asarray(buffers["observed_packets"], dtype=np.int64),
            "available_packets": np.asarray(buffers["available_packets"], dtype=np.int64),
        }
    return {
        "byte": np.zeros((0, MAX_PACKETS_PER_FLOW, byte_feature_size), dtype=np.float32),
        "byte_header_payload": np.zeros((0, MAX_PACKETS_PER_FLOW, byte_feature_size), dtype=np.int64),
        "packet": np.zeros((0, MAX_PACKETS_PER_FLOW, PACKET_FEATURE_DIM), dtype=np.float32),
        "flow": np.zeros((0, FLOW_FEATURE_DIM), dtype=np.float32),
        "burst": np.zeros((0, MAX_BURSTS_PER_FLOW, BURST_FEATURE_DIM), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
        "observed_packets": np.zeros((0,), dtype=np.int64),
        "available_packets": np.zeros((0,), dtype=np.int64),
    }


def class_distribution(labels: np.ndarray, label_names: list[str]) -> dict[str, int]:
    counts = np.bincount(labels.astype(np.int64), minlength=len(label_names)) if labels.size else np.zeros(len(label_names), dtype=np.int64)
    return {name: int(counts[idx]) for idx, name in enumerate(label_names)}


def save_split_arrays(
    output_dir: Path,
    split_buffers: dict[str, dict],
    label_names: list[str],
    byte_feature_size: int,
) -> dict:
    distributions = {}
    for split in ("train", "val", "test"):
        arrays = buffers_to_arrays(split_buffers[split], byte_feature_size)
        for name, value in arrays.items():
            np.save(output_dir / f"{split}_{name}.npy", value)
        distributions[split] = class_distribution(arrays["labels"], label_names)
        print(
            f"{split}: {len(arrays['labels'])} samples, "
            f"byte={tuple(arrays['byte'].shape)}, "
            f"byte_header_payload={tuple(arrays['byte_header_payload'].shape)}, "
            f"packet={tuple(arrays['packet'].shape)}, "
            f"flow={tuple(arrays['flow'].shape)}, burst={tuple(arrays['burst'].shape)}"
        )
        print(f"  class_distribution={distributions[split]}")

    prefix_packets = tuple(split_buffers["train"]["prefixes"])
    for prefix in prefix_packets:
        prefix_dir = output_dir / f"prefix_{prefix}"
        prefix_dir.mkdir(parents=True, exist_ok=True)
        print(f"prefix_{prefix}: recomputed from at most the first {prefix} packets")
        for split in ("train", "val", "test"):
            arrays = buffers_to_arrays(split_buffers[split]["prefixes"][prefix], byte_feature_size)
            root_labels = np.asarray(split_buffers[split]["labels"], dtype=np.int64)
            if not np.array_equal(arrays["labels"], root_labels):
                raise RuntimeError(f"prefix_{prefix}/{split} labels are not aligned with the root view")
            for name, value in arrays.items():
                np.save(prefix_dir / f"{split}_{name}.npy", value)
            short_count = int((arrays["available_packets"] < prefix).sum())
            print(
                f"  {split}: {len(arrays['labels'])} samples, "
                f"observed_mean={arrays['observed_packets'].mean() if len(arrays['labels']) else 0.0:.2f}, "
                f"shorter_than_budget={short_count}"
            )
    return distributions


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(value, fp, ensure_ascii=False, indent=2)


def write_metadata(
    args,
    output_dir: Path,
    label_map: dict[str, int],
    distributions: dict,
    split_info: dict,
):
    write_json(output_dir / "label_map.json", label_map)
    write_json(output_dir / "class_distribution.json", distributions)

    info = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "split_mode": args.split_mode,
        "window_size": args.window_size,
        "byte_feature_size": args.byte_feature_size,
        "anonymize_header": bool(args.anonymize_header),
        "anonymized_header_fields": (
            [
                "ipv4_source_address",
                "ipv4_destination_address",
                "ipv4_identification",
                "ipv4_ttl",
                "ipv4_checksum",
                "tcp_udp_source_port",
                "tcp_udp_destination_port",
                "tcp_udp_checksum",
                "tcp_sequence_number",
                "tcp_acknowledgement_number",
                "tcp_window",
                "tcp_urgent_pointer",
                "tcp_options",
            ]
            if args.anonymize_header
            else []
        ),
        "byte_header_payload_encoding": {
            "file": "{split}_byte_header_payload.npy",
            "shape": ["samples", MAX_PACKETS_PER_FLOW, args.byte_feature_size],
            "values": {
                str(BYTE_REGION_PADDING): "padding",
                str(BYTE_REGION_HEADER): "ip_plus_transport_header",
                str(BYTE_REGION_PAYLOAD): "transport_payload",
            },
            "note": (
                "The original byte view is unchanged and stores bytes(pkt[IP])[:byte_feature_size]. "
                "This companion view marks which byte positions belong to header or payload."
            ),
        },
        "test_size": args.test_size,
        "val_size": args.val_size,
        "seed": args.seed,
        "prefix_packets": list(args.prefix_packets),
        "prefix_protocol": (
            "For prefix_K, byte/packet/flow/burst features are recomputed from packets[:K]. "
            "Shorter samples keep all available packets and no samples are dropped."
        ),
        "label_map": label_map,
        "class_distribution": distributions,
    }
    info.update(split_info)
    write_json(output_dir / "split_info.json", info)

    for prefix in args.prefix_packets:
        prefix_dir = output_dir / f"prefix_{prefix}"
        prefix_info = dict(info)
        prefix_info.update(
            {
                "output_dir": str(prefix_dir),
                "data_view": f"prefix_{prefix}",
                "packet_budget": prefix,
            }
        )
        write_json(prefix_dir / "label_map.json", label_map)
        write_json(prefix_dir / "class_distribution.json", distributions)
        write_json(prefix_dir / "split_info.json", prefix_info)


def build_pcap_split(
    pcap_entries: list[PcapEntry],
    label_map: dict[str, int],
    args,
) -> tuple[dict[str, dict[str, list]], dict]:
    pcap_splits = split_items_by_label(
        pcap_entries,
        [entry.label for entry in pcap_entries],
        args.val_size,
        args.test_size,
        args.seed,
    )
    split_buffers = {split: empty_buffers(args.prefix_packets) for split in ("train", "val", "test")}
    pcap_assignments = []

    for split in ("train", "val", "test"):
        for entry in tqdm(pcap_splits[split], desc=f"process {split} pcaps"):
            before = len(split_buffers[split]["labels"])
            records = reassemble_flows_windowed(
                entry.path,
                args.window_size,
                anonymize_header=args.anonymize_header,
            )
            for record in records:
                append_feature(
                    split_buffers[split],
                    record,
                    label_map[entry.label],
                    {
                        "pcap": entry.rel_path,
                        "flow_key": record.flow_key_text,
                        "window_index": record.window_index,
                    },
                    args.byte_feature_size,
                )
            pcap_assignments.append(
                {
                    "pcap": entry.rel_path,
                    "label": entry.label,
                    "split": split,
                    "samples": len(split_buffers[split]["labels"]) - before,
                }
            )

    return split_buffers, {
        "leakage_control": "pcap files are split before window extraction",
        "pcap_assignments": pcap_assignments,
    }


def collect_all_windows(
    pcap_entries: list[PcapEntry],
    label_map: dict[str, int],
    args,
) -> dict[str, list]:
    buffers = empty_buffers(args.prefix_packets)
    for entry in tqdm(pcap_entries, desc="windowize pcaps"):
        records = reassemble_flows_windowed(
            entry.path,
            args.window_size,
            anonymize_header=args.anonymize_header,
        )
        for record in records:
            append_feature(
                buffers,
                record,
                label_map[entry.label],
                {
                    "pcap": entry.rel_path,
                    "label": entry.label,
                    "flow_key": record.flow_key_text,
                    "flow_group": f"{entry.rel_path}::{record.flow_key_text}",
                    "window_index": record.window_index,
                },
                args.byte_feature_size,
            )
    return buffers


def build_window_split(
    pcap_entries: list[PcapEntry],
    label_map: dict[str, int],
    args,
) -> tuple[dict[str, dict[str, list]], dict]:
    all_buffers = collect_all_windows(pcap_entries, label_map, args)
    labels = np.asarray(all_buffers["labels"], dtype=np.int64)
    index_splits, note = random_window_split(labels, args.val_size, args.test_size, args.seed)

    split_buffers = {split: empty_buffers(args.prefix_packets) for split in ("train", "val", "test")}
    for split, indices in index_splits.items():
        for idx in indices:
            copy_buffer_item(all_buffers, split_buffers[split], int(idx))

    return split_buffers, {
        "leakage_control": "none; window-level random split is kept only for old-baseline comparison",
        "split_note": note,
    }


def build_flow_split(
    pcap_entries: list[PcapEntry],
    label_map: dict[str, int],
    args,
) -> tuple[dict[str, dict[str, list]], dict]:
    all_buffers = collect_all_windows(pcap_entries, label_map, args)
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    group_to_label: dict[str, str] = {}
    group_to_pcap: dict[str, str] = {}
    group_to_flow: dict[str, str] = {}
    for idx, meta in enumerate(all_buffers["meta"]):
        group = meta["flow_group"]
        group_to_indices[group].append(idx)
        group_to_label[group] = meta["label"]
        group_to_pcap[group] = meta["pcap"]
        group_to_flow[group] = meta["flow_key"]

    groups = sorted(group_to_indices)
    group_splits = split_items_by_label(
        groups,
        [group_to_label[group] for group in groups],
        args.val_size,
        args.test_size,
        args.seed,
    )

    split_buffers = {split: empty_buffers(args.prefix_packets) for split in ("train", "val", "test")}
    group_assignments = []
    for split in ("train", "val", "test"):
        for group in group_splits[split]:
            indices = group_to_indices[group]
            group_assignments.append(
                {
                    "pcap": group_to_pcap[group],
                    "flow_key": group_to_flow[group],
                    "label": group_to_label[group],
                    "split": split,
                    "windows": len(indices),
                }
            )
            for idx in indices:
                copy_buffer_item(all_buffers, split_buffers[split], idx)

    return split_buffers, {
        "leakage_control": "all windows from the same pcap+flow_key group are assigned to one split",
        "flow_group_count": len(groups),
        "flow_group_assignments": group_assignments,
    }


def validate_no_group_crossing(split_buffers: dict[str, dict[str, list]], mode: str) -> dict:
    if mode == "window":
        return {"checked": False, "reason": "window split intentionally allows pcap/flow overlap"}
    key_name = "pcap" if mode == "pcap" else "flow_group"
    seen = {}
    violations = []
    for split, buffers in split_buffers.items():
        for meta in buffers["meta"]:
            key = meta[key_name]
            if key in seen and seen[key] != split:
                violations.append({"key": key, "first_split": seen[key], "second_split": split})
            else:
                seen[key] = split
    return {
        "checked": True,
        "group_key": key_name,
        "violation_count": len(violations),
        "violations": violations[:20],
    }


def main():
    args = parse_args()
    if args.test_size <= 0 or args.val_size <= 0 or args.test_size + args.val_size >= 1:
        raise ValueError("--test-size and --val-size must be positive and sum to less than 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pcap_entries = collect_pcap_entries(args.input_dir)
    if not pcap_entries:
        raise RuntimeError(f"No labeled pcap files found: {args.input_dir}")
    label_map = build_label_map([entry.label for entry in pcap_entries])
    label_names = [None] * len(label_map)
    for name, idx in label_map.items():
        label_names[idx] = name

    print(
        f"found {len(pcap_entries)} labeled pcap files, split_mode={args.split_mode}, "
        f"window={args.window_size}s, prefix_packets={list(args.prefix_packets)}, "
        f"anonymize_header={args.anonymize_header}"
    )
    print(f"label_map={label_map}")

    if args.split_mode == "pcap":
        split_buffers, split_info = build_pcap_split(pcap_entries, label_map, args)
    elif args.split_mode == "flow":
        split_buffers, split_info = build_flow_split(pcap_entries, label_map, args)
    else:
        split_buffers, split_info = build_window_split(pcap_entries, label_map, args)

    split_info["pcap_file_count"] = len(pcap_entries)
    split_info["group_crossing_check"] = validate_no_group_crossing(split_buffers, args.split_mode)
    distributions = save_split_arrays(args.output_dir, split_buffers, label_names, args.byte_feature_size)
    write_metadata(args, args.output_dir, label_map, distributions, split_info)
    print(f"saved to: {args.output_dir}")
    print("metadata: split_info.json, class_distribution.json, label_map.json")


if __name__ == "__main__":
    main()
