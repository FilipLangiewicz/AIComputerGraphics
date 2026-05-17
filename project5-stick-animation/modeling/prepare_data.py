import os
import numpy as np
from sklearn.model_selection import train_test_split

TRIMMED_ROOT = "trimmed_data"
OUT_DIR = "data"
FRAMES = 48
RANDOM_SEED = 42
LABEL_MAP = {"walk": 0, "jump": 1}
AUG_PER_CLASS = {0: 7, 1: 13}  # walk=7, jump=13

MIRROR_PAIRS = [
    (3, 6), (4, 7), (5, 8),
    (9, 12), (10, 13), (11, 14),
]


def get_subtype(fname: str, label: str) -> str:
    raw = fname.replace(".npy", "").rsplit("_", 2)[0]
    if label == "walk":
        return "slow_walk" if raw == "slow_walk" else "walk_rest"
    return "forward" if raw in ("forward_jump", "jumpforward") else raw


def resample_sequence(seq: np.ndarray, n_frames: int) -> np.ndarray:
    T = seq.shape[0]
    indices = np.linspace(0, T - 1, n_frames)
    left = np.floor(indices).astype(int)
    right = np.clip(left + 1, 0, T - 1)
    alpha = (indices - left)[:, None, None]
    return (1 - alpha) * seq[left] + alpha * seq[right]


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    center = seq[:, 2:3, :].mean(axis=0, keepdims=True)
    return seq - center


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    m = seq.copy()
    for a, b in MIRROR_PAIRS:
        m[:, a, :], m[:, b, :] = seq[:, b, :].copy(), seq[:, a, :].copy()
    m[:, :, 0] *= -1
    return m


def rotate_sequence(seq: np.ndarray, degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    r = seq.copy()
    x, y = seq[:, :, 0], seq[:, :, 1]
    r[:, :, 0] = c * x - s * y
    r[:, :, 1] = s * x + c * y
    return r


def random_augment(seq: np.ndarray, n: int, rng: np.random.Generator) -> list:
    samples = []
    for _ in range(n):
        angle = rng.uniform(0, 360)
        aug = rotate_sequence(seq, angle)
        if rng.random() < 0.5:
            aug = mirror_sequence(aug)
        samples.append(aug)
    return samples


def prepare_data():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    all_files, subtypes = [], []
    for label_name, label_idx in LABEL_MAP.items():
        folder = os.path.join(TRIMMED_ROOT, label_name)
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".npy"):
                all_files.append((os.path.join(folder, fname), label_idx))
                subtypes.append(f"{label_name}_{get_subtype(fname, label_name)}")

    train_files, test_files = train_test_split(
        all_files, test_size=0.2, random_state=RANDOM_SEED, stratify=subtypes
    )

    for split_name, split_files in [("train", train_files), ("test", test_files)]:
        sequences, labels = [], []
        for path, label in split_files:
            seq = np.load(path)
            seq = resample_sequence(seq, FRAMES)
            seq = normalize_sequence(seq)
            if split_name == "train":
                for aug in random_augment(seq, AUG_PER_CLASS[label], rng):
                    sequences.append(aug)
                    labels.append(label)
            else:
                sequences.append(seq)
                labels.append(label)

        sequences = np.stack(sequences).astype(np.float32)
        labels = np.array(labels, dtype=np.int64)
        np.savez(os.path.join(OUT_DIR, f"{split_name}.npz"), sequences=sequences, labels=labels)
        print(f"{split_name}: {sequences.shape}, labels: {np.bincount(labels)}")

