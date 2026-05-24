import numpy as np
from scipy import linalg


def extract_features(motion: np.ndarray) -> np.ndarray:
    N, T, J, C = motion.shape

    mean_pos = motion.mean(axis=1)
    std_pos = motion.std(axis=1)
    vel = np.diff(motion, axis=1)
    mean_vel = vel.mean(axis=1)

    feat = np.concatenate([
        mean_pos.reshape(N, -1),
        std_pos.reshape(N, -1),
        mean_vel.reshape(N, -1),
    ], axis=1)

    return feat.astype(np.float64)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fmd(real: np.ndarray, generated: np.ndarray) -> float:
    feat_r = extract_features(real)
    feat_g = extract_features(generated)

    mu_r, sigma_r = feat_r.mean(0), np.cov(feat_r, rowvar=False)
    mu_g, sigma_g = feat_g.mean(0), np.cov(feat_g, rowvar=False)

    return frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


def compute_mpjpe(real: np.ndarray, generated: np.ndarray) -> float:
    N_g, T, J, C = generated.shape
    N_r = real.shape[0]

    gen_flat = generated.reshape(N_g, -1)
    real_flat = real.reshape(N_r, -1)

    errors = []

    for g in gen_flat:
        dists = np.linalg.norm(real_flat - g, axis=1)
        best_idx = int(np.argmin(dists))

        diff = generated[errors.__len__()] - real[best_idx]
        per_joint = np.linalg.norm(diff, axis=-1).mean()

        errors.append(per_joint)

    return float(np.mean(errors))


def compute_mpjpe_v2(real: np.ndarray, generated: np.ndarray) -> float:
    mean_real = real.mean(axis=0)
    diff = generated - mean_real[None]

    return float(np.linalg.norm(diff, axis=-1).mean())


def compute_sample_variance(generated: np.ndarray) -> dict:
    N, T, J, C = generated.shape

    feat = extract_features(generated)

    pairwise = []

    for i in range(N):
        for j in range(i + 1, N):
            pairwise.append(np.linalg.norm(feat[i] - feat[j]))

    mean_pw = float(np.mean(pairwise)) if pairwise else 0.0

    joint_std = generated.std(axis=0).mean()

    vel = np.diff(generated, axis=1)
    vel_std = vel.std(axis=0).mean()

    return {
        "mean_pairwise_dist": mean_pw,
        "joint_position_std": float(joint_std),
        "velocity_std": float(vel_std),
    }
