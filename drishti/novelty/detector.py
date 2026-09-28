from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class NoveltyDetector:
    """
    Out-of-Distribution (OOD) Radar Novelty and Anomaly Detector.

    Identifies previously uncataloged hostile emitter waveforms, unfamiliar PRFs,
    or unprecedented electronic attack modes by evaluating the Mahalanobis distance
    of observed pulse-train parameter vectors against pre-mission threat reference libraries.
    """

    FEATURE_NAMES = ["period", "duty_cycle", "power_dbm", "hop_diversity"]

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = float(threshold)
        self.mean: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.fitted: bool = False

    def fit_reference_library(self, reference_features: np.ndarray) -> None:
        """
        Fits reference library distribution from pre-mission known threat profiles.

        Args:
            reference_features: Array of shape (N_samples, 4) with columns:
                                [period, duty_cycle, power_dbm, hop_diversity]
        """
        X = np.asarray(reference_features, dtype=np.float64)
        assert X.ndim == 2 and X.shape[1] == 4, "Reference features must have shape (N, 4)."

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-4

        # Regularized covariance matrix to avoid singularity
        cov = np.cov(X, rowvar=False) + np.eye(4) * 1e-4
        self.inv_cov = np.linalg.pinv(cov)
        self.fitted = True

    def score_novelty(self, feature_vector: np.ndarray) -> Tuple[float, bool]:
        """
        Computes the Mahalanobis distance score and flags whether the emitter is novel.

        Returns:
            (mahalanobis_distance, is_novel_flag)
        """
        if not self.fitted or self.mean is None or self.inv_cov is None:
            # Unfitted baseline fallback: simple zero novelty
            return 0.0, False

        x = np.asarray(feature_vector, dtype=np.float64).flatten()
        diff = x - self.mean
        dist_sq = float(np.dot(np.dot(diff, self.inv_cov), diff))
        dist = float(np.sqrt(max(0.0, dist_sq)))

        is_novel = dist >= self.threshold
        return dist, is_novel

    def explain_novelty(self, feature_vector: np.ndarray) -> str:
        """
        Produces natural-language explanation of which feature drove the novelty detection.
        """
        dist, is_novel = self.score_novelty(feature_vector)
        if not is_novel:
            return f"Nominal threat signature (Mahalanobis D={dist:.2f} < threshold {self.threshold:.1f})."

        x = np.asarray(feature_vector, dtype=np.float64).flatten()
        # Find which feature had the highest z-score deviation
        z_scores = np.abs((x - self.mean) / self.std)
        top_feat_idx = int(np.argmax(z_scores))
        top_name = self.FEATURE_NAMES[top_feat_idx]

        return (
            f"🚨 NOVEL THREAT DETECTED (D={dist:.2f} >= {self.threshold:.1f}). "
            f"Primary anomaly in '{top_name}': observed {x[top_feat_idx]:.2f} "
            f"vs reference mean {self.mean[top_feat_idx]:.2f} (z-score: {z_scores[top_feat_idx]:.2f}σ)."
        )
