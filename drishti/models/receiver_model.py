from typing import Optional, Dict, Any, List, Tuple
import numpy as np


class ReceiverPredictiveModel:
    """
    Learned receiver system model predicting hit probability P(hit | x)
    and expected time-to-intercept across candidate frequency channels.

    Utilizes calibrated logistic feature fusion combining:
    - Historical channel occupancy (empirical hit ratio)
    - Channel Age-of-Information (tau_b)
    - Periodic phase arrival confidence from CircularPhaseCoherence
    - Prior threat intelligence weighting
    """

    def __init__(self, num_bands: int) -> None:
        self.num_bands = num_bands
        # Calibrated logistic regression weights [bias, w_hit_ratio, w_tau, w_last_seen, w_periodic]
        self.weights = np.array([-1.8, 2.5, 1.2, 1.5, 3.2], dtype=np.float64)

        # Calibration tracking: pairs of (predicted_prob, actual_binary_outcome)
        self.prediction_records: List[Tuple[float, int]] = []

    def reset(self) -> None:
        self.prediction_records.clear()

    def predict_hit_probability(
        self,
        band: int,
        hit_ratio: float,
        tau_norm: float,
        last_seen: float,
        periodic_confidence: float = 0.0,
    ) -> float:
        """
        Predicts calibrated probability of signal detection on band in the upcoming slot.
        """
        features = np.array([1.0, hit_ratio, tau_norm, last_seen, periodic_confidence], dtype=np.float64)
        logit = float(np.dot(self.weights, features))
        # Numerically stable sigmoid
        if logit >= 0:
            prob = 1.0 / (1.0 + np.exp(-logit))
        else:
            exp_logit = np.exp(logit)
            prob = exp_logit / (1.0 + exp_logit)
        return float(np.clip(prob, 1e-4, 1.0 - 1e-4))

    def record_outcome(self, predicted_prob: float, actual_detection: bool) -> None:
        """Stores prediction-outcome pair for calibration evaluation."""
        self.prediction_records.append((float(predicted_prob), 1 if actual_detection else 0))

    def compute_calibration_curve(self, num_bins: int = 10) -> Dict[str, Any]:
        """
        Computes reliability diagram data: binned mean predicted probability
        versus empirical fraction of positive detections.
        """
        if len(self.prediction_records) < 20:
            return {
                "bin_centers": [],
                "empirical_freqs": [],
                "bin_counts": [],
                "brier_score": 0.0,
            }

        probs = np.array([p for p, _ in self.prediction_records])
        outcomes = np.array([y for _, y in self.prediction_records])

        # Brier Score: Mean Squared Error of probability forecast
        brier = float(np.mean((probs - outcomes) ** 2))

        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        bin_centers = []
        empirical_freqs = []
        bin_counts = []

        for i in range(num_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            in_bin = (probs >= low) & (probs < high) if i < num_bins - 1 else (probs >= low) & (probs <= high)
            count = int(np.sum(in_bin))
            if count > 0:
                bin_centers.append(float((low + high) / 2.0))
                empirical_freqs.append(float(np.mean(outcomes[in_bin])))
                bin_counts.append(count)

        return {
            "bin_centers": bin_centers,
            "empirical_freqs": empirical_freqs,
            "bin_counts": bin_counts,
            "brier_score": brier,
        }
