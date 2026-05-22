"""
ThreatSentinel - Anomaly Detection Models
Three complementary detection approaches:
  1. Isolation Forest   (tree-based, fast, robust in high dimensions)
  2. One-Class SVM      (kernel boundary around normal behavior)
  3. Autoencoder (PCA)  (reconstruction error detects anomalies)

Each model is trained exclusively on normal (benign) traffic.
Anomaly scores are normalized to [0, 1] for risk aggregation.

Dr. Oluwaseyi Paul Babalola, Ph.D.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (roc_auc_score, precision_score,
                              recall_score, f1_score,
                              average_precision_score)


class BaseDetector:
    """Base class with shared scoring and evaluation interface."""

    def __init__(self, name):
        self.name = name
        self.score_scaler = MinMaxScaler()
        self.fitted = False

    def fit(self, X_normal):
        raise NotImplementedError

    def anomaly_scores(self, X):
        raise NotImplementedError

    def normalized_scores(self, X):
        raw = self.anomaly_scores(X).reshape(-1, 1)
        return np.clip(self.score_scaler.transform(raw).flatten(), 0, 1)

    def predict(self, X, threshold=0.5):
        return (self.normalized_scores(X) >= threshold).astype(int)

    def evaluate(self, X_test, y_test, threshold=0.5):
        scores = self.normalized_scores(X_test)
        preds = (scores >= threshold).astype(int)
        return {
            'model': self.name,
            'precision': precision_score(y_test, preds, zero_division=0),
            'recall': recall_score(y_test, preds, zero_division=0),
            'f1': f1_score(y_test, preds, zero_division=0),
            'roc_auc': roc_auc_score(y_test, scores),
            'avg_precision': average_precision_score(y_test, scores),
            'threshold': threshold,
            'n_alerts': int(preds.sum()),
            'alert_rate': float(preds.mean()),
        }


class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest anomaly detector.

    Isolates anomalies by randomly partitioning the feature space.
    Anomalies require fewer partitions (shorter path length).
    Highly effective for high-dimensional network telemetry.
    """

    def __init__(self, n_estimators=200, contamination=0.05, random_state=42):
        super().__init__('Isolation Forest')
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_normal):
        self.model.fit(X_normal)
        raw = -self.model.score_samples(X_normal)
        self.score_scaler.fit(raw.reshape(-1, 1))
        self.fitted = True
        return self

    def anomaly_scores(self, X):
        return -self.model.score_samples(X)


class OneClassSVMDetector(BaseDetector):
    """
    One-Class SVM anomaly detector.

    Learns a tight decision boundary around normal traffic.
    Penalizes false positives more strictly than Isolation Forest.
    RBF kernel captures non-linear normal behavior manifolds.
    """

    def __init__(self, nu=0.05, kernel='rbf', gamma='scale'):
        super().__init__('One-Class SVM')
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit(self, X_normal):
        # Subsample for speed on large datasets
        n = min(len(X_normal), 5000)
        idx = np.random.choice(len(X_normal), n, replace=False)
        self.model.fit(X_normal[idx])
        raw = -self.model.decision_function(X_normal[idx])
        self.score_scaler.fit(raw.reshape(-1, 1))
        self.fitted = True
        return self

    def anomaly_scores(self, X):
        return -self.model.decision_function(X)


class AutoencoderDetector(BaseDetector):
    """
    PCA-based Autoencoder anomaly detector.

    Learns a low-dimensional reconstruction of normal traffic.
    Anomalous events produce high reconstruction error because
    the model was trained only on normal patterns.

    This implements the standard PCA autoencoder approach:
    normal data is projected to a lower-dimensional subspace
    and reconstructed; the residual error is the anomaly score.
    This is mathematically equivalent to a linear autoencoder
    and is widely used in production security analytics.

    Reference: Shyu et al., "A Novel Anomaly Detection Scheme
    Based on Principal Component Classifier", 2003.
    """

    def __init__(self, n_components=None, variance_explained=0.95,
                 random_state=42):
        """
        Parameters
        ----------
        n_components : int or None
            Number of PCA components. If None, set by variance_explained.
        variance_explained : float
            Target fraction of variance to retain (used if n_components=None)
        """
        super().__init__('Autoencoder (PCA)')
        self.n_components = n_components
        self.variance_explained = variance_explained
        self.random_state = random_state
        self.pca = None

    def fit(self, X_normal):
        """
        Fit PCA on normal traffic.
        The number of components is chosen to explain variance_explained
        fraction of variance in normal data.
        """
        n_comp = self.n_components or min(
            len(X_normal) - 1,
            X_normal.shape[1],
            50  # cap for speed
        )
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca.fit(X_normal)

        # Trim components to meet variance target
        cumvar = np.cumsum(self.pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumvar, self.variance_explained)) + 1
        self.pca.components_ = self.pca.components_[:n_keep]
        self.pca.n_components_ = n_keep

        # Calibrate score scaler on normal reconstruction errors
        raw = self._recon_errors(X_normal)
        self.score_scaler.fit(raw.reshape(-1, 1))
        self.fitted = True
        return self

    def _recon_errors(self, X):
        """Compute per-sample mean squared reconstruction error."""
        X_proj = self.pca.transform(X)
        X_recon = self.pca.inverse_transform(X_proj)
        return np.mean((X - X_recon) ** 2, axis=1)

    def anomaly_scores(self, X):
        return self._recon_errors(X)


def train_all_detectors(X_normal, verbose=True):
    """
    Train all three detectors on normal (benign) traffic only.

    Parameters
    ----------
    X_normal : np.ndarray
        Preprocessed normal traffic feature vectors
    verbose : bool

    Returns
    -------
    dict mapping detector key to fitted BaseDetector instance
    """
    detectors = {
        'isolation_forest': IsolationForestDetector(n_estimators=200),
        'one_class_svm': OneClassSVMDetector(nu=0.05),
        'autoencoder': AutoencoderDetector(variance_explained=0.90),
    }

    for name, detector in detectors.items():
        if verbose:
            print(f"  Training {detector.name}...")
        detector.fit(X_normal)
        if verbose:
            print(f"    Done.")

    return detectors


def evaluate_all(detectors, X_test, y_test, threshold=0.5):
    """
    Evaluate all detectors. Returns summary DataFrame.
    """
    results = []
    for detector in detectors.values():
        results.append(detector.evaluate(X_test, y_test, threshold))
    return pd.DataFrame(results).set_index('model')
