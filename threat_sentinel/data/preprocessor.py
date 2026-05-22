"""
ThreatSentinel — Feature Preprocessor
Handles cleaning, normalization, and train/test splitting
for network intrusion detection datasets.

Dr. Oluwaseyi Paul Babalola, Ph.D.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split


class ThreatPreprocessor:
    """
    Production-grade preprocessor for network telemetry data.

    Handles infinite values, outlier-robust normalization,
    low-variance feature removal, and stratified splitting.
    Designed for one-class anomaly detection: trains only on
    normal (benign) traffic.
    """

    def __init__(self, scaler='robust', variance_threshold=0.01):
        """
        Parameters
        ----------
        scaler : str
            'robust' (recommended for network data with outliers) or 'standard'
        variance_threshold : float
            Remove features with variance below this threshold
        """
        self.scaler_type = scaler
        self.variance_threshold = variance_threshold
        self.scaler = None
        self.selector = None
        self.feature_names = None
        self.selected_features = None

    def fit(self, X_normal):
        """
        Fit preprocessor on normal (benign) traffic only.
        This is the correct approach for anomaly detection.

        Parameters
        ----------
        X_normal : pd.DataFrame or np.ndarray
            Normal traffic features only (no attack samples)
        """
        if isinstance(X_normal, pd.DataFrame):
            self.feature_names = X_normal.columns.tolist()
            X = X_normal.values.astype(np.float64)
        else:
            X = X_normal.astype(np.float64)

        # Clean: replace inf/nan with median
        X = self._clean(X)

        # Remove low-variance features
        self.selector = VarianceThreshold(threshold=self.variance_threshold)
        X = self.selector.fit_transform(X)

        if self.feature_names:
            mask = self.selector.get_support()
            self.selected_features = [
                f for f, m in zip(self.feature_names, mask) if m
            ]

        # Fit scaler on normal traffic
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        self.scaler.fit(X)
        return self

    def transform(self, X):
        """
        Apply fitted preprocessing pipeline to new data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray

        Returns
        -------
        np.ndarray : cleaned, variance-filtered, scaled features
        """
        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float64)
        else:
            X = X.astype(np.float64)

        X = self._clean(X)
        X = self.selector.transform(X)
        X = self.scaler.transform(X)
        return X

    def fit_transform(self, X_normal):
        return self.fit(X_normal).transform(X_normal)

    def _clean(self, X):
        """Replace inf and nan with column median."""
        X = np.where(np.isinf(X), np.nan, X)
        col_medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])
        return X

    @property
    def n_features_out(self):
        return self.selector.transform(
            np.zeros((1, len(self.feature_names or [0])))
        ).shape[1] if self.selector else None


def prepare_data(df, feature_cols=None, label_col='label',
                 test_size=0.3, random_state=42):
    """
    Split dataset into train (normal only) and test (mixed) sets.

    For anomaly detection, the model is trained exclusively on
    normal traffic and evaluated on the full test set.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list or None (auto-detected if None)
    label_col : str
    test_size : float
    random_state : int

    Returns
    -------
    dict with keys:
        X_train_normal, X_test, y_test,
        attack_types_test, preprocessor
    """
    # Auto-detect feature columns
    non_feature = {label_col, 'attack_type', 'source', 'risk_score',
                   'timestamp', 'entity_id'}
    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if c not in non_feature
                        and df[c].dtype in [np.float64, np.float32,
                                            np.int64, np.int32]]

    X = df[feature_cols]
    y = df[label_col].values
    attack_types = df['attack_type'].values if 'attack_type' in df.columns else None

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if attack_types is not None:
        at_array = np.array(attack_types)
        _, _, _, attack_types_test = train_test_split(
            X, at_array, test_size=test_size,
            stratify=y, random_state=random_state
        )
    else:
        attack_types_test = None

    # Extract only normal traffic for training
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]

    # Fit preprocessor on normal training data only
    preprocessor = ThreatPreprocessor(scaler='robust')
    preprocessor.fit(X_train_normal)

    X_train_normal_scaled = preprocessor.transform(X_train_normal)
    X_test_scaled = preprocessor.transform(X_test)

    print(f"Training set (normal only): {X_train_normal_scaled.shape}")
    print(f"Test set (mixed):           {X_test_scaled.shape}")
    print(f"Test attack rate:           {y_test.mean()*100:.1f}%")

    return {
        'X_train_normal': X_train_normal_scaled,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'attack_types_test': attack_types_test,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.selected_features,
    }
