"""
ThreatSentinel — Risk Aggregation Engine
Combines anomaly signals from multiple detectors into
explainable, dynamic risk scores per entity (user/device/session).

Risk levels:
  LOW      0 - 30   No action required
  MEDIUM   30 - 60  Monitor closely
  HIGH     60 - 80  Trigger step-up authentication
  CRITICAL 80 - 100 Block and alert

Dr. Oluwaseyi Paul Babalola, Ph.D.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ── Risk level thresholds ─────────────────────────────────────────────────────
RISK_LEVELS = {
    'LOW': (0, 30),
    'MEDIUM': (30, 60),
    'HIGH': (60, 80),
    'CRITICAL': (80, 100),
}

# ── Model weights (tuned for precision: OCSVM penalizes false positives more) ─
DEFAULT_WEIGHTS = {
    'isolation_forest': 0.35,
    'one_class_svm': 0.40,
    'autoencoder': 0.25,
}


def get_risk_level(score):
    """Map numeric score to risk level label."""
    for level, (lo, hi) in RISK_LEVELS.items():
        if lo <= score < hi:
            return level
    return 'CRITICAL'


def get_risk_color(level):
    """Return color string for risk level."""
    return {
        'LOW': '#2ecc71',
        'MEDIUM': '#f39c12',
        'HIGH': '#e67e22',
        'CRITICAL': '#e74c3c',
    }.get(level, '#e74c3c')


@dataclass
class RiskEvent:
    """A single scored security event."""
    event_id: str
    timestamp: datetime
    entity_id: str
    entity_type: str   # 'user', 'device', 'session'
    source_dataset: str
    model_scores: Dict[str, float]
    composite_score: float
    risk_level: str
    top_indicators: List[str]
    raw_features: Optional[np.ndarray] = None


@dataclass
class EntityRiskProfile:
    """Aggregated risk profile for a user/device/session."""
    entity_id: str
    entity_type: str
    current_score: float = 0.0
    peak_score: float = 0.0
    event_count: int = 0
    alert_count: int = 0
    risk_level: str = 'LOW'
    recent_events: List[RiskEvent] = field(default_factory=list)
    score_history: List[float] = field(default_factory=list)
    last_updated: Optional[datetime] = None

    def update(self, event: RiskEvent, decay_factor=0.85):
        """
        Update entity profile with new event.
        Applies exponential decay to prevent score inflation from stale signals.
        """
        # Exponential moving average with decay
        if self.event_count == 0:
            self.current_score = event.composite_score
        else:
            self.current_score = (
                decay_factor * self.current_score +
                (1 - decay_factor) * event.composite_score
            )

        self.peak_score = max(self.peak_score, event.composite_score)
        self.event_count += 1
        if event.risk_level in ('HIGH', 'CRITICAL'):
            self.alert_count += 1

        self.risk_level = get_risk_level(self.current_score)
        self.recent_events.append(event)
        self.score_history.append(self.current_score)
        self.last_updated = event.timestamp

        # Keep only last 100 events in memory
        if len(self.recent_events) > 100:
            self.recent_events = self.recent_events[-100:]


class RiskAggregator:
    """
    Real-time risk aggregation engine.

    Consumes scored events from multiple detectors,
    computes weighted composite risk scores, and maintains
    dynamic entity risk profiles.

    Parameters
    ----------
    weights : dict
        Per-model weights for ensemble scoring
    decay_factor : float
        Exponential decay applied to entity scores over time
    alert_threshold : float
        Minimum composite score to raise an alert
    """

    def __init__(self, weights=None, decay_factor=0.85,
                 alert_threshold=50.0):
        self.weights = weights or DEFAULT_WEIGHTS
        self.decay_factor = decay_factor
        self.alert_threshold = alert_threshold
        self.entity_profiles: Dict[str, EntityRiskProfile] = {}
        self.event_log: List[RiskEvent] = []
        self._event_counter = 0

    def score_event(self, model_scores: Dict[str, float],
                    entity_id: str = None,
                    entity_type: str = 'session',
                    source: str = 'unknown',
                    timestamp: datetime = None,
                    feature_names: List[str] = None,
                    raw_features: np.ndarray = None) -> RiskEvent:
        """
        Compute composite risk score for a single event.

        Parameters
        ----------
        model_scores : dict
            Raw [0,1] anomaly scores from each detector
        entity_id : str
        entity_type : str  ('user', 'device', 'session')
        source : str
        timestamp : datetime
        feature_names : list
        raw_features : np.ndarray

        Returns
        -------
        RiskEvent
        """
        self._event_counter += 1
        if timestamp is None:
            timestamp = datetime.utcnow()
        if entity_id is None:
            entity_id = f"entity_{self._event_counter % 50}"

        # Weighted composite score scaled to 0-100
        total_weight = sum(
            self.weights.get(k, 1.0) for k in model_scores
        )
        composite = sum(
            score * self.weights.get(model, 1.0) / total_weight
            for model, score in model_scores.items()
        ) * 100.0
        composite = np.clip(composite, 0, 100)

        risk_level = get_risk_level(composite)

        # Top indicators (models contributing most to the score)
        top_indicators = sorted(
            model_scores.keys(),
            key=lambda m: model_scores[m] * self.weights.get(m, 1.0),
            reverse=True
        )[:3]

        event = RiskEvent(
            event_id=f"EVT-{self._event_counter:06d}",
            timestamp=timestamp,
            entity_id=entity_id,
            entity_type=entity_type,
            source_dataset=source,
            model_scores=model_scores,
            composite_score=float(composite),
            risk_level=risk_level,
            top_indicators=top_indicators,
            raw_features=raw_features,
        )

        # Update entity profile
        if entity_id not in self.entity_profiles:
            self.entity_profiles[entity_id] = EntityRiskProfile(
                entity_id=entity_id, entity_type=entity_type
            )
        self.entity_profiles[entity_id].update(event, self.decay_factor)
        self.event_log.append(event)

        return event

    def process_batch(self, detectors, X_test, y_test=None,
                      attack_types=None, source='unknown') -> pd.DataFrame:
        """
        Process a batch of events through all detectors.

        Parameters
        ----------
        detectors : dict of fitted BaseDetector instances
        X_test : np.ndarray
        y_test : np.ndarray (optional, for evaluation)
        attack_types : np.ndarray (optional)
        source : str

        Returns
        -------
        pd.DataFrame with one row per event
        """
        n = len(X_test)
        base_time = datetime(2026, 5, 20, 8, 0, 0)
        records = []

        # Get scores from all detectors
        all_scores = {
            name: det.normalized_scores(X_test)
            for name, det in detectors.items()
        }

        for i in range(n):
            model_scores = {name: float(scores[i])
                            for name, scores in all_scores.items()}
            entity_id = f"entity_{i % 50 + 1:03d}"
            entity_type = ['user', 'device', 'session'][i % 3]
            timestamp = base_time + timedelta(seconds=i * 2)

            event = self.score_event(
                model_scores=model_scores,
                entity_id=entity_id,
                entity_type=entity_type,
                source=source,
                timestamp=timestamp,
            )

            record = {
                'event_id': event.event_id,
                'timestamp': event.timestamp,
                'entity_id': event.entity_id,
                'entity_type': event.entity_type,
                'composite_score': event.composite_score,
                'risk_level': event.risk_level,
                **{f'score_{k}': v for k, v in model_scores.items()},
            }
            if y_test is not None:
                record['true_label'] = int(y_test[i])
            if attack_types is not None:
                record['attack_type'] = attack_types[i]

            records.append(record)

        return pd.DataFrame(records)

    def get_entity_summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all entity risk profiles."""
        rows = []
        for eid, profile in self.entity_profiles.items():
            rows.append({
                'entity_id': eid,
                'entity_type': profile.entity_type,
                'current_score': round(profile.current_score, 2),
                'peak_score': round(profile.peak_score, 2),
                'risk_level': profile.risk_level,
                'event_count': profile.event_count,
                'alert_count': profile.alert_count,
                'last_updated': profile.last_updated,
            })
        return (pd.DataFrame(rows)
                .sort_values('current_score', ascending=False)
                .reset_index(drop=True))

    def get_alerts(self, min_score=None) -> pd.DataFrame:
        """Return all HIGH and CRITICAL events."""
        threshold = min_score or self.alert_threshold
        alerts = [e for e in self.event_log
                  if e.composite_score >= threshold]
        if not alerts:
            return pd.DataFrame()
        return pd.DataFrame([{
            'event_id': e.event_id,
            'timestamp': e.timestamp,
            'entity_id': e.entity_id,
            'entity_type': e.entity_type,
            'composite_score': round(e.composite_score, 2),
            'risk_level': e.risk_level,
            'top_indicators': ', '.join(e.top_indicators),
            'source': e.source_dataset,
        } for e in alerts]).sort_values('composite_score', ascending=False)

    @property
    def detection_stats(self) -> dict:
        """Summary statistics for the current session."""
        if not self.event_log:
            return {}
        scores = [e.composite_score for e in self.event_log]
        levels = [e.risk_level for e in self.event_log]
        return {
            'total_events': len(self.event_log),
            'total_entities': len(self.entity_profiles),
            'alerts_raised': sum(1 for l in levels
                                  if l in ('HIGH', 'CRITICAL')),
            'mean_risk_score': round(np.mean(scores), 2),
            'max_risk_score': round(np.max(scores), 2),
            'level_counts': pd.Series(levels).value_counts().to_dict(),
        }
