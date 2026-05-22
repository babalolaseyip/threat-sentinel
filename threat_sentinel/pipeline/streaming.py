"""
ThreatSentinel — Streaming Detection Pipeline
Simulates near real-time ZTNA audit log ingestion and
event-by-event risk scoring, demonstrating the production
pipeline architecture without requiring Kafka/Flink.

Dr. Oluwaseyi Paul Babalola, Ph.D.
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Generator, Dict, Iterator
import queue
import threading


class EventStream:
    """
    Simulates a real-time ZTNA audit log event stream.

    In production, this would consume from a Kafka topic or
    a Kinesis stream. Here we replay pre-processed test events
    with configurable speed and batch size.
    """

    def __init__(self, X_test, y_test=None, attack_types=None,
                 events_per_second=50, batch_size=10):
        """
        Parameters
        ----------
        X_test : np.ndarray
            Pre-processed event feature vectors
        y_test : np.ndarray (optional)
            Ground truth labels for evaluation
        attack_types : np.ndarray (optional)
        events_per_second : int
            Simulated ingestion rate
        batch_size : int
            Events per mini-batch
        """
        self.X = X_test
        self.y = y_test
        self.attack_types = attack_types
        self.eps = events_per_second
        self.batch_size = batch_size
        self.n = len(X_test)
        self._base_time = datetime(2026, 5, 20, 8, 0, 0)

    def stream(self, simulate_delay=False) -> Generator:
        """
        Yield mini-batches of events as a generator.

        Parameters
        ----------
        simulate_delay : bool
            If True, sleep between batches to simulate real-time rate

        Yields
        ------
        dict with 'features', 'labels', 'attack_types', 'timestamps'
        """
        delay = self.batch_size / self.eps if simulate_delay else 0

        for start in range(0, self.n, self.batch_size):
            end = min(start + self.batch_size, self.n)
            batch_idx = slice(start, end)

            timestamps = [
                self._base_time + timedelta(seconds=(start + i) / self.eps)
                for i in range(end - start)
            ]

            batch = {
                'features': self.X[batch_idx],
                'labels': self.y[batch_idx] if self.y is not None else None,
                'attack_types': (self.attack_types[batch_idx]
                                  if self.attack_types is not None else None),
                'timestamps': timestamps,
                'batch_id': start // self.batch_size,
                'entity_ids': [f"entity_{(start + i) % 50 + 1:03d}"
                                for i in range(end - start)],
                'entity_types': [['user', 'device', 'session'][(start + i) % 3]
                                   for i in range(end - start)],
            }

            if simulate_delay and delay > 0:
                time.sleep(delay)

            yield batch

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


class StreamingDetectionPipeline:
    """
    End-to-end streaming detection pipeline.

    Architecture:
        EventStream -> FeatureProcessor -> [IF, OCSVM, AE] -> RiskAggregator

    Processes events in mini-batches, accumulates risk events,
    and surfaces high-fidelity threat signals.
    """

    def __init__(self, detectors, aggregator,
                 source='CICIDS-2017 + UNSW-NB15'):
        """
        Parameters
        ----------
        detectors : dict of fitted BaseDetector instances
        aggregator : RiskAggregator instance
        source : str
        """
        self.detectors = detectors
        self.aggregator = aggregator
        self.source = source
        self.processed_batches = 0
        self.processed_events = 0
        self.pipeline_log = []

    def process_stream(self, stream: EventStream,
                       simulate_delay=False,
                       max_batches=None,
                       verbose=True) -> pd.DataFrame:
        """
        Run the full detection pipeline over an event stream.

        Parameters
        ----------
        stream : EventStream
        simulate_delay : bool
        max_batches : int (optional, for demo/testing)
        verbose : bool

        Returns
        -------
        pd.DataFrame with all scored events
        """
        all_records = []
        start_time = time.time()

        for batch in stream.stream(simulate_delay=simulate_delay):
            if max_batches and self.processed_batches >= max_batches:
                break

            X_batch = batch['features']
            labels = batch['labels']
            attack_types = batch['attack_types']
            timestamps = batch['timestamps']
            entity_ids = batch['entity_ids']
            entity_types = batch['entity_types']

            # Score batch through all detectors
            model_scores_batch = {
                name: det.normalized_scores(X_batch)
                for name, det in self.detectors.items()
            }

            # Process each event in batch
            for i in range(len(X_batch)):
                model_scores = {
                    name: float(scores[i])
                    for name, scores in model_scores_batch.items()
                }

                event = self.aggregator.score_event(
                    model_scores=model_scores,
                    entity_id=entity_ids[i],
                    entity_type=entity_types[i],
                    source=self.source,
                    timestamp=timestamps[i],
                )

                record = {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'entity_id': event.entity_id,
                    'entity_type': event.entity_type,
                    'composite_score': round(event.composite_score, 2),
                    'risk_level': event.risk_level,
                    **{f'{k}_score': round(v, 4)
                       for k, v in model_scores.items()},
                }

                if labels is not None:
                    record['true_label'] = int(labels[i])
                if attack_types is not None:
                    record['attack_type'] = attack_types[i]

                all_records.append(record)

            self.processed_batches += 1
            self.processed_events += len(X_batch)

            if verbose and self.processed_batches % 10 == 0:
                elapsed = time.time() - start_time
                throughput = self.processed_events / elapsed
                stats = self.aggregator.detection_stats
                print(f"  Batch {self.processed_batches:4d} | "
                      f"Events: {self.processed_events:5d} | "
                      f"Throughput: {throughput:.0f} ev/s | "
                      f"Alerts: {stats.get('alerts_raised', 0)}")

        elapsed = time.time() - start_time
        if verbose:
            print(f"\nPipeline complete: {self.processed_events} events "
                  f"in {elapsed:.1f}s "
                  f"({self.processed_events/elapsed:.0f} events/sec)")

        return pd.DataFrame(all_records)

    def get_threat_summary(self) -> Dict:
        """Return threat detection summary for the current session."""
        stats = self.aggregator.detection_stats
        alerts = self.aggregator.get_alerts()
        entity_summary = self.aggregator.get_entity_summary()

        critical_entities = entity_summary[
            entity_summary['risk_level'] == 'CRITICAL'
        ]['entity_id'].tolist() if len(entity_summary) > 0 else []

        return {
            **stats,
            'n_critical_entities': len(critical_entities),
            'critical_entities': critical_entities[:10],
            'pipeline_events_processed': self.processed_events,
            'n_total_alerts': len(alerts),
        }
