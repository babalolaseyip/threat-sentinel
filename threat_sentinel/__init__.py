"""ThreatSentinel — Autonomous Threat Detection Platform"""
from .data.loader import generate_cicids_sample, generate_unsw_sample, load_cicids, load_unsw
from .data.preprocessor import prepare_data, ThreatPreprocessor
from .models.detectors import IsolationForestDetector, OneClassSVMDetector, AutoencoderDetector, train_all_detectors, evaluate_all
from .risk.aggregator import RiskAggregator, RiskEvent, EntityRiskProfile
from .pipeline.streaming import EventStream, StreamingDetectionPipeline

__version__ = '0.1.0'
__author__ = 'Dr. Oluwaseyi Paul Babalola'
