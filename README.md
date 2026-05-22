# ThreatSentinel — Autonomous Threat Detection Platform

**Dr. Oluwaseyi Paul Babalola, Ph.D.**  
Senior Member, IEEE | NRF Y2-Rated Researcher  
[Google Scholar](https://scholar.google.com/citations?user=z6viTLkAAAAJ) | [LinkedIn](https://www.linkedin.com/in/oluwaseyi-babalola-06384715)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://threat-sentinel.streamlit.app)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/babalolaseyip/threat-sentinel/HEAD?filepath=notebooks/ThreatSentinel_Demo.ipynb)

---

## Overview

ThreatSentinel is an end-to-end ML-powered threat detection platform that demonstrates production-grade anomaly detection and risk aggregation for network security telemetry. It implements the core detection and scoring components of a Zero Trust Network Access (ZTNA) security analytics pipeline.

This is a research portfolio prototype. The architecture and methods are designed to scale to production systems.

---

## Capabilities Demonstrated

### ML Anomaly Detection (Three Complementary Models)
- **Isolation Forest** — tree-based partitioning, fast on high-dimensional telemetry
- **One-Class SVM** — tight kernel boundary around normal behavior, low false positive rate
- **Autoencoder (PCA)** — reconstruction error detects novel attack patterns

### Risk Aggregation Engine
- Weighted ensemble scoring combining all three model signals
- Dynamic per-entity risk profiles (user / device / session)
- Exponential decay to prevent score inflation from stale signals
- Four risk tiers: LOW / MEDIUM / HIGH / CRITICAL

### Streaming Detection Pipeline
- Mini-batch event processing simulating near real-time ZTNA audit log ingestion
- Throughput: 50+ events/second on standard hardware
- Risk Sentinel integration layer for adaptive access enforcement

### Datasets
- **CICIDS-2017** (Canadian Institute for Cybersecurity) — 78 flow-based network features, 12 attack types
- **UNSW-NB15** (UNSW Sydney) — 49 features, 9 attack categories

---

## Project Structure

```
threat-sentinel/
├── threat_sentinel/
│   ├── data/
│   │   ├── loader.py          # CICIDS-2017 & UNSW-NB15 data loading
│   │   └── preprocessor.py    # Feature cleaning, normalization, splitting
│   ├── models/
│   │   └── detectors.py       # IF, One-Class SVM, Autoencoder
│   ├── risk/
│   │   └── aggregator.py      # Risk scoring, entity profiles, alerts
│   └── pipeline/
│       └── streaming.py       # Real-time streaming detection pipeline
├── dashboard/
│   └── app.py                 # Streamlit interactive dashboard
├── notebooks/
│   └── ThreatSentinel_Demo.ipynb
├── requirements.txt
└── README.md
```

---

## Quick Start

### Run in Browser (No Installation)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/babalolaseyip/threat-sentinel/HEAD?filepath=notebooks/ThreatSentinel_Demo.ipynb)

### Local Installation
```bash
git clone https://github.com/babalolaseyip/threat-sentinel.git
cd threat-sentinel
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/ThreatSentinel_Demo.ipynb

# Launch the dashboard
streamlit run dashboard/app.py
```

### Using Real Datasets
Download from official sources and use the provided loaders:
```python
from threat_sentinel.data.loader import load_cicids, load_unsw

# CICIDS-2017: https://www.unb.ca/cic/datasets/ids-2017.html
df_cicids = load_cicids('path/to/CICIDS2017.csv')

# UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
df_unsw = load_unsw('path/to/UNSW_NB15.csv')
```

---

## Detection Performance (CICIDS-2017, Synthetic Demo)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Isolation Forest | 0.21 | 0.87 | 0.34 | 0.74 |
| One-Class SVM | 0.24 | 0.87 | 0.37 | 0.92 |
| Autoencoder (PCA) | 0.36 | 0.87 | 0.51 | 0.93 |
| **Ensemble** | **0.31** | **0.87** | **0.46** | **0.91** |

*These results reflect unsupervised detection (no labels used in training), consistent with realistic production deployments where labelled attack data is unavailable.*

---

## Roadmap (Commercial Version)

- Deep learning detectors: Transformer-based sequence models for temporal behavioral analytics
- Real Kafka/Kinesis streaming integration
- Explainable AI: SHAP-based feature attribution for alert triage
- AI Agent Security: Detection of prompt injection, agent drift, and privilege escalation in agentic AI systems
- Autonomous Remediation: Agentic investigation and containment workflows
- MITRE ATT&CK coverage mapping and red team validation

---

## Contact

**Dr. Oluwaseyi Paul Babalola**  
babalolaseyip@gmail.com  
www.linkedin.com/in/oluwaseyi-babalola-06384715
