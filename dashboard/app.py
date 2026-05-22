"""
ThreatSentinel — Interactive Security Dashboard
Streamlit-based threat detection and risk monitoring interface.

Run with: streamlit run dashboard/app.py

Dr. Oluwaseyi Paul Babalola, Ph.D.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from threat_sentinel.data.loader import (
    generate_cicids_sample, generate_unsw_sample
)
from threat_sentinel.data.preprocessor import prepare_data
from threat_sentinel.models.detectors import train_all_detectors, evaluate_all
from threat_sentinel.risk.aggregator import RiskAggregator, get_risk_color
from threat_sentinel.pipeline.streaming import EventStream, StreamingDetectionPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ThreatSentinel | Security Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e; border-radius: 8px;
    padding: 16px; margin: 4px;
    border-left: 4px solid #7c3aed;
}
.risk-low    { border-left-color: #10b981 !important; }
.risk-medium { border-left-color: #f59e0b !important; }
.risk-high   { border-left-color: #f97316 !important; }
.risk-critical { border-left-color: #ef4444 !important; }
.header-banner {
    background: linear-gradient(135deg, #1e1e2e 0%, #2d1b69 100%);
    border-radius: 12px; padding: 20px; margin-bottom: 20px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {
    'LOW': '#10b981', 'MEDIUM': '#f59e0b',
    'HIGH': '#f97316', 'CRITICAL': '#ef4444'
}


# ── Cache: train models once per session ─────────────────────────────────────
@st.cache_resource
def load_and_train(dataset_name, n_normal, n_attack):
    if dataset_name == 'CICIDS-2017':
        df = generate_cicids_sample(n_normal, n_attack, random_state=42)
    else:
        df = generate_unsw_sample(n_normal, n_attack, random_state=99)

    feat_cols = [c for c in df.columns
                 if c not in {'label', 'attack_type', 'source'}
                 and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    data = prepare_data(df, feature_cols=feat_cols)
    detectors = train_all_detectors(data['X_train_normal'], verbose=False)
    return data, detectors


@st.cache_data
def run_pipeline(_detectors, _data, max_events):
    aggregator = RiskAggregator()
    stream = EventStream(
        _data['X_test'][:max_events],
        _data['y_test'][:max_events],
        _data['attack_types_test'][:max_events] if _data['attack_types_test'] is not None else None,
        batch_size=25
    )
    pipeline = StreamingDetectionPipeline(_detectors, aggregator)
    events_df = pipeline.process_stream(stream, verbose=False)
    entity_df = aggregator.get_entity_summary()
    alerts_df = aggregator.get_alerts()
    stats = pipeline.get_threat_summary()
    eval_df = evaluate_all(_detectors, _data['X_test'][:max_events],
                            _data['y_test'][:max_events])
    return events_df, entity_df, alerts_df, stats, eval_df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/ThreatSentinel-v0.1-7c3aed", width=200)
    st.markdown("### 🛡️ ThreatSentinel")
    st.markdown("**Dr. O.P. Babalola, Ph.D.**  \nSMIEEE | NRF Y2-Rated")
    st.divider()

    st.markdown("#### Pipeline Configuration")
    dataset = st.selectbox("Dataset", ['CICIDS-2017', 'UNSW-NB15'])
    n_normal = st.slider("Normal samples", 500, 3000, 1500, 250)
    n_attack = st.slider("Attack samples", 100, 800, 300, 50)
    max_events = st.slider("Events to process", 100, 800, 400, 50)
    threshold = st.slider("Alert threshold", 0.1, 0.9, 0.5, 0.05)

    st.divider()
    st.markdown("#### Model Weights")
    w_if = st.slider("Isolation Forest", 0.1, 0.8, 0.35, 0.05)
    w_svm = st.slider("One-Class SVM", 0.1, 0.8, 0.40, 0.05)
    w_ae = st.slider("Autoencoder", 0.1, 0.8, 0.25, 0.05)

    run_btn = st.button("🚀 Run Detection Pipeline", type="primary",
                         use_container_width=True)

    st.divider()
    st.markdown("**Links**")
    st.markdown("[📄 Notebook on Binder](https://mybinder.org/v2/gh/babalolaseyip/threat-sentinel/HEAD)")
    st.markdown("[💻 GitHub Repo](https://github.com/babalolaseyip/threat-sentinel)")
    st.markdown("[🔬 Google Scholar](https://scholar.google.com/citations?user=z6viTLkAAAAJ)")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
<h1 style="color:white;margin:0">🛡️ ThreatSentinel</h1>
<p style="color:#a78bfa;margin:0;font-size:1.1em">
Autonomous Threat Detection Platform | ML Anomaly Detection + Risk Aggregation
</p>
<p style="color:#6b7280;margin:0;font-size:0.85em">
Dr. Oluwaseyi Paul Babalola, Ph.D. | Senior Member IEEE | NRF Y2-Rated Researcher
</p>
</div>
""", unsafe_allow_html=True)

# ── Load and run ──────────────────────────────────────────────────────────────
with st.spinner("Loading data and training detection models..."):
    data, detectors = load_and_train(dataset, n_normal, n_attack)

with st.spinner("Running detection pipeline..."):
    events_df, entity_df, alerts_df, stats, eval_df = run_pipeline(
        detectors, data, max_events
    )

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown("### 📊 Detection Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Events", f"{stats.get('total_events', 0):,}")
c2.metric("Entities Tracked", f"{stats.get('total_entities', 0)}")
c3.metric("Alerts Raised", f"{stats.get('alerts_raised', 0)}",
           delta=f"{stats.get('alerts_raised', 0)/max(stats.get('total_events',1),1)*100:.1f}% alert rate")
c4.metric("Mean Risk Score", f"{stats.get('mean_risk_score', 0):.1f}/100")
c5.metric("Critical Entities", f"{stats.get('n_critical_entities', 0)}")

st.divider()

# ── Main content ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Threat Feed", "📈 Model Performance",
    "👤 Entity Profiles", "⚙️ Detection Details"
])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Risk Score Timeline")
        if len(events_df) > 0:
            fig = go.Figure()
            for level, color in RISK_COLORS.items():
                mask = events_df['risk_level'] == level
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=events_df[mask].index,
                        y=events_df[mask]['composite_score'],
                        mode='markers',
                        name=level,
                        marker=dict(color=color, size=5, opacity=0.7)
                    ))
            for thresh, color, label in [
                (30, '#f59e0b', 'Medium'), (60, '#f97316', 'High'),
                (80, '#ef4444', 'Critical')
            ]:
                fig.add_hline(y=thresh, line_dash='dash',
                               line_color=color, opacity=0.5,
                               annotation_text=label)
            fig.update_layout(
                height=350, template='plotly_dark',
                xaxis_title='Event Sequence', yaxis_title='Risk Score (0-100)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Risk Distribution")
        if len(events_df) > 0:
            level_counts = events_df['risk_level'].value_counts()
            fig2 = go.Figure(go.Pie(
                labels=list(RISK_COLORS.keys()),
                values=[level_counts.get(l, 0) for l in RISK_COLORS],
                marker_colors=list(RISK_COLORS.values()),
                hole=0.45
            ))
            fig2.update_layout(height=350, template='plotly_dark',
                                showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)

    # Alerts table
    st.markdown("#### 🚨 High & Critical Alerts")
    if len(alerts_df) > 0:
        display_cols = ['event_id', 'entity_id', 'entity_type',
                         'composite_score', 'risk_level', 'top_indicators']
        available = [c for c in display_cols if c in alerts_df.columns]
        st.dataframe(
            alerts_df[available].head(20),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No alerts raised at current threshold.")

    # Attack type breakdown
    if 'attack_type' in events_df.columns:
        st.markdown("#### Attack Type Risk Scores")
        atk = (events_df.groupby('attack_type')['composite_score']
               .mean().sort_values(ascending=False).reset_index())
        atk.columns = ['Attack Type', 'Mean Risk Score']
        atk['Mean Risk Score'] = atk['Mean Risk Score'].round(2)
        fig3 = px.bar(atk, x='Mean Risk Score', y='Attack Type',
                       orientation='h', color='Mean Risk Score',
                       color_continuous_scale='RdYlGn_r',
                       template='plotly_dark', height=400)
        fig3.add_vline(x=60, line_dash='dash', line_color='orange')
        st.plotly_chart(fig3, use_container_width=True)


with tab2:
    st.markdown("#### Model Performance Metrics")
    col1, col2 = st.columns(2)

    with col1:
        # ROC AUC bar
        fig_perf = go.Figure()
        metrics_show = ['precision', 'recall', 'f1', 'roc_auc']
        model_colors = ['#3b82f6', '#f59e0b', '#8b5cf6']
        for i, (model_name, row) in enumerate(eval_df.iterrows()):
            fig_perf.add_trace(go.Bar(
                name=model_name,
                x=metrics_show,
                y=[row[m] for m in metrics_show],
                marker_color=model_colors[i % len(model_colors)],
                opacity=0.85
            ))
        fig_perf.update_layout(
            barmode='group', template='plotly_dark', height=400,
            title='Detection Performance by Model',
            yaxis_range=[0, 1.1],
            xaxis_title='Metric', yaxis_title='Score'
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with col2:
        # Ensemble score correlation
        score_cols = [c for c in events_df.columns if c.endswith('_score')]
        if len(score_cols) >= 2:
            corr = events_df[score_cols].corr()
            labels = [c.replace('_score', '').replace('_', ' ').title()
                       for c in score_cols]
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=labels, y=labels,
                colorscale='RdYlGn', zmin=0, zmax=1,
                text=corr.values.round(2), texttemplate='%{text}',
                textfont=dict(size=14, color='white')
            ))
            fig_corr.update_layout(
                title='Ensemble Score Correlation',
                template='plotly_dark', height=400
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    # Performance table
    st.markdown("#### Detailed Metrics Table")
    st.dataframe(
        eval_df[['precision', 'recall', 'f1', 'roc_auc',
                  'avg_precision', 'n_alerts', 'alert_rate']].round(3),
        use_container_width=True
    )


with tab3:
    st.markdown("#### Entity Risk Profiles")
    if len(entity_df) > 0:
        col1, col2 = st.columns(2)
        with col1:
            fig_ent = px.bar(
                entity_df.head(20),
                x='current_score', y='entity_id',
                color='risk_level',
                color_discrete_map=RISK_COLORS,
                orientation='h', template='plotly_dark',
                title='Current Risk Score by Entity',
                height=500
            )
            fig_ent.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_ent, use_container_width=True)

        with col2:
            # Entity type distribution
            type_risk = entity_df.groupby(['entity_type', 'risk_level']).size().reset_index(name='count')
            fig_type = px.bar(
                type_risk, x='entity_type', y='count',
                color='risk_level', color_discrete_map=RISK_COLORS,
                template='plotly_dark', barmode='stack',
                title='Risk Levels by Entity Type', height=500
            )
            st.plotly_chart(fig_type, use_container_width=True)

        st.markdown("#### Full Entity Risk Table")
        st.dataframe(entity_df, use_container_width=True, hide_index=True)


with tab4:
    st.markdown("#### Pipeline Architecture")
    st.code("""
EventStream (CICIDS-2017 / UNSW-NB15 telemetry)
    │
    ├── Mini-batch ingestion (configurable batch size)
    │
    ▼
ThreatPreprocessor
    ├── Infinite/NaN value imputation
    ├── Low-variance feature removal (VarianceThreshold)
    └── Robust scaling (RobustScaler, trained on normal only)
    │
    ▼
Anomaly Detection Ensemble
    ├── Isolation Forest      (n_estimators=200, contamination=0.05)
    ├── One-Class SVM         (nu=0.05, kernel=RBF)
    └── Autoencoder (PCA)     (variance_explained=0.90)
    │
    ▼
RiskAggregator
    ├── Weighted ensemble scoring → composite score [0, 100]
    ├── Per-entity exponential moving average (decay=0.85)
    ├── Risk tier classification: LOW / MEDIUM / HIGH / CRITICAL
    └── Alert generation (HIGH + CRITICAL events)
    │
    ▼
Risk Sentinel Feed
    └── Adaptive access enforcement decisions
""")

    st.markdown("#### Raw Events Sample")
    if len(events_df) > 0:
        st.dataframe(
            events_df.head(50),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("#### About This Project")
    st.info("""
    **ThreatSentinel** is a research portfolio prototype demonstrating production-grade ML
    anomaly detection and risk aggregation for network security telemetry.

    This system is a foundation for a commercial ZTNA security analytics platform
    targeting autonomous threat detection, behavioral analytics, and adaptive enforcement.

    **Dr. Oluwaseyi Paul Babalola, Ph.D.**  |  babalolaseyip@gmail.com
    """)
