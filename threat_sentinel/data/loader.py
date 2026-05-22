"""
ThreatSentinel — Data Loader
Supports CICIDS-2017, UNSW-NB15, and synthetic demo data
mirroring real dataset feature schemas.

Dr. Oluwaseyi Paul Babalola, Ph.D.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── CICIDS-2017 feature schema (78 flow-based features) ──────────────────────
CICIDS_FEATURES = [
    'flow_duration', 'tot_fwd_pkts', 'tot_bwd_pkts',
    'totlen_fwd_pkts', 'totlen_bwd_pkts',
    'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std',
    'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std',
    'flow_byts_s', 'flow_pkts_s',
    'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
    'fwd_iat_tot', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',
    'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min',
    'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',
    'fwd_header_len', 'bwd_header_len',
    'fwd_pkts_s', 'bwd_pkts_s',
    'pkt_len_min', 'pkt_len_max', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var',
    'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt',
    'ack_flag_cnt', 'urg_flag_cnt', 'cwe_flag_count', 'ece_flag_cnt',
    'down_up_ratio', 'pkt_size_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg',
    'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'fwd_blk_rate_avg',
    'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'bwd_blk_rate_avg',
    'subflow_fwd_pkts', 'subflow_fwd_byts', 'subflow_bwd_pkts', 'subflow_bwd_byts',
    'init_fwd_win_byts', 'init_bwd_win_byts',
    'fwd_act_data_pkts', 'fwd_seg_size_min',
    'active_mean', 'active_std', 'active_max', 'active_min',
    'idle_mean', 'idle_std', 'idle_max', 'idle_min',
]

CICIDS_ATTACK_TYPES = [
    'BENIGN', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
    'DoS slowloris', 'Heartbleed', 'Web Attack', 'PortScan',
    'Bot', 'DDoS', 'FTP-Patator', 'SSH-Patator'
]

# ── UNSW-NB15 feature schema (49 features) ───────────────────────────────────
UNSW_FEATURES = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate',
    'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
    'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
    'smean', 'dmean', 'trans_depth', 'response_body_len',
    'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports',
    'proto_tcp', 'proto_udp', 'proto_other',
    'service_http', 'service_ftp', 'service_ssh', 'service_other',
    'state_FIN', 'state_INT', 'state_other',
]

UNSW_ATTACK_TYPES = [
    'Normal', 'Fuzzers', 'Analysis', 'Backdoors', 'DoS',
    'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms'
]


def generate_cicids_sample(n_normal=2000, n_attack=400, random_state=42):
    """
    Generate synthetic data mirroring the CICIDS-2017 feature schema.

    Attack distributions modelled after real CICIDS statistical profiles.
    Replace with real data by calling load_cicids() with the actual CSV paths.

    Parameters
    ----------
    n_normal : int
        Number of benign flow records
    n_attack : int
        Number of attack flow records
    random_state : int

    Returns
    -------
    pd.DataFrame
        Features + 'label' column (0=benign, 1=attack) + 'attack_type'
    """
    rng = np.random.RandomState(random_state)
    n_features = len(CICIDS_FEATURES)

    # Normal traffic: low packet rates, moderate durations, balanced IAT
    X_normal = np.column_stack([
        rng.exponential(500000, n_normal),        # flow_duration (microseconds)
        rng.poisson(15, n_normal),                # tot_fwd_pkts
        rng.poisson(12, n_normal),                # tot_bwd_pkts
        rng.exponential(800, n_normal),           # totlen_fwd_pkts
        rng.exponential(600, n_normal),           # totlen_bwd_pkts
        *[rng.exponential(scale, n_normal)
          for scale in np.linspace(50, 500, n_features - 5)]
    ])

    # Attack traffic: anomalous patterns per attack type
    attack_types_sample = rng.choice(CICIDS_ATTACK_TYPES[1:], n_attack)
    X_attack_list = []

    for atype in attack_types_sample:
        if 'DoS' in atype or 'DDoS' in atype:
            # High packet rates, very short flows
            row = [
                rng.exponential(5000),      # very short duration
                rng.poisson(500),           # very high fwd packets
                rng.poisson(2),             # low bwd packets
                rng.exponential(5000),
                rng.exponential(50),
                *[abs(rng.normal(200, 50)) for _ in range(n_features - 5)]
            ]
        elif 'PortScan' in atype:
            # Many short connections, no data
            row = [
                rng.exponential(1000),
                rng.poisson(2),
                rng.poisson(0.5),
                rng.exponential(10),
                rng.exponential(5),
                *[abs(rng.normal(10, 5)) for _ in range(n_features - 5)]
            ]
        elif 'Bot' in atype:
            # Periodic beaconing — regular IAT
            row = [
                rng.normal(300000, 10000),
                rng.poisson(8),
                rng.poisson(8),
                rng.exponential(400),
                rng.exponential(400),
                *[abs(rng.normal(100, 10)) for _ in range(n_features - 5)]
            ]
        else:
            # Generic attack
            row = [
                rng.exponential(100000),
                rng.poisson(50),
                rng.poisson(5),
                rng.exponential(2000),
                rng.exponential(500),
                *[abs(rng.normal(300, 100)) for _ in range(n_features - 5)]
            ]
        X_attack_list.append(row)

    X_attack = np.array(X_attack_list)

    # Combine
    X = np.vstack([X_normal, X_attack])
    X = np.clip(X, 0, None)  # No negative values in network features
    labels = np.array([0] * n_normal + [1] * n_attack)
    attack_col = ['BENIGN'] * n_normal + list(attack_types_sample)

    df = pd.DataFrame(X, columns=CICIDS_FEATURES)
    df['label'] = labels
    df['attack_type'] = attack_col
    df['source'] = 'CICIDS-2017'

    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def generate_unsw_sample(n_normal=2000, n_attack=400, random_state=99):
    """
    Generate synthetic data mirroring the UNSW-NB15 feature schema.

    Parameters
    ----------
    n_normal : int
    n_attack : int
    random_state : int

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.RandomState(random_state)
    n_features = len(UNSW_FEATURES)

    # Normal records
    X_normal = np.abs(rng.randn(n_normal, n_features) * np.linspace(1, 50, n_features))
    X_normal[:, 0] = rng.exponential(0.5, n_normal)    # dur
    X_normal[:, 1] = rng.poisson(10, n_normal)         # spkts
    X_normal[:, 2] = rng.poisson(8, n_normal)          # dpkts
    X_normal[:, 3] = rng.exponential(500, n_normal)    # sbytes
    X_normal[:, 4] = rng.exponential(400, n_normal)    # dbytes

    # Attack records
    attack_types_sample = rng.choice(UNSW_ATTACK_TYPES[1:], n_attack)
    X_attack = np.abs(rng.randn(n_attack, n_features) * np.linspace(2, 100, n_features))

    for i, atype in enumerate(attack_types_sample):
        if atype in ['DoS', 'Generic', 'Exploits']:
            X_attack[i, 1] = rng.poisson(200)   # high spkts
            X_attack[i, 3] = rng.exponential(5000)
            X_attack[i, 0] = rng.exponential(0.01)  # very short
        elif atype == 'Reconnaissance':
            X_attack[i, 1] = rng.poisson(300)
            X_attack[i, 2] = rng.poisson(1)
            X_attack[i, 0] = rng.exponential(0.001)
        elif atype == 'Backdoors':
            X_attack[i, 0] = rng.normal(10, 1)  # long connection
            X_attack[i, 5] = rng.exponential(0.1)  # low rate

    X = np.vstack([X_normal, X_attack])
    X = np.clip(X, 0, None)
    labels = np.array([0] * n_normal + [1] * n_attack)
    attack_col = ['Normal'] * n_normal + list(attack_types_sample)

    df = pd.DataFrame(X, columns=UNSW_FEATURES)
    df['label'] = labels
    df['attack_type'] = attack_col
    df['source'] = 'UNSW-NB15'

    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def load_cicids(csv_path):
    """
    Load real CICIDS-2017 dataset from CSV.
    Download from: https://www.unb.ca/cic/datasets/ids-2017.html

    Parameters
    ----------
    csv_path : str or Path
        Path to CICIDS CSV file

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    df['attack_type'] = df[label_col].str.strip()
    df['label'] = (df['attack_type'] != 'BENIGN').astype(int)
    df['source'] = 'CICIDS-2017'
    return df


def load_unsw(csv_path):
    """
    Load real UNSW-NB15 dataset from CSV.
    Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset

    Parameters
    ----------
    csv_path : str or Path

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    if 'attack_cat' in df.columns:
        df['attack_type'] = df['attack_cat'].fillna('Normal').str.strip()
    if 'label' not in df.columns:
        df['label'] = (df['attack_type'] != 'Normal').astype(int)
    df['source'] = 'UNSW-NB15'
    return df


def load_combined_sample(n_normal=2000, n_attack=400):
    """
    Load combined synthetic CICIDS + UNSW-NB15 sample for demonstration.
    Replace with real data using load_cicids() and load_unsw().
    """
    cicids = generate_cicids_sample(n_normal, n_attack)
    unsw = generate_unsw_sample(n_normal, n_attack)

    # Align on shared feature set
    shared = list(set(CICIDS_FEATURES) & set(UNSW_FEATURES))
    cicids_shared = cicids[shared + ['label', 'attack_type', 'source']]
    unsw_shared = unsw[shared + ['label', 'attack_type', 'source']]

    combined = pd.concat([cicids_shared, unsw_shared], ignore_index=True)
    return combined.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == '__main__':
    df = load_combined_sample()
    print(f"Combined dataset: {df.shape}")
    print(f"Normal: {(df.label==0).sum()} | Attack: {(df.label==1).sum()}")
    print(f"Sources: {df.source.value_counts().to_dict()}")
    print(f"Attack types: {df.attack_type.value_counts().head(8)}")
