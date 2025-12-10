import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from ctgan import CTGAN

# Configs
CTGAN_CONFIG = {
    "epochs": 120,
    "batch_size": 500,
    "embedding_dim": 64,
    "generator_dim": (128, 128),
    "discriminator_dim": (128, 128),
    "pac": 5,
    "verbose": True,
}

RARE_ATTACK_CLUSTERS = [
    ["Worms", "ransomware"],
    ["password", "injection", "xss"],
    ["DDoS", "scanning"],
    ["Backdoor", "Shellcode", "Exploits"],
]

LOG_COLS = [
    "IN_BYTES", "OUT_BYTES",
    "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
]

POST_INT_COLS = [
    "L4_SRC_PORT", "L4_DST_PORT",
    "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "OUT_BYTES",
    "IN_PKTS", "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
    "Label",
]

# Preprocessing helpers
def log_transform(df):
    df = df.copy()
    for c in LOG_COLS:
        if c in df.columns:
            df[c] = np.log1p(np.clip(df[c].astype(float), a_min=0, a_max=None))
    return df


def inverse_log_transform(df):
    df = df.copy()
    for c in LOG_COLS:
        if c in df.columns:
            df[c] = np.expm1(df[c])
    return df


def post_process(df):
    df = df.copy()
    for c in POST_INT_COLS:
        if c in df.columns:
            df[c] = df[c].round().clip(lower=0).astype(int)

    if "L4_SRC_PORT" in df.columns:
        df["L4_SRC_PORT"] = df["L4_SRC_PORT"].clip(0, 65535)
    if "L4_DST_PORT" in df.columns:
        df["L4_DST_PORT"] = df["L4_DST_PORT"].clip(0, 65535)
    if "PROTOCOL" in df.columns:
        df["PROTOCOL"] = df["PROTOCOL"].clip(0, 255)
    if "L7_PROTO" in df.columns:
        df["L7_PROTO"] = df["L7_PROTO"].clip(0, 255)

    return df


def build_discrete_columns(df):
    candidates = [
        "Attack", "Label",
        "PROTOCOL", "L7_PROTO", "TCP_FLAGS",
        "L4_SRC_PORT", "L4_DST_PORT",
    ]
    return [c for c in candidates if c in df.columns]

# Load & balance data
data_path = Path(__file__).resolve().parent / "NIDS_DF_processed.parquet"
df = pd.read_parquet(data_path)

# We don't need Attack_int here (we'll use Attack as string label)
if "Attack_int" in df.columns:
    df = df.drop(columns=["Attack_int"])

# Per-class cap to keep things balanced and manageable
k = 870
small_df = (
    df.groupby("Attack", group_keys=False)
      .apply(lambda g: g.sample(n=min(len(g), k), random_state=42))
      .reset_index(drop=True)
)

# Log-transform before training CTGANs
small_df_proc = log_transform(small_df)

# Map each attack to cluster id
attack_to_cluster: Dict[str, int] = {}
for idx, cluster in enumerate(RARE_ATTACK_CLUSTERS):
    for label in cluster:
        attack_to_cluster[label] = idx

# Train one CTGAN per cluster, conditioned on Attack
cluster_models: Dict[int, CTGAN] = {}

discrete_cols = build_discrete_columns(small_df_proc)

for cluster_id, cluster_labels in enumerate(RARE_ATTACK_CLUSTERS):
    subset = small_df_proc[small_df_proc["Attack"].isin(cluster_labels)].copy()
    if subset.empty:
        continue

    # Train CTGAN on this cluster with Attack as condition
    model = CTGAN(**CTGAN_CONFIG)
    model.fit(subset, discrete_columns=discrete_cols)
    cluster_models[cluster_id] = model
    print(f"Trained CTGAN for cluster {cluster_id}: {cluster_labels} with {len(subset)} rows.")

def sample_synthetic_for_attack(cluster_id, attack_label, n_samples):
    model = cluster_models.get(cluster_id)
    if model is None or n_samples <= 0:
        return pd.DataFrame()

    synth = model.sample(
        n_samples,
        condition_column="Attack",
        condition_value=attack_label
    )
    return synth

def synthesize_for_rare_attacks(base_proc_df, factor=1.0):
    parts = []

    for attack_label, cluster_id in attack_to_cluster.items():
        if attack_label not in base_proc_df["Attack"].unique():
            continue
        if cluster_id not in cluster_models:
            continue

        real_count = (base_proc_df["Attack"] == attack_label).sum()
        n = int(round(real_count * factor))
        if n <= 0:
            continue

        synth_proc = sample_synthetic_for_attack(cluster_id, attack_label, n)
        if synth_proc.empty:
            continue

        parts.append(synth_proc)

    if not parts:
        print("No synthetic data generated for rare attacks.")
        return pd.DataFrame(columns=base_proc_df.columns)

    out_proc = pd.concat(parts, ignore_index=True)

    # Inverse log-transform + post-process
    out = inverse_log_transform(out_proc)
    out = post_process(out)

    return out

# Generate synthetic data for rare attacks only
synthetic_df = synthesize_for_rare_attacks(small_df_proc, factor=1.0)
if not synthetic_df.empty:
    print(synthetic_df["Attack"].value_counts())
else:
    print("No synthetic rows generated.")

# Prepare data for XGBoost
feature_cols = [c for c in small_df.columns if c != "Attack"]

X_real = small_df[feature_cols]
y_real = small_df["Attack"]

X_synth = synthetic_df[feature_cols] if not synthetic_df.empty else pd.DataFrame(columns=feature_cols)
y_synth = synthetic_df["Attack"] if not synthetic_df.empty else pd.Series([], dtype=str)

le = LabelEncoder()
y_real_enc = le.fit_transform(y_real)

# Only transform y_synth if we actually have synthetic data
if len(y_synth) > 0:
    y_synth_enc = le.transform(y_synth)
else:
    y_synth_enc = np.array([], dtype=int)

class_names = le.classes_

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real,
    y_real_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_real_enc,
)

# XGBoost helper
def run_xgb_experiment(X_train, y_train, X_test, y_test, name: str):
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    # Map raw labels using train classes only
    train_unique = np.unique(y_train)
    raw_to_local = {raw: i for i, raw in enumerate(train_unique.tolist())}
    local_to_raw = {i: raw for raw, i in raw_to_local.items()}

    y_train_local = np.array([raw_to_local[int(v)] for v in y_train], dtype=int)

    # Filter test to only classes that appear in training
    test_mask = np.array([int(v) in raw_to_local for v in y_test], dtype=bool)
    X_test_filt = X_test.loc[test_mask] if isinstance(X_test, pd.DataFrame) else X_test[test_mask]
    y_test_filt = y_test[test_mask]
    y_test_local = np.array([raw_to_local[int(v)] for v in y_test_filt], dtype=int)

    local_class_names = [class_names[local_to_raw[i]] for i in range(len(train_unique))]

    # Compact class set for this split
    clf = XGBClassifier(
        objective="multi:softmax",
        num_class=len(local_class_names),
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
    )

    clf.fit(X_train, y_train_local)
    y_pred_local = clf.predict(X_test_filt)
    y_pred_raw = np.array([local_to_raw[int(i)] for i in y_pred_local], dtype=int)

    report_dict = classification_report(
        y_test_filt,
        y_pred_raw,
        target_names=local_class_names,
        digits=4,
        output_dict=True,
    )

    return {
        "name": name,
        "classes": local_class_names,
        "report": report_dict,
    }

# Run the three scenarios

# Real-only
res_real = run_xgb_experiment(
    X_train_real,
    y_train_real,
    X_test_real,
    y_test_real,
    "Real-only training",
)

# Synthetic-only (TSTR) – rare attacks only
if len(X_synth) > 0:
    n_train = len(X_train_real)
    replace_flag = len(X_synth) < n_train
    X_train_synth = X_synth.sample(n=n_train, replace=replace_flag, random_state=42)
    y_train_synth = le.transform(y_synth.loc[X_train_synth.index])

    res_synth = run_xgb_experiment(
        X_train_synth,
        y_train_synth,
        X_test_real,
        y_test_real,
        "Synthetic-only training (TSTR, rare attacks only)",
    )
else:
    print("Skipping Synthetic-only training: no synthetic data available.")

# Mixed Real + Synthetic
if len(X_synth) > 0:
    X_train_mix = pd.concat([X_train_real, X_train_synth], ignore_index=True)
    y_train_mix = np.concatenate([y_train_real, y_train_synth])

    res_mix = run_xgb_experiment(
        X_train_mix,
        y_train_mix,
        X_test_real,
        y_test_real,
        "Mixed Real + Synthetic training",
    )
else:
    print("Skipping Mixed training: no synthetic data available.")

# Save results as JSON
results = {
    "model": "XGBoost (clustered CTGAN rare attacks)",
    "classes": list(class_names),
    "scenarios": [
        res_real,
        *( [res_synth] if 'res_synth' in locals() else [] ),
        *( [res_mix] if 'res_mix' in locals() else [] ),
    ],
    "ctgan": {
        "config": CTGAN_CONFIG,
        "clusters": RARE_ATTACK_CLUSTERS,
        "trained_clusters": sorted(list(cluster_models.keys())),
    },
}

out_path = Path(__file__).resolve().parent / "results_xgboost2_tstr.json"
with open(out_path, "w") as f:
    import json
    json.dump(results, f, indent=2)
