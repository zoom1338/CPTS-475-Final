import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from ctgan import CTGAN

# Configurations for CTGAN and data processing
CTGAN_CONFIG = {
    "epochs": 120,
    "batch_size": 500,
    "embedding_dim": 64,
    "generator_dim": (128, 128),
    "discriminator_dim": (128, 128),
    "pac": 5,
    "verbose": True,
}

WEAK_ATTACKS = [
    "password",
    "injection",
    "xss",
    "scanning",
    "DDoS",
    "Worms",
    "ransomware",
]

LOG_COLS = [
    "IN_BYTES", "OUT_BYTES",
    "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
]

POST_INT_COLS = [
    "L4_SRC_PORT","L4_DST_PORT","PROTOCOL","L7_PROTO",
    "IN_BYTES","OUT_BYTES","IN_PKTS","OUT_PKTS",
    "TCP_FLAGS","FLOW_DURATION_MILLISECONDS","Label"
]


# Pre-processing functions
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


# Load and preprocess data
data_path = Path(__file__).resolve().parent / "NIDS_DF_processed.parquet"
df = pd.read_parquet(data_path)

if "Attack_int" in df.columns:
    df = df.drop(columns=["Attack_int"])

# Cap per-class rows to control runtime
MAX_PER_CLASS = 50000
small_df = (
    df.groupby("Attack", group_keys=False)
      .apply(lambda g: g.sample(n=min(len(g), MAX_PER_CLASS), random_state=42))
      .reset_index(drop=True)
)

small_df_proc = log_transform(small_df)


# Train per-class CTGAN models for weak attacks
per_class_models = {}

for label in WEAK_ATTACKS:
    subset = small_df_proc[small_df_proc["Attack"] == label] \
        .drop(columns=["Attack"], errors="ignore")

    model = CTGAN(**CTGAN_CONFIG)
    model.fit(subset)
    per_class_models[label] = model


# Synthesize weak attack samples
def synthesize_weak_attacks(base_proc_df, factor=1.0):
    parts = []

    for label, model in per_class_models.items():
        real_count = (base_proc_df["Attack"] == label).sum()
        n = int(round(real_count * factor))
        if n <= 0:
            continue

        synth_proc = model.sample(n)
        synth_proc["Attack"] = label
        parts.append(synth_proc)

    if not parts:
        return pd.DataFrame(columns=base_proc_df.columns)

    out_proc = pd.concat(parts, ignore_index=True)
    out = inverse_log_transform(out_proc)
    out = post_process(out)
    return out

synthetic_df = synthesize_weak_attacks(small_df_proc, factor=1.0)



# XGBoost classification setup
feature_cols = [c for c in small_df.columns if c != "Attack"]

X_real = small_df[feature_cols]
y_real = small_df["Attack"]

X_synth = synthetic_df[feature_cols]
y_synth = synthetic_df["Attack"]

le = LabelEncoder()
y_real_enc = le.fit_transform(y_real)
y_synth_enc = le.transform(y_synth)

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real,
    y_real_enc,
    test_size=0.2,
    stratify=y_real_enc,
    random_state=42
)

# Xgboost training and evaluation
def run_xgb(X_train, y_train, X_test, y_test, name):
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    train_unique = np.unique(y_train)
    raw_to_local = {raw: i for i, raw in enumerate(train_unique.tolist())}
    local_to_raw = {i: raw for raw, i in raw_to_local.items()}

    y_train_local = np.array([raw_to_local[int(v)] for v in y_train], dtype=int)

    test_mask = np.array([int(v) in raw_to_local for v in y_test], dtype=bool)
    if isinstance(X_test, pd.DataFrame):
        X_test_filt = X_test.loc[test_mask]
    else:
        X_test_filt = X_test[test_mask]
    y_test_filt = y_test[test_mask]
    y_test_local = np.array([raw_to_local[int(v)] for v in y_test_filt], dtype=int)

    local_class_names = [le.classes_[local_to_raw[i]] for i in range(len(train_unique))]

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

    return {"name": name, "report": report_dict}



# Run evaluations for the three scenarios (Real-only, Synthetic-only, Mixed)
# Real-only
res_real = run_xgb(X_train_real, y_train_real, X_test_real, y_test_real, "Real-only")

# Synthetic-only (TSTR)
n = len(X_train_real)
X_s = X_synth.sample(n=n, replace=True, random_state=42)
y_s = y_synth_enc[X_s.index]

res_synth = run_xgb(X_s, y_s, X_test_real, y_test_real, "Synthetic-only (TSTR)")

# Mixed
X_mix = pd.concat([X_train_real, X_s], ignore_index=True)
y_mix = np.concatenate([y_train_real, y_s])

res_mix = run_xgb(X_mix, y_mix, X_test_real, y_test_real, "Mixed Real + Synthetic")

# Save results to JSON
results = {
    "model": "XGBoost per-selected Attack Classification",
    "classes": list(le.classes_),
    "scenarios": [res_real, res_synth, res_mix],
    "ctgan": {
        "config": {k: (int(v) if isinstance(v, bool) else v) for k, v in CTGAN_CONFIG.items()},
        "weak_attacks": WEAK_ATTACKS,
        "trained_models": sorted(list(per_class_models.keys())),
    },
}

out_path = Path(__file__).resolve().parent / "results_xgboost3_tstr.json"
with open(out_path, "w") as f:
    import json
    json.dump(results, f, indent=2)
