import pandas as pd
from ctgan import CTGAN

def post_processing(df):
    nonneg_int_cols = [
        "L4_SRC_PORT",
        "L4_DST_PORT",
        "PROTOCOL",
        "L7_PROTO",
        "IN_BYTES",
        "OUT_BYTES",
        "IN_PKTS",
        "OUT_PKTS",
        "TCP_FLAGS",
        "FLOW_DURATION_MILLISECONDS",
        "Label",
    ]

    df = df.copy()

    df[nonneg_int_cols] = df[nonneg_int_cols].round()
    df[nonneg_int_cols] = df[nonneg_int_cols].clip(lower=0)
    df[nonneg_int_cols] = df[nonneg_int_cols].astype(int)

    df["L4_SRC_PORT"] = df["L4_SRC_PORT"].clip(0, 65535)
    df["L4_DST_PORT"] = df["L4_DST_PORT"].clip(0, 65535)
    df["PROTOCOL"] = df["PROTOCOL"].clip(0, 255)
    df["L7_PROTO"] = df["L7_PROTO"].clip(0, 255)

    return df

# Load and prepare data
df = pd.read_parquet("NF-UQ-NIDS.parquet")
df_small = df.sample(20000, random_state=42)
df_small = df_small.drop(columns=["Dataset"])

categorical_cols = ["Attack", "Label"] # Later we can select only one or the other for comparison like isabella said

# Train CTGAN
ctgan = CTGAN(
    epochs=30,
    verbose=True
)
ctgan.fit(df_small, categorical_cols)

# Generate and clean synthetic data
synthetic_df = ctgan.sample(5000)
synthetic_df = post_processing(synthetic_df)

# Preview rows
print(synthetic_df.head())
print("\n#--------------------------------------------------#")

# Compare class distribution
real_attack_ctns = df_small["Attack"].value_counts(normalize=True)
synthetic_attack_ctns = synthetic_df["Attack"].value_counts(normalize=True)

comparison = pd.concat([real_attack_ctns, synthetic_attack_ctns], axis=1)
comparison.columns = ["Real Dist", "Synthetic Dist"]

print(comparison)
print("#--------------------------------------------------#\n")

# Summary statistics for synthetic data
print(synthetic_df.describe())
