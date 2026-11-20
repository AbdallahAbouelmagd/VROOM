import os
import pandas as pd
from sklearn.metrics import f1_score

csv_files = [f for f in os.listdir("") if f.endswith(".csv")]

TRAFFIC_CLASSES = ["low", "normal", "peak"]

all_data = []

for csv_file in csv_files:

    try:
        df = pd.read_csv(csv_file)

        required_cols = ["data_type", "gt_label", "pred_label"]

        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {csv_file} (missing required columns)")
            continue

        df["data_type"] = df["data_type"].astype(str).str.strip().str.lower()
        df = df[df["data_type"] == "testing"].copy()

        if len(df) == 0:
            print(f"Skipping {csv_file} (no testing rows)")
            continue

        df["gt_label"] = df["gt_label"].astype(str).str.strip().str.lower()
        df["pred_label"] = df["pred_label"].astype(str).str.strip().str.lower()

        df = df.dropna(subset=["gt_label", "pred_label"])

        all_data.append(df[["gt_label", "pred_label"]])

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

if len(all_data) == 0:
    print("No valid data found.")
    exit()

df_all = pd.concat(all_data, ignore_index=True)


for tclass in TRAFFIC_CLASSES:

    subset = df_all[df_all["gt_label"] == tclass]

    if len(subset) == 0:
        print(f"{tclass}: no samples")
        continue

    y_true = subset["gt_label"]
    y_pred = subset["pred_label"]

    accuracy = (y_pred == tclass).mean()
    f1 = f1_score(y_true, y_pred, labels=[tclass], average="macro", zero_division=0)

    print(f"Class: {tclass}")
    print(f"  Samples : {len(subset)}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1      : {f1:.4f}\n")

print("Done.")