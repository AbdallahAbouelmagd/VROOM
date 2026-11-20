import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

csv_files = [f for f in os.listdir("") if f.endswith(".csv")]

ILLUM_CLASSES = ["low", "high"]

dfs = []

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)

        required_cols = ["data_type", "gt_illum", "gt_label", "pred_label"]
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {csv_file} (missing required columns)")
            continue

        df["data_type"] = df["data_type"].astype(str).str.strip().str.lower()
        df = df[df["data_type"] == "testing"].copy()

        if len(df) == 0:
            print(f"Skipping {csv_file} (no testing rows)")
            continue

        df["gt_illum"] = df["gt_illum"].astype(str).str.strip().str.lower()
        df["gt_label"] = df["gt_label"].astype(str).str.strip().str.lower()
        df["pred_label"] = df["pred_label"].astype(str).str.strip().str.lower()

        df = df.dropna(subset=["gt_illum", "gt_label", "pred_label"])
        df["camera_id"] = csv_file

        dfs.append(df)

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

if len(dfs) == 0:
    print("No valid data found.")
    exit()

df_all = pd.concat(dfs, ignore_index=True)


def compute_metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    )


results = {}


for illum in ILLUM_CLASSES:
    illum_df = df_all[df_all["gt_illum"] == illum]

    if len(illum_df) == 0:
        continue

    camera_metrics = []

    for cam_id, cam_df in illum_df.groupby("camera_id"):
        if len(cam_df) == 0:
            continue

        acc, macro_f1, weighted_f1 = compute_metrics(
            cam_df["gt_label"],
            cam_df["pred_label"]
        )

        camera_metrics.append((acc, macro_f1, weighted_f1))

    if len(camera_metrics) == 0:
        continue

    acc = sum(m[0] for m in camera_metrics) / len(camera_metrics)
    macro_f1 = sum(m[1] for m in camera_metrics) / len(camera_metrics)
    weighted_f1 = sum(m[2] for m in camera_metrics) / len(camera_metrics)

    results[illum] = (acc, macro_f1, weighted_f1)

    print(f"\nIllumination: {illum}")
    print(f"Cameras: {len(camera_metrics)}")
    print(f"Equal-Cam Accuracy: {acc:.4f}")
    print(f"Equal-Cam Macro F1: {macro_f1:.4f}")
    print(f"Equal-Cam Weighted F1: {weighted_f1:.4f}")


labels = ["Accuracy", "Macro F1", "Weighted F1"]

low_vals = results.get("low", (0, 0, 0))
high_vals = results.get("high", (0, 0, 0))

x = range(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar([i - width/2 for i in x], low_vals, width, label="Low Illumination")
plt.bar([i + width/2 for i in x], high_vals, width, label="High Illumination")

plt.xticks(list(x), labels)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.legend()

plt.tight_layout()
plt.savefig("illumination_lvl_plot.pdf")
plt.close()

print("\nSaved: illumination_lvl_plot.pdf")
print("Done.")