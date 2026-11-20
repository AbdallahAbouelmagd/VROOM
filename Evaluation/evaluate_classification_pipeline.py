import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

CSV_PATH = r"G:\Hwy 51 at Arden\vehicle_density_with_predictions.csv" #Enter correct path here

RUN_ON_ALL_DATA = False

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

df["data_type"] = df["data_type"].astype(str).str.strip().str.lower()

train_df = df[df["data_type"] == "training"].copy()
test_df = df[df["data_type"] == "testing"].copy()

train_gt_values = train_df["Avg_Vehicles_Per_Frame"].values
gt_p90 = np.percentile(train_gt_values, 90) if len(train_gt_values) > 0 else 1

df["gt_norm_density"] = np.nan
df["gt_label"] = pd.NA
df["label_match"] = pd.NA

df["accuracy"] = np.nan
df["macro_f1"] = np.nan
df["weighted_f1"] = np.nan

for index, row in df.iterrows():
    if pd.isna(row["pred_label"]):
        continue

    gt_raw = row["Avg_Vehicles_Per_Frame"]
    gt_norm = gt_raw / gt_p90 if gt_p90 != 0 else 0

    if gt_norm <= 0.33:
        gt_label = "low"
    elif gt_norm >= 0.66:
        gt_label = "high"
    else:
        gt_label = "normal"

    df.loc[index, "gt_norm_density"] = gt_norm
    df.loc[index, "gt_label"] = gt_label
    df.loc[index, "label_match"] = (row["pred_label"] == gt_label)

if RUN_ON_ALL_DATA:
    eval_df = df.dropna(subset=["gt_label", "pred_label"])
    accuracy = eval_df["label_match"].mean()
    print("Accuracy:", accuracy)
else:
    eval_df = df.loc[test_df.index].dropna(subset=["gt_label", "pred_label"])
    accuracy = eval_df["label_match"].mean()
    print("Test accuracy:", accuracy)

macro_f1 = f1_score(
    eval_df["gt_label"],
    eval_df["pred_label"],
    average="macro"
)

weighted_f1 = f1_score(
    eval_df["gt_label"],
    eval_df["pred_label"],
    average="weighted"
)

df["accuracy"] = accuracy
df["macro_f1"] = macro_f1
df["weighted_f1"] = weighted_f1

output_path = CSV_PATH.replace("_with_predictions.csv", "_evaluated.csv")
df.to_csv(output_path, index=False)

print("Saved:", output_path)
print("Macro F1:", macro_f1)
print("Weighted F1:", weighted_f1)
