import pandas as pd
import glob
import os
from sklearn.metrics import accuracy_score, f1_score

csv_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.csv"))

camera_results = []

for file in csv_files:
    df = pd.read_csv(file)

    df["pred_illum"] = df["pred_illum"].astype(str).str.strip().str.lower()
    df["gt_illum"] = df["gt_illum"].astype(str).str.strip().str.lower()

    df = df[df["pred_illum"].isin(["low", "high"])]
    df = df[df["gt_illum"].isin(["low", "high"])]

    if len(df) == 0:
        continue

    acc = accuracy_score(df["gt_illum"], df["pred_illum"])
    f1_macro = f1_score(df["gt_illum"], df["pred_illum"], average="macro")
    f1_weighted = f1_score(df["gt_illum"], df["pred_illum"], average="weighted")

    camera_results.append({
        "camera": os.path.basename(file),
        "n_samples": len(df),
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted
    })

cam_df = pd.DataFrame(camera_results)

final_accuracy = cam_df["accuracy"].mean()
final_macro_f1 = cam_df["macro_f1"].mean()
final_weighted_f1 = cam_df["weighted_f1"].mean()

final = pd.DataFrame([{
    "accuracy": final_accuracy,
    "macro_f1": final_macro_f1,
    "weighted_f1": final_weighted_f1
}])

print(final)

print(cam_df)

final.to_csv("illumination_final_camera_balanced.csv", index=False)
cam_df.to_csv("illumination_per_camera_report.csv", index=False)