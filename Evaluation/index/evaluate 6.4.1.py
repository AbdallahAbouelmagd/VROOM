import pandas as pd
import glob
import os
from sklearn.metrics import accuracy_score, f1_score

csv_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.csv"))

all_dfs = []

for file in csv_files:
    df = pd.read_csv(file)

    df["pred_label"] = df["pred_label"].astype(str).str.strip().str.lower()
    df["gt_label"] = df["gt_label"].astype(str).str.strip().str.lower()

    df = df[df["data_type"] == "testing"]
    df = df[df["pred_label"].isin(["low", "normal", "peak"])]
    df = df[df["gt_label"].isin(["low", "normal", "peak"])]

    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)
df["correct"] = df["pred_label"] == df["gt_label"]

print("Total testing rows:", len(df))
print("Total correct predictions:", df["correct"].sum())
print("Accuracy:", df["correct"].mean())
print("Label distribution:")
print(df["gt_label"].value_counts())

y_true = df["gt_label"]
y_pred = df["pred_label"]

global_acc = accuracy_score(y_true, y_pred)
global_macro = f1_score(y_true, y_pred, average="macro")
global_weighted = f1_score(y_true, y_pred, average="weighted")

print("Accuracy:", global_acc)
print("Macro F1:", global_macro)
print("Weighted F1:", global_weighted)

camera_results = []

for file in csv_files:
    df_cam = pd.read_csv(file)

    df_cam["pred_label"] = df_cam["pred_label"].astype(str).str.strip().str.lower()
    df_cam["gt_label"] = df_cam["gt_label"].astype(str).str.strip().str.lower()

    df_cam = df_cam[df_cam["data_type"] == "testing"]
    df_cam = df_cam[df_cam["pred_label"].isin(["low", "normal", "peak"])]
    df_cam = df_cam[df_cam["gt_label"].isin(["low", "normal", "peak"])]

    if len(df_cam) == 0:
        continue

    y_t = df_cam["gt_label"]
    y_p = df_cam["pred_label"]

    camera_results.append({
        "n": len(df_cam),
        "acc": accuracy_score(y_t, y_p),
        "macro": f1_score(y_t, y_p, average="macro"),
        "weighted": f1_score(y_t, y_p, average="weighted")
    })

cam_df = pd.DataFrame(camera_results)

cam_acc = cam_df["acc"].mean()
cam_macro = cam_df["macro"].mean()
cam_weighted = cam_df["weighted"].mean()

print("camera Accuracy:", cam_acc)
print("camera Macro F1:", cam_macro)
print("camera Weighted F1:", cam_weighted)


print(cam_df)