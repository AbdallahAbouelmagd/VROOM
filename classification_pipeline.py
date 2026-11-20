import os
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from traffic_detector import process_single_video, collect_raw_areas_from_video, brightness_variation
from sklearn.metrics import f1_score

CSV_PATH = r"G:\Hwy 51 at Arden\vehicle_density.csv"       #Enter correct path here
BASE_PATH = r"G:\Hwy 51 at Arden"                          #Enter correct path here

RUN_ON_ALL_DATA = False

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

df["data_type"] = df["data_type"].astype(str).str.strip().str.lower()

train_df = df[df["data_type"] == "training"].copy()
test_df = df[df["data_type"] == "testing"].copy()

video_to_row = df.set_index("Video")

train_video_set = set(train_df["Video"].tolist())
test_video_set = set(test_df["Video"].tolist())
train_paths = []
test_paths = []
train_features_illum = []
all_paths = []
all_train_areas_night = []
all_train_areas_day = []

for root, _, files in os.walk(BASE_PATH):
    for f in files:
        if f.endswith(".mp4"):
            all_paths.append(os.path.join(root, f))

for p in all_paths:
    name = os.path.basename(p)
    if name in train_video_set:
        train_paths.append(p)
    elif name in test_video_set:
        test_paths.append(p)





for p in train_paths:
    feat = brightness_variation(p)
    if feat is None:
        continue
    train_features_illum.append(feat)

if len(train_features_illum) < 2:
    print("not enough data")
    exit()

X = np.array(train_features_illum).reshape(-1, 1)
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(X)
clusters = gmm.predict(X)
cluster_means = [X[clusters == i].mean() for i in range(2)]
high_var_cluster = np.argmax(cluster_means)





for p in train_paths:

    video_name = os.path.basename(p)
    if video_name not in video_to_row.index:
        continue

    row = video_to_row.loc[video_name]
    feat = brightness_variation(p)
    cluster = gmm.predict([[feat]])[0]

    if cluster == high_var_cluster:
        illumination_pred = "low"
    else:
        illumination_pred = "high"
    label = "night" if illumination_pred == "low" else "day"

    areas_night, areas_day = collect_raw_areas_from_video(p, label == "night")

    if label == "night":
        all_train_areas_night.extend(areas_night)
    else:
        all_train_areas_day.extend(areas_day)

all_train_areas_night = np.array(all_train_areas_night)
print(len(all_train_areas_night), "NIGHT areas found in training set")
all_train_areas_day = np.array(all_train_areas_day)

if len(all_train_areas_day) >= 20:

    areas_day_log = np.log(all_train_areas_day + 1).reshape(-1, 1)

    gmm_day = GaussianMixture(
        n_components=3,
        random_state=42
    )

    gmm_day.fit(areas_day_log)

    day_labels = gmm_day.predict(areas_day_log)

    day_cluster_order = np.argsort(gmm_day.means_.flatten())
    vehicle_cluster_day = day_cluster_order[1]

    vehicle_day_areas = np.exp(
        areas_day_log[day_labels == vehicle_cluster_day]
    ).flatten()

    if len(vehicle_day_areas) < 20:
        vehicle_day_areas = all_train_areas_day

    min_area_day = np.percentile(vehicle_day_areas, 5)
    max_area_day = np.percentile(vehicle_day_areas, 95)

else:
    print("Insufficient DAY samples for GMM")
    min_area_day = 50
    max_area_day = 5000

print ("min_area_day:", min_area_day)
print("max_area_day:", max_area_day)
if len(all_train_areas_night) >= 20:

    areas_night_log = np.log(all_train_areas_night + 1).reshape(-1, 1)

    gmm_night = GaussianMixture(
        n_components=3,
        random_state=42
    )

    gmm_night.fit(areas_night_log)

    night_labels = gmm_night.predict(areas_night_log)

    night_cluster_order = np.argsort(gmm_night.means_.flatten())
    vehicle_cluster_night = night_cluster_order[1]

    vehicle_night_areas = np.exp(
        areas_night_log[night_labels == vehicle_cluster_night]
    ).flatten()

    if len(vehicle_night_areas) < 20:
        vehicle_night_areas = all_train_areas_night

    min_area_night = np.percentile(vehicle_night_areas, 5)
    max_area_night = np.percentile(vehicle_night_areas, 95)

else:
    print("Insufficient NIGHT samples for GMM")
    min_area_night = 10
    max_area_night = 1000
print("min_area_night:", min_area_night)
print("max_area_night:", max_area_night)
train_densities = []

for p in train_paths:

    video_name = os.path.basename(p)
    if video_name not in video_to_row.index:
        continue

    row = video_to_row.loc[video_name]

    feat = brightness_variation(p)
    cluster = gmm.predict([[feat]])[0]

    if cluster == high_var_cluster:
        illumination_pred = "low"
    else:
        illumination_pred = "high"
    label = "night" if illumination_pred == "low" else "day"
    df.loc[df["Video"] == video_name, "pred_illum"] = illumination_pred

    if label == "night":
        min_area = min_area_night
        max_area = max_area_night
    else:
        min_area = min_area_day
        max_area = max_area_day

    density = process_single_video(
        p,
        min_area=min_area,
        max_area=max_area,
        is_night=(label == "night"),
    )

    train_densities.append(density)

max_density = (
    np.percentile(train_densities, 90)
    if len(train_densities) > 0
    else 1
)

print("Max found density in training:", max_density)

train_gt_values = train_df["Avg_Vehicles_Per_Frame"].values
gt_p90 = np.percentile(train_gt_values, 90) if len(train_gt_values) > 0 else 1


df["pred_density"] = np.nan
df["pred_norm_density"] = np.nan

df["pred_label"] = pd.NA
df["gt_label"] = pd.NA
df["pred_illum"] = pd.NA

df["gt_norm_density"] = np.nan
df["label_match"] = pd.NA

df["accuracy"] = np.nan
df["macro_f1"] = np.nan
df["weighted_f1"] = np.nan

inference_paths = all_paths if RUN_ON_ALL_DATA else test_paths

for p in inference_paths:

    video_name = os.path.basename(p)

    if video_name not in video_to_row.index:
        continue

    row = video_to_row.loc[video_name]

    feat = brightness_variation(p)
    cluster = gmm.predict([[feat]])[0]

    if cluster == high_var_cluster:
        illumination_pred = "low"
    else:
        illumination_pred = "high"
    label = "night" if illumination_pred == "low" else "day"
    df.loc[df["Video"] == video_name, "pred_illum"] = illumination_pred

    if label == "night":
        min_area = min_area_night
        max_area = max_area_night
    else:
        min_area = min_area_day
        max_area = max_area_day

    density = process_single_video(
        p,
        min_area=min_area,
        max_area=max_area,
        is_night=(label == "night"),
    )
    if density is None:
        density = 0
    norm_density = density / max_density if max_density != 0 else 0

    if norm_density <= 0.33:
        pred_label = "low"
    elif norm_density >= 0.66:
        pred_label = "high"
    else:
        pred_label = "normal"

    gt_raw = row["Avg_Vehicles_Per_Frame"]
    gt_norm = gt_raw / gt_p90 if gt_p90 != 0 else 0

    if gt_norm <= 0.33:
        gt_label = "low"
    elif gt_norm >= 0.66:
        gt_label = "high"
    else:
        gt_label = "normal"

    df.loc[df["Video"] == video_name, "pred_density"] = density
    df.loc[df["Video"] == video_name, "pred_norm_density"] = norm_density
    df.loc[df["Video"] == video_name, "pred_label"] = pred_label

    df.loc[df["Video"] == video_name, "gt_norm_density"] = gt_norm
    df.loc[df["Video"] == video_name, "gt_label"] = gt_label

    df.loc[df["Video"] == video_name, "label_match"] = (pred_label == gt_label)


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

output_path = CSV_PATH.replace(".csv", "_with_predictions.csv")
df.to_csv(output_path, index=False)

print("Saved:", output_path)


print("Macro F1:", macro_f1)
print("Weighted F1:", weighted_f1)