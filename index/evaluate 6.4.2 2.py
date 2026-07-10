import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import make_interp_spline

csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]

x_samples = []
y_f1 = []
file_names = []

for csv_file in csv_files:

    try:
        df = pd.read_csv(csv_file)

        required_cols = ["data_type", "gt_label", "pred_label"]
        if not all(col in df.columns for col in required_cols):
            continue

        df["data_type"] = df["data_type"].astype(str).str.strip().str.lower()
        df = df[df["data_type"] == "testing"].copy()

        if len(df) == 0:
            continue

        df["gt_label"] = df["gt_label"].astype(str).str.strip().str.lower()
        df["pred_label"] = df["pred_label"].astype(str).str.strip().str.lower()
        df = df.dropna(subset=["gt_label", "pred_label"])

        normal = df[df["gt_label"] == "normal"]

        if len(normal) == 0:
            continue

        f1 = f1_score(
            normal["gt_label"],
            normal["pred_label"],
            labels=["normal"],
            average="macro",
            zero_division=0
        )

        x_samples.append(len(normal))
        y_f1.append(f1)
        file_names.append(os.path.splitext(csv_file)[0])
        print(csv_file)

    except Exception:
        continue

if len(x_samples) == 0:
    print("No data found.")
    exit()

df_plot = pd.DataFrame({
    "samples": x_samples,
    "f1": y_f1,
    "name": file_names
})

df_plot = df_plot.sort_values("samples")

df_plot["bin"] = pd.cut(df_plot["samples"], bins=5, labels=False)
binned = df_plot.groupby("bin", observed=True).mean(numeric_only=True).reset_index()

print("\nBinned results:")
print(binned[["bin", "samples", "f1"]].to_string(index=False))

with PdfPages("figs/normal_class_correlation.pdf") as pdf:

    plt.figure(figsize=(10, 6))

    x = binned["samples"].values
    y = binned["f1"].values

    x_smooth = np.linspace(x.min(), x.max(), 300)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)

    plt.xlim(x.min(), x.max())

    plt.fill_between(
        x_smooth,
        y_smooth,
        color="#ff7f0e",
        alpha=1
    )

    plt.plot(
        x_smooth,
        y_smooth,
        color="black",
        linewidth=2
    )


    items = sorted(zip(x_samples, file_names), key=lambda t: t[0])

    placed_labels = []

    for i, (x_val, name) in enumerate(items):

        y_val = spline(x_val)

        x_pos = x_val
        y_pos = max(y_val - 0.03, 0.05)
        for px, py in placed_labels:
            if abs(px - x_pos) < 5 and abs(py - y_pos) < 0.05:
                x_pos += 2
        if name == "I-5 31 Zoo Drive":
            x_pos -= 1.4
        if name == "Hwy 80 at Truxel":
            x_pos += 1
        if name == "Hwy 51 at Arden":
            x_pos -= 0.2


        plt.text(x_pos, y_pos, name, fontsize=8)
        placed_labels.append((x_pos, y_pos))



    plt.xlabel("Normal samples per camera")
    plt.ylabel("Normal Traffic F1 Score")
    plt.title("")

    plt.grid(False)
    plt.ylim(bottom=0)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("Saved: normal_class_correlation.pdf")