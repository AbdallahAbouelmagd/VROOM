import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt

CLASS_COL = "gt_label"
FILTER_COL = "data_type"
FILTER_VALUE = "testing"

CLASSES = ["low", "normal", "high"]
OUTPUT_PDF = "class_distribution.pdf"

CAMERA_ORDER = [
    "Hwy 80 at Reed",
    "Hwy 51 at Arden",
    "I-5 31 Zoo Drive",
    "Hwy 51 at T St",
    "Hwy 5 at Florin",
    "I-10 83 Crenshaw Blvd On-Ramp",
    "US-101 618 Moorpark St",
    "I-10 81 Overland Ave On-Ramp",
    "Hwy 80 at Truxel",
    "Hwy 99 at 47th Ave",
]

csv_files = glob.glob("*.csv")

camera_results = {}

for file in csv_files:
    cam_name = os.path.splitext(os.path.basename(file))[0]

    df = pd.read_csv(file)

    if FILTER_COL in df.columns:
        df = df[df[FILTER_COL] == FILTER_VALUE]

    if CLASS_COL not in df.columns:
        continue

    df[CLASS_COL] = df[CLASS_COL].replace({"peak": "high"})
    counts = df[CLASS_COL].value_counts()

    camera_results[cam_name] = {c: counts.get(c, 0) for c in CLASSES}

df = pd.DataFrame.from_dict(camera_results, orient="index")
df = df[CLASSES]

df = df.reindex(CAMERA_ORDER)

df_pct = df.div(df.sum(axis=1), axis=0) * 100

def plot_grouped_bar(data, title, ylabel):
    cameras = data.index
    x = np.arange(len(cameras))
    width = 0.25

    fig, ax = plt.subplots(figsize=(20, 8))
    plt.rcParams.update({"font.size": 16})
    for i, cls in enumerate(CLASSES):
        ax.bar(x + i * width, data[cls], width, label=cls)

    ax.set_title(title)

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_yticks([20, 40, 60])
    ax.set_xticks(x + width)
    ax.set_xticklabels(cameras, rotation=45, ha="right", fontsize=14)
    ax.legend(title="Traffic class", loc="upper right")
    for spine in ax.spines.values():
        spine.set_linewidth(0.1)

    plt.tight_layout()

    return fig

with PdfPages(OUTPUT_PDF) as pdf:

    fig1 = plot_grouped_bar(df, "", "Number of samples")
    pdf.savefig(fig1)
    plt.close(fig1)

    fig2 = plot_grouped_bar(df_pct, "Class Distribution per Camera (%)", "Percentage")
    pdf.savefig(fig2)
    plt.close(fig2)


print(plt.rcParams['axes.prop_cycle'].by_key()['color'])