import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


cm = np.array([
    [23, 0, 2],
    [0, 3, 8],
    [0, 3, 6]
])

plt.figure(figsize=(8, 6))

ax = sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="magma",
    cbar=True,
    linewidths=0.5,
    linecolor="black",
    xticklabels=["low", "normal", "high"],
    yticklabels=["high", "normal", "low"]
)

ax.set_xlabel("Predicted label", fontsize=12)
ax.set_ylabel("Ground truth label", fontsize=12)

plt.tight_layout()

plt.savefig("confusion_matrix.pdf", format="pdf", bbox_inches="tight")

