# % KD vs TCKD with the same KL_weight
# % ResNet20:  70.75, 69.72
# % ResNet56:  71.25, 67.22
# % ResNet110: 70.81, 68.27
# % ResNet152: 71.16, 67.34

import numpy as np
import matplotlib.pyplot as plt

# Data
models = ["ResNet20", "ResNet56", "ResNet110", "ResNet152"]
KD = [70.75, 71.25, 70.81, 71.16]
TCKD = [69.72, 67.22, 68.27, 67.34]

# Bar width
bar_width = 0.3
x = np.arange(len(models))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
bar1 = ax.bar(x - bar_width/2, KD, bar_width, label='KD', color='b')
bar2 = ax.bar(x + bar_width/2, TCKD, bar_width, label='TCKD', color='g')

# Add values above bars
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')

add_values(bar1)
add_values(bar2)

# Labels and Titles
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("KD vs TCKD with the same KL_weight")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(66, 72)  # Limit y-axis range
ax.legend()

# Show plot
plt.savefig(f"data/tckd/bar_plot.pdf", format="pdf", bbox_inches="tight")
