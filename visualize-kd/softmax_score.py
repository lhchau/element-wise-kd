# % p^t_{T}
# % For T=1, Teacher;
# % Resnet20:  0.75
# % Resnet56:  0.93
# % Resnet110: 0.97
# % ResNet152: 0.98
# % For T=2, Teacher;
# % Resnet20:  0.50
# % Resnet56:  0.72
# % Resnet110: 0.81
# % ResNet152: 0.85
# % For Temperature T=4. Teacher:
# % Resnet20:  0.17
# % Resnet56:  0.26
# % Resnet110: 0.32
# % ResNet152: 0.34
# % z^t_{T}
# % Resnet20:  12.32
# % Resnet56:  14.98
# % Resnet110: 16.42
# % ResNet152: 16.96

import numpy as np
import matplotlib.pyplot as plt

# Data
models = ["ResNet20", "ResNet56", "ResNet110", "ResNet152"]
T1 = [0.75, 0.93, 0.97, 0.98]
T2 = [0.50, 0.72, 0.81, 0.85]
T4 = [0.17, 0.26, 0.32, 0.34]

# Bar width
bar_width = 0.2
x = np.arange(len(models))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
bar1 = ax.bar(x - bar_width, T1, bar_width, label='T=1', color='b')
bar2 = ax.bar(x, T2, bar_width, label='T=2', color='g')
bar3 = ax.bar(x + bar_width, T4, bar_width, label='T=4', color='r')

# Add values above bars
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')

add_values(bar1)
add_values(bar2)
add_values(bar3)

# Labels and Titles
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Teacher Performance for Different Temperatures")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Show plot
plt.savefig(f"data/softmax_score/bar_plot.pdf", format="pdf", bbox_inches="tight")

