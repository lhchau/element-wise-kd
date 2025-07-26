import numpy as np
import matplotlib.pyplot as plt

# Data
models = ["ResNet20", "ResNet56", "ResNet110", "ResNet152"]
max_zT_except_t = [9.20, 10.08, 9.70, 10.87]
zT_t = [11.92, 13.80, 13.21, 15.15]

# Bar width
bar_width = 0.3
x = np.arange(len(models))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
bar1 = ax.bar(x - bar_width/2, max_zT_except_t, bar_width, label='max(z_S except t)', color='b')
bar2 = ax.bar(x + bar_width/2, zT_t, bar_width, label='z^t_S', color='g')

# Add values above bars
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')

add_values(bar1)
add_values(bar2)

# Labels and Titles
ax.set_xlabel("Model")
ax.set_ylabel("Value")
ax.set_title("Comparison of max(z_S except t) and z^t_S")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Show plot
plt.savefig(f"data/logit_compare_student/bar_plot.pdf", format="pdf", bbox_inches="tight")
