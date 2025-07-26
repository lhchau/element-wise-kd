import matplotlib.pyplot as plt

# Data
methods = [
    "AT", "FitNets", "RKD", "CRD", "OFD", "ReviewKD",
    "CAT-KD", "DKD", "KD", "WTTM", "KD+LS", "Our ATS"
]
methods = [
    "AT", "FitNets", "CRD", "OFD", "ReviewKD",
    "CAT-KD", "DKD", "KD", "WTTM", "KD+LS", "Our ATS"
]
times = [21.9, 18.4, 26.2, 26.9, 33.2, 
         19, 18.7, 17.8, 17.8, 18.5, 18.]
accuracies = [73.5, 73.44, 75.51, 74.95, 75.63, 76.91, 76.32, 73.33, 76.06, 76.62, 76.96]

# Color setup
main_color = "#1f77b4"
highlight_color = "#d62728"

time_threshold = 25     # e.g., low if <= 20ms
acc_threshold = 74.    # e.g., high if >= 75%

colors = []
for i in range(len(methods)):
    if methods[i] == "Our ATS":
        colors.append("green")  # ATS is always green
    elif times[i] > time_threshold and accuracies[i] >= acc_threshold:
        colors.append("orange")  # High time + high accuracy
    elif times[i] <= time_threshold and accuracies[i] < acc_threshold:
        colors.append("red")     # Low time + low accuracy
    elif times[i] <= time_threshold and accuracies[i] >= acc_threshold:
        colors.append("lightgreen")  # Low time + high accuracy
    else:
        colors.append("orange")

fontsize = 24
number_size = 20
plt.figure(figsize=(10, 6))

# Plot all methods except ATS
for i, method in enumerate(methods):
    if method == "Our ATS":
        plt.scatter(times[i], accuracies[i], color=colors[i], marker='*', s=300, label='ATS')
        plt.annotate(method, (times[i]-0.5, accuracies[i] - 0.08), ha='center', va='top', fontsize=18, color='darkgreen')
    elif method == "CAT-KD":
        plt.scatter(times[i], accuracies[i], color=colors[i], marker='o', s=125)
        plt.annotate(method, (times[i]+1, accuracies[i] - 0.08), ha='center', va='top', fontsize=18)
    elif method == "FitNets":
        plt.scatter(times[i], accuracies[i], color=colors[i], marker='o', s=125)
        plt.annotate(method, (times[i]+1, accuracies[i] - 0.08), ha='center', va='top', fontsize=18)
    else:
        plt.scatter(times[i], accuracies[i], color=colors[i], marker='o', s=125)
        plt.annotate(method, (times[i], accuracies[i] - 0.08), ha='center', va='top', fontsize=18)

plt.xlabel("Training time per batch (ms)", fontsize=fontsize)
plt.ylabel("Accuracy (%)", fontsize=fontsize)
plt.xlim(min(times) - 2, max(times) + 5)
plt.ylim(min(accuracies) - 0.5, max(accuracies) + 0.5)
plt.xticks(fontsize=number_size)
plt.yticks(fontsize=number_size)
plt.grid(False)
plt.tight_layout()
plt.savefig('./data/acc_vs_training_time/acc_vs_training_time.pdf', format="pdf", bbox_inches="tight")
