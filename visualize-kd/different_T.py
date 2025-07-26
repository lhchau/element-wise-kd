import matplotlib.pyplot as plt
import numpy as np

# Data
teachers = ["T=4", "T=5", "T=6", "T=7", "T=8"]
teacher_values = [70.81, 71.19, 71.14, 71.36, 70.8]
# students = ["ResNet20", "ResNet56", "ResNet110", "ResNet152"]
# student_values = [3.11, 4.18, 4.68, 4.82]

x_teacher = np.arange(len(teachers))
# x_student = np.arange(len(students))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x_teacher, teacher_values, width=0.6)
# ax.bar(x_student + 0.2, student_values, width=0.4, label="Student")

fontsize = 24
number_size = 20

# Labels and Titles
ax.set_xticks(x_teacher)
ax.set_xticklabels(teachers, rotation=0, fontsize=number_size)
ax.set_ylabel("Accuracy", fontsize=fontsize)
ax.set_title("Test accuracy of ResNet20 when KD with different T", fontsize=fontsize)
ax.set_ylim(70.5, 71.5)  # Limit y-axis range
ax.legend(fontsize=number_size)
plt.tick_params(axis='both', which='major', labelsize=number_size)

# Add values above bars
for i, v in enumerate(teacher_values):
    ax.text(x_teacher[i], v + 0.03, f"{v:.2f}", ha='center', fontsize=number_size)
# for i, v in enumerate(student_values):
    # ax.text(x_student[i] + 0.2, v + 0.2, f"{v:.2f}", ha='center', fontsize=number_size)

# Show plot
plt.tight_layout()
plt.savefig(f"data/different_T/bar_plot.pdf", format="pdf", bbox_inches="tight")
