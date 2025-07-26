import matplotlib.pyplot as plt
import numpy as np

# Data
teachers = ["ResNet20", "ResNet56", "ResNet110", "ResNet152"]
teacher_values = [3.30, 5.39, 6.74, 7.24]

students = ["ResNet20", "ResNet56", "ResNet110", "ResNet152"]
student_values = [3.11, 4.18, 4.68, 4.82]

x_teacher = np.arange(len(teachers))
x_student = np.arange(len(students))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x_teacher - 0.2, teacher_values, width=0.4, label="Teacher")
ax.bar(x_student + 0.2, student_values, width=0.4, label="Student")

fontsize = 24
number_size = 20

# Labels and Titles
ax.set_xticks(x_teacher)
ax.set_xticklabels(teachers, rotation=0, fontsize=number_size)
ax.set_ylabel("Logit Gap", fontsize=fontsize)
ax.set_title("The gap between max logit and 2nd max logit", fontsize=fontsize)
ax.set_ylim(1, 8)  # Limit y-axis range
ax.legend(fontsize=number_size)
plt.tick_params(axis='both', which='major', labelsize=number_size)

# Add values above bars
for i, v in enumerate(teacher_values):
    ax.text(x_teacher[i] - 0.2, v + 0.2, f"{v:.2f}", ha='center', fontsize=number_size)
for i, v in enumerate(student_values):
    ax.text(x_student[i] + 0.2, v + 0.2, f"{v:.2f}", ha='center', fontsize=number_size)

# Show plot
plt.tight_layout()
plt.savefig(f"data/logit_gap/bar_plot.pdf", format="pdf", bbox_inches="tight")
