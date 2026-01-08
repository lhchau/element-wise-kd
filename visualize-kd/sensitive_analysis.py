import pandas as pd
import matplotlib.pyplot as plt

# Data
data = {
    '1.0':  [73.33, 74.85, 74.95, 75.40, 75.52],
    '2.0':  [73.33, 74.48, 75.44, 75.79, 76.01],
    '4.0':  [73.33, 74.51, 76.35, 76.26, 76.41],
    '8.0':  [73.33, 75.26, 76.96, 76.49, 76.49]
}
index = ['Vanilla KD', r'$\rho$ = 2.0', r'$\rho$ = 3.0', r'$\rho$ = 4.0', r'$\rho$ = 5.0']
df = pd.DataFrame(data, index=index)


fontsize = 28
number_size = 28
# Plot
plt.figure(figsize=(8, 6))
for rho_idx, rho in enumerate(df.index[1:], start=1):
    plt.plot(df.columns, df.iloc[rho_idx], label=rho, marker='o', linewidth=3)

# Add Vanilla KD line
plt.axhline(df.iloc[0, 0], color='gray', linestyle='--', label='Vanilla KD', linewidth=3)

plt.xlabel(r'$\lambda_{KL}$', fontsize=fontsize)
plt.ylabel('Test Accuracy (%)', fontsize=fontsize)
# plt.title(r'Test Accuracy vs $\lambda_{KL}$ for Different $\rho$', fontsize=fontsize)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()

plt.xticks(['1.0', '2.0', '4.0', '8.0'], fontsize=number_size)
plt.yticks(fontsize=number_size)

plt.savefig('./data/entropy/sensitive_analysis.pdf', format="pdf", bbox_inches="tight")
