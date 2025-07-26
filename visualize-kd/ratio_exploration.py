import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def r(z1, z2, C):
    sigma_z1     = sigmoid(z1)
    sigma_z2     = sigmoid(z2)
    sigma_z1_C   = sigmoid(z1 + C)
    sigma_z2_C   = sigmoid(z2 - C)
    numerator    = 1 - sigma_z1_C - sigma_z2_C
    denominator  = 1 - sigma_z1 - sigma_z2
    denominator  = np.clip(denominator, 1e-6, None)  # avoid /0
    return numerator / denominator

# Domain restricted to z1, z2 < 0
fontsize = 24
number_size = 20
z1_neg = np.linspace(-10, 0, 200)
z2_neg = np.linspace(-10, 0, 200)
Z1_neg, Z2_neg = np.meshgrid(z1_neg, z2_neg)
C = 1
R_neg = r(Z1_neg, Z2_neg, C)

plt.figure(figsize=(10, 6))
cp = plt.contourf(Z1_neg, Z2_neg, R_neg, levels=50, cmap='viridis')
cbar = plt.colorbar(cp, label='r')
cbar.ax.tick_params(labelsize=number_size-4)
cbar.set_label('r', fontsize=number_size)


# --- add r = 1 contour line ---
c1 = plt.contour(Z1_neg, Z2_neg, R_neg, levels=[1], colors='red', linewidths=2)
plt.clabel(c1, fmt='r=1', inline=True, fontsize=number_size)

# plt.title(fr"$r(z_1, z_2)$ with $z_1, z_2 < 0$ and $C={C}$")
plt.xlabel(r"$z_1$", fontsize=fontsize)
plt.ylabel(r"$z_2$", fontsize=fontsize)
plt.xticks(fontsize=number_size)
plt.yticks(fontsize=number_size)

z1_mid = (z1_neg[0] + z1_neg[-1]) / 2
z2_mid = (z2_neg[0] + z2_neg[-1]) / 2
plt.axvline(x=z1_mid, color='white', linestyle='--', linewidth=1.5)
plt.axhline(y=z2_mid, color='white', linestyle='--', linewidth=1.5)

# Add text at the center of each part
dx = (z1_neg[-1] - z1_neg[0]) / 4
dy = (z2_neg[-1] - z2_neg[0]) / 4
plt.text(z1_mid - dx, z2_mid + dy, "Final Phase", fontsize=fontsize, ha='center', va='center', color='white')
plt.text(z1_mid + dx, z2_mid + dy, "Early Phase", fontsize=fontsize, ha='center', va='center', color='white')
plt.text(z1_mid - dx, z2_mid - dy, "Transitional Phase", fontsize=fontsize, ha='center', va='center', color='white')
plt.text(z1_mid + dx, z2_mid - dy, "Rare Phase", fontsize=fontsize, ha='center', va='center', color='white')

plt.grid(True)
plt.tight_layout()
plt.savefig(f'./data/ratio_exploration/c={C}.pdf', format="pdf", bbox_inches="tight")
