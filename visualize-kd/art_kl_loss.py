import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
gr = 'data/art_logit_scale'
data = pd.read_csv(f'./{gr}/kl_loss.csv')

# Filter out columns that contain 'MIN' or 'MAX'
filtered_data = data.loc[:, ~data.columns.str.contains('MIN|MAX')]
filtered_data = filtered_data.loc[:, ~filtered_data.columns.duplicated()]
copy_filtered_data = filtered_data.copy()
filtered_data = filtered_data[((filtered_data.index + 1) % 5 == 0) | (filtered_data.index == 0)]

color_blind_friendly = {'20250315_050513': ('orange', 'x'),
                'sc_fa=0.9': ('cyan', '.'),
                'sc_fa=0.8': ('red', '*'),
                'sc_fa=0.7': ('blue', '^')}

# Plot the line graph
plt.figure(figsize=(10, 6))
for column in filtered_data.columns:
    if column != "Step":
        for key in color_blind_friendly.keys():
            if key in column:
                label = key.replace('20250315_050513', 'sc_fa=1').replace('sc_fa', 'scaling_factor')
                color, marker = color_blind_friendly[key]
                plt.plot(filtered_data['Step'], filtered_data[column], label=label, color=color,
                marker=marker, markersize=7.5, linewidth=2)
        
fontsize = 24
number_size = 20
# Add labels and title
plt.xlabel('Epoch', fontsize=fontsize)
plt.ylabel('Training KL loss', fontsize=fontsize)
plt.title('Training KL Loss of Student ResNet20', fontsize=fontsize)
plt.legend(fontsize=number_size)
plt.grid(True)

plt.tick_params(axis='both', which='major', labelsize=number_size)


# Display the plot
# plt.show()
# plt.savefig('distribution.png', dpi=300, bbox_inches='tight')
plt.savefig(f"./{gr}/kl_loss.pdf", format="pdf", bbox_inches="tight")
