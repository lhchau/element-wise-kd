import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
gr = 'parameters/T=wrn_40_2_S=wrn_40_1_T=4_alpha=1_09'
data_a = pd.read_csv(f'./{gr}/groupA.csv')
data_b = pd.read_csv(f'./{gr}/groupB.csv')
data_c = pd.read_csv(f'./{gr}/groupC.csv')
data_d = pd.read_csv(f'./{gr}/groupD.csv')

data = pd.concat([data_a, data_b, data_c, data_d], axis=1)
# Filter out columns that contain 'MIN' or 'MAX'
filtered_data = data.loc[:, ~data.columns.str.contains('MIN|MAX')]
filtered_data = filtered_data.loc[:, ~filtered_data.columns.duplicated()]
alpha = 0.5
copy_filtered_data = filtered_data.copy()
for column in copy_filtered_data.columns:
    if column != "Step":
        filtered_data[column] = filtered_data[column] / 569780
        filtered_data[column] = filtered_data[column].ewm(alpha=alpha).mean()
filtered_data = filtered_data[((filtered_data.index + 1) % 5 == 0) | (filtered_data.index == 0)]


color_blind_friendly = [('GroupA', '#E69F00', '*'),
                ('GroupB', '#56B4E9', 's'),
                ('GroupC1', 'red', 'x'),
                ('GroupC2', '#CC79A7', 'o')]
# Plot the line graph
plt.figure(figsize=(10, 6))
i = 0
for column in filtered_data.columns:
    if column != "Step":
        plt.plot(filtered_data['Step'], filtered_data[column], label=color_blind_friendly[i][0].replace('C2', 'D').replace('C1', 'C'), color=color_blind_friendly[i][1],
        marker=color_blind_friendly[i][2], markersize=7.5, linewidth=2)
        i += 1
        
fontsize = 24
number_size = 20
# Add labels and title
plt.xlabel('Epoch', fontsize=fontsize)
plt.ylabel('Num para / Total para (%)', fontsize=fontsize)
plt.title('Distribution of group A, B, C, D', fontsize=fontsize)
plt.legend(fontsize=number_size)
plt.grid(True)

plt.tick_params(axis='both', which='major', labelsize=number_size)


# Display the plot
# plt.show()
# plt.savefig('distribution.png', dpi=300, bbox_inches='tight')
plt.savefig(f"./{gr}/distribution.pdf", format="pdf", bbox_inches="tight")
