import os
import pickle
import pandas as pd
import numpy as np
import glob
from mrdja.stats import friedman_nemenyi_test

results_path = "/home/scpmaotj/Github/mrdja/results_experiments_ransaclp/Open3D" 
pkl_files = glob.glob(results_path + "/**/*.pkl", recursive=True)

# Initialize an empty dictionary to hold all the data
all_data = {}
  
for file_path in pkl_files:
    # Load data from the current file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        # data is a dict. Remove all the keys that are not mean*
        data = {key: data[key] for key in data.keys() if key.startswith("mean")}
    print(data)
    # retain only the last two sub directories and the filename of file_path
    file_path = "/".join(file_path.split("/")[-3:])
    all_data[file_path] = data

# Convert the dictionary of dictionaries into a DataFrame

df = pd.DataFrame(all_data).T

# retain only the 20 first rows
# df = df.iloc[:10]

# remove "mean\_number\_inliers\_line\_" from the start of the column names
df.columns = df.columns.str.replace("mean_number_inliers_", "")
df.columns = df.columns.str.replace("line_RANSAC_", "RANSAC-LP-")
df.columns = df.columns.str.replace("standard_RANSAC_", "RANSAC-")
# remove all columns except these four: RANSAC-LP-957, RANSAC-957, RANSAC-LP-747, RANSAC-747
# df = df[["RANSAC-LP-957", "RANSAC-239", "RANSAC-109"]]

# divide all the columns by the column "RANSAC-109"
# df = df.div(df["RANSAC-109"], axis=0)

print("original df")
print(df)
print("original df columns")

df = df.div(df["RANSAC-109"], axis=0)



# substract 1 from all the columns and multiply by 100
df = (df - 1) * 100
# get the mean of all columns
df2 = df.mean(axis=0)

print("original df")
print(df2)
print("original df columns")

df2 = df.rank(axis=1, method='min', ascending=True)
#results = friedman_nemenyi_test(df)

print("original df")
print(df2)
print("original df columns")

# rename the columns
# df.columns = ["RANSAC-LP", "RANSAC-LP-ICP", "RANSAC-LP-ICP-2"]

results = friedman_nemenyi_test(df2)
friedman_result = results["friedman_result"]
mean_ranks = results["mean_ranks"]
print("Friedman result:", friedman_result)
print("Mean ranks:\n", mean_ranks)

# df2 = df.rank(axis=1, method='min', ascending=False)
# order the columns in df2 by name
df2 = df2.reindex(sorted(df2.columns), axis=1)
# reverse the order of the columns
df2 = df2.iloc[:, ::-1]
print("original df")
print(df2)
print("original df columns")
results = friedman_nemenyi_test(df2)
nemenyi_df = results["nemenyi_df"]
nemenyi_table = results["nemenyi_table"]
nemenyi_result = results["nemenyi_result"]
nemenyi_df_2 = results["nemenyi_df_2"]
nemenyi_table_2 = results["nemenyi_table_2"]


print("Nemenyi result:\n", nemenyi_result)
print("Nemenyi df:\n", nemenyi_df)
print("Nemenyi table:\n", nemenyi_table)
print("Nemenyi df 2:\n", nemenyi_df_2)
print("Nemenyi table 2:\n", nemenyi_table_2)
print(df2.columns)

# an example of how to create a vertical table with cell coloring

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'RANSAC-LP-957': [1.000, 0.998, 0.442, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000],
    'RANSAC-LP-747': [0.998, 1.000, 0.984, -0.013, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000],
    'RANSAC-LP-558': [0.442, 0.984, 1.000, 0.596, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000],
    'RANSAC-LP-388': [-1.000, -0.013, 0.596, 1.000, 0.043, -1.000, 0.007, 0.001, -1.000, -1.000, -1.000, -1.000, -1.000],
    'RANSAC-LP-239': [-1.000, -1.000, -1.000, 0.043, 1.000, -1.000, 1.000, 1.000, 0.905, 0.083, -1.000, -1.000, -1.000],
    'RANSAC-LP-109': [-1.000, -1.000, -1.000, -1.000, -1.000, 1.000, -1.000, -1.000, -1.000, -0.006, 0.751, 0.994, 1.000],
}

df = pd.DataFrame(data)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Create a heatmap using imshow
cax = ax.matshow(df, cmap='coolwarm', aspect='auto')

# Add a colorbar
cbar = fig.colorbar(cax)

# Set column names as labels on the y-axis (vertical)
ax.set_yticks(np.arange(len(df.index)))
ax.set_yticklabels(df.index)

# Set row names as labels on the x-axis (horizontal)
ax.set_xticks(np.arange(len(df.columns)))
ax.set_xticklabels(df.columns, rotation='vertical')

# Add text to each cell based on the values and conditions
for i in range(len(df.index)):
    for j in range(len(df.columns)):
        value = df.iloc[i, j]
        if value > 0:
            color = 'green'
        elif value < 0:
            color = 'red'
        else:
            color = 'gray'
        ax.text(j, i, f"{value:.3f}", va='center', ha='center', color=color, fontweight='bold' if value != 0 else 'normal')

# Set plot title
plt.title("Vertical Table with Cell Coloring")

# Save the figure as a JPEG image
plt.savefig("vertical_table.jpg", format='jpg', bbox_inches='tight')

# Show the plot
plt.show()
