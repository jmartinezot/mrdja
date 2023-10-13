import os
import subprocess
import pickle
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Create a custom function to convert the DataFrame to good data for a heatmap
# the values < 0.05 and "good" for the algorithm in the column are going to be 2 - value
# the values < 0.05 and "bad" for the algorithm in the row are going to be value
# the rest of values are going to be 1
# this function returns a DataFrame with the same shape as the original one
# always keep track of the name of the row and column of each cell,
# because the heatmap will be created using the row and column names
# a value is good if means[row_name] > means[col_name]
# a value is bad if means[row_name] < means[col_name]
def to_heatmap_custom(df, means):
    heatmap_data = pd.DataFrame(1, index=df.index, columns=df.columns)
    for i, row_name in enumerate(df.index):
        for j, col_name in enumerate(df.columns):
            value = df.loc[row_name, col_name]
            if value < 0.05:
                heatmap_data.loc[row_name, col_name] = 2 - value if means[i] > means[j] else value 
            if i == j:
                heatmap_data.loc[row_name, col_name] = 3      
    return heatmap_data

def to_image_custom(df_original, df_heatmap_data, filename):
    # Create a colormap from the list of colors
    cmap = ListedColormap(['red', 'gray', 'green', 'white'])
    sns.set(font_scale=1.2)
    plt.figure(figsize=(12, 10))
    col_labels = df_original.columns
    row_labels = df_original.index
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
    sns.heatmap(df_heatmap_data, fmt="d", linewidths=.5, cbar=False, cmap=cmap, xticklabels=col_labels, yticklabels=row_labels)
    # save the figure
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

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

ordered_columns = ["RANSAC-LP-957", "RANSAC-LP-747", "RANSAC-LP-558", "RANSAC-LP-388", "RANSAC-LP-239", "RANSAC-LP-109", 
                "RANSAC-957", "RANSAC-747", "RANSAC-558", "RANSAC-388", "RANSAC-239", "RANSAC-109"]
df = df.loc[:,ordered_columns]

# divide all the columns by the column "RANSAC-109"
# df = df.div(df["RANSAC-109"], axis=0)
df = df.div(df["RANSAC-109"], axis=0)

# Step 2: Save DataFrame to CSV
df.to_csv('temp_data.csv')

# Step 4: Run R script
subprocess.run(["Rscript", "/home/scpmaotj/Github/mrdja/mrdja/experiments/run_friedman.R"])

# Step 5: Read results back into Python
results_df = pd.read_csv('temp_result.csv')

# Step 6: Clean up
os.remove('temp_data.csv')
os.remove('temp_result.csv')

# order the columns in results_df by name
# results_df = results_df.reindex(sorted(results_df.columns), axis=1)
# reverse the order of the columns
# results_df = results_df.iloc[:, ::-1]

# change the "." in the columns names to "-"
results_df.columns = results_df.columns.str.replace(".", "-")
# Step 2: Set the "Unnamed: 0" column as the index
results_df.set_index("Unnamed: 0", inplace=True)
# Step 3: Replace '.' with '-' in the index
results_df.index = results_df.index.str.replace('.', '-')

# Compute mean values for each algorithm
means = df.mean()
results_df_heatmap_data = to_heatmap_custom(results_df, means)
print("results_df_heatmap_data")
print(results_df_heatmap_data)
to_image_custom(results_df, results_df_heatmap_data, "nemenyi_open3d.png")

'''
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
print("original df columns last before friedman")

# order the columns in df2 by name
df2 = df2.reindex(sorted(df2.columns), axis=1)
# reverse the order of the columns
df2 = df2.iloc[:, ::-1]
print("original df")
print(df2)
print("original df columns")
results = friedman_nemenyi_test(df2, already_transposed=True, filename="nemenyi_open3d.png")
friedman_result = results["friedman_result"]
mean_ranks = results["mean_ranks"]
nemenyi_df = results["nemenyi_df"]
nemenyi_table = results["nemenyi_table"]
nemenyi_result = results["nemenyi_result"]
nemenyi_df_2 = results["nemenyi_df_2"]
nemenyi_table_2 = results["nemenyi_table_2"]

print("Friedman result:", friedman_result)
print("Mean ranks:\n", mean_ranks)
print("Nemenyi result:\n", nemenyi_result)
print("Nemenyi df:\n", nemenyi_df)
print("Nemenyi table:\n", nemenyi_table)
print("Nemenyi df 2:\n", nemenyi_df_2)
print("Nemenyi table 2:\n", nemenyi_table_2)
print(df2.columns)
'''
