import os
import pickle
import pandas as pd
import numpy as np
import glob
from mrdja.stats import friedman_nemenyi_test

# Define the directory where the data files are located
data_dir = "/home/scpmaotj/pkl_files/me/"  # replace with your directory

# Initialize an empty dictionary to hold all the data
all_data = {}

# Iterate over all files in the directory that match the pattern '*means*.pkl'
for file_path in glob.glob(os.path.join(data_dir, '*percentage*.pkl')):
    # Load data from the current file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Add the loaded data to the all_data dictionary
    all_data[os.path.basename(file_path)] = data

# Convert the dictionary of dictionaries into a DataFrame
df = pd.DataFrame(all_data).T

results = friedman_nemenyi_test(df)

nemenyi_df = results["nemenyi_df"]
nemenyi_table = results["nemenyi_table"]
friedman_result = results["friedman_result"]
mean_ranks = results["mean_ranks"]
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



