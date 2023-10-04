import os
import pickle
import pandas as pd
import numpy as np
import glob
from mrdja.stats import friedman_nemenyi_test

results_path = "/home/scpmaotj/Github/mrdja/results_experiments_ransaclp/S3DIS" 
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

# remove "mean\_number\_inliers\_line\_" from the start of the column names
df.columns = df.columns.str.replace("mean_number_inliers_", "")
df.columns = df.columns.str.replace("line_RANSAC_", "RANSAC-LP-")
df.columns = df.columns.str.replace("standard_RANSAC_", "RANSAC-")

# divide all the columns by the column "RANSAC-109"
# df = df.div(df["RANSAC-109"], axis=0)
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
print("original df columns last before friedman")

# order the columns in df2 by name
df2 = df2.reindex(sorted(df2.columns), axis=1)
# reverse the order of the columns
df2 = df2.iloc[:, ::-1]
print("original df")
print(df2)
print("original df columns")
results = friedman_nemenyi_test(df2, already_transposed=True, filename="nemenyi_s3dis.png")
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
