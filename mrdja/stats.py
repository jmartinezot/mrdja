import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman as posthoc_nemenyi
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def determine_color(val, row_name, col_name, means):
    if val < 0.05:
        if means[row_name] > means[col_name]:
            return 'green'
        else:
            return 'red'
    else:
        return 'gray'
    
def format_text(val):
    return f"{val:.2f}"

def friedman_dunn_test(dataframe):
    # Make a copy of the dataframe to avoid modifying the original one
    df = dataframe.copy()
    # Transpose the DataFrame to get the algorithms as rows and datasets as columns
    df = df.T
    # Perform the Friedman test
    friedman_result = pg.friedman(df)

    # Display the Friedman test results
    print(friedman_result)
    # Get the p-value from the Friedman test result
    friedman_pvalue = friedman_result['p-unc'][0]

    # Perform the Bonferroni-Dunn post hoc test
    posthoc_dunn = pg.pairwise_ttests(data=df, parametric=False, padjust='bonferroni')

    # Display the post hoc test results
    print(posthoc_dunn)


def friedman_nemenyi_test(dataframe, already_transposed=False, filename="nemenyi.jpg"):
    # Make a copy of the dataframe to avoid modifying the original one
    df = dataframe.copy()
    
    # Replace '_' (not preceded by a backslash) with '\_' in column names for LaTeX compatibility
    df.columns = df.columns.str.replace(r'(?<!\\)_', r'\_')
    
    # Compute mean values for each algorithm
    means = df.mean()

    means_2 = means.copy()
    means_2.reset_index(drop=True, inplace=True)
    means_2.index = means_2.index.astype(str)

    # Perform Friedman test
    ranks = df.rank(axis=1, method='average', ascending=False)
    friedman_result = friedmanchisquare(*ranks.values.T)

    # Compute mean ranks
    mean_ranks = ranks.mean()

    # Transpose the DataFrame to get the algorithms as rows and datasets as columns
    if not already_transposed:
        df = df.T

    print(df)

    # Perform Nemenyi posthoc test
    nemenyi_result = posthoc_nemenyi(df.values)
    print("nemenyi_result")
    print(nemenyi_result)
    print(df.index)
    nemenyi_result.columns = df.columns
    nemenyi_result.index = df.columns

    # Create a DataFrame of the Nemenyi results
    nemenyi_df = pd.DataFrame(nemenyi_result)
    print("nemenyi_df")
    print(nemenyi_df)

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

    # Create a custom function to convert the DataFrame to a LaTeX table
    def to_latex_custom(df, means):
        header = ' & ' + ' & '.join(df.columns) + ' \\\\ \\hline'
        rows = []
        for row_name, row in df.iterrows():
            formatted_values = []
            for col_name, val in row.iteritems():
                if val < 0.05:
                    direction = '+' if means[row_name] > means[col_name] else '-'
                    formatted_values.append(f"\\bf{{{val:.3f} {direction}}}")
                else:
                    formatted_values.append(f"{val:.3f}")
            rows.append(f"{row_name} & " + ' & '.join(formatted_values) + ' \\\\ \\hline')
        return '\\begin{tabular}{|' + 'c|' * (df.shape[1] + 1) + '}\n' + header + '\n' + '\n'.join(rows) + '\n\\end{tabular}'

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

    # Convert the DataFrame to a LaTeX table
    nemenyi_table = to_latex_custom(nemenyi_df, means)

    nemenyi_df_2 = nemenyi_df.copy()
    nemenyi_df_2.reset_index(drop=True, inplace=True)
    nemenyi_df_2.columns = nemenyi_df_2.index
    # change the column names to strings
    nemenyi_df_2.columns = nemenyi_df_2.columns.astype(str)

    nemenyi_table_2 = to_latex_custom(nemenyi_df_2, means_2)
    nemenyi_df.columns = df.columns
    nemenyi_df.index = df.columns
    print("nemenyi_df_before_ordering")
    print(nemenyi_df)
    ordered_columns = ["RANSAC-LP-957", "RANSAC-LP-747", "RANSAC-LP-558", "RANSAC-LP-388", "RANSAC-LP-239", "RANSAC-LP-109", 
                    "RANSAC-957", "RANSAC-747", "RANSAC-558", "RANSAC-388", "RANSAC-239", "RANSAC-109"]
    nemenyi_df = nemenyi_df.loc[ordered_columns,ordered_columns]
    nemenyi_df_heatmap_data = to_heatmap_custom(nemenyi_df, means)
    print("nemenyi_df_heatmap_data")
    print(nemenyi_df_heatmap_data)
    to_image_custom(nemenyi_df, nemenyi_df_heatmap_data, filename)

    results = {
        'friedman_result': friedman_result,
        'mean_ranks': mean_ranks,
        'nemenyi_result': nemenyi_result,
        'nemenyi_df': nemenyi_df,
        'nemenyi_table': nemenyi_table,
        'nemenyi_df_2': nemenyi_df_2,
        'nemenyi_table_2': nemenyi_table_2
    }

    return results






