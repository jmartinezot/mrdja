import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi

def friedman_nemenyi_test(dataframe):
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
    df = df.T

    # Perform Nemenyi posthoc test
    nemenyi_result = posthoc_nemenyi(df.values)
    nemenyi_result.columns = df.index
    nemenyi_result.index = df.index

    # Create a DataFrame of the Nemenyi results
    nemenyi_df = pd.DataFrame(nemenyi_result)

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

    # Convert the DataFrame to a LaTeX table
    nemenyi_table = to_latex_custom(nemenyi_df, means)

    nemenyi_df_2 = nemenyi_df.copy()
    nemenyi_df_2.reset_index(drop=True, inplace=True)
    nemenyi_df_2.columns = nemenyi_df_2.index
    # change the column names to strings
    nemenyi_df_2.columns = nemenyi_df_2.columns.astype(str)

    nemenyi_table_2 = to_latex_custom(nemenyi_df_2, means_2)

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








