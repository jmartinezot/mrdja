import glob
import pickle
import os

def find_max_value(dictionary, key_to_find, exclude_key=None):
    max_value = float('-inf')
    for key, value in dictionary.items():
        if key == exclude_key:
            continue
        if key == key_to_find:
            max_value = max(max_value, value)
        if isinstance(value, dict):
            max_value = max(max_value, find_max_value(value, key_to_find, exclude_key))
        elif isinstance(value, list) and isinstance(value[0], dict):
            for item in value:
                max_value = max(max_value, find_max_value(item, key_to_find, exclude_key))
    return max_value

def find_max_values_in_files(directory):
    max_values = {}
    for filepath in glob.glob(directory + '**/*.pkl', recursive=True):
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            relative_path = os.path.relpath(filepath, directory)
            directory_name = os.path.dirname(relative_path)
            key = directory_name.replace(os.path.sep, '_') + '_maximum'
            max_values[key] = find_max_value(data, 'n_inliers_maximum', 'standard_RANSAC_100000')
    return max_values

directory = "/home/scpmaotj/Stanford3dDataset_v1.2/"  # specify your directory
max_values = find_max_values_in_files(directory)

# Save the results to disk
with open('max_values.pkl', 'wb') as file:
    pickle.dump(max_values, file)


