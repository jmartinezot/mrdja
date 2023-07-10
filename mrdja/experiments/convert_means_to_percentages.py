import os
import glob
import pickle

def normalize_values(dictionary):
    max_value = dictionary['maximum']
    normalized_dict = {key: value / max_value for key, value in dictionary.items() if isinstance(value, (int, float))}
    return normalized_dict

def create_percentage_files(directory):
    for filepath in glob.glob(directory + '**/*_means.pkl', recursive=True):
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        normalized_data = normalize_values(data)
        new_filepath = filepath.replace('_means.pkl', '_percentages.pkl')
        with open(new_filepath, 'wb') as file:
            pickle.dump(normalized_data, file)

directory = "/home/scpmaotj/pkl_files/me/"  # specify your directory
create_percentage_files(directory)
