import os
import glob
import pickle

# Load the max_values dictionary
with open('/home/scpmaotj/Stanford3dDataset_v1.2/max_values.pkl', 'rb') as file:
    max_values = pickle.load(file)

def add_maximum_to_files(directory):
    for filepath in glob.glob(directory + '**/*.pkl', recursive=True):
        print(filepath)
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        relative_path = os.path.relpath(filepath, directory)
        key = relative_path.replace(os.path.sep, '_').replace('.pkl', '_maximum')
        #remove _means from key
        key = key.replace('_means', '')
        print("Key: ", key)
        if key in max_values:
            print(key)
            data['maximum'] = max_values[key]
            with open(filepath, 'wb') as file:
                pickle.dump(data, file)

directory = "/home/scpmaotj/pkl_files/me/" # specify your directory
add_maximum_to_files(directory)