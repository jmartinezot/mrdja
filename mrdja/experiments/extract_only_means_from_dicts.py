import os
import pickle

directory = "/home/scpmaotj/pkl_files/"  # Replace with the actual directory path

for filename in os.listdir(directory):
    if filename.endswith(".pkl"):
        old_filepath = os.path.join(directory, filename)
        new_filename = f"{os.path.splitext(filename)[0]}_means.pkl"
        new_filepath = os.path.join(directory, new_filename)

        with open(old_filepath, 'rb') as file:
            old_dict = pickle.load(file)

        new_dict = {
            key: value for key, value in old_dict.items()
            if 'mean' in key
        }

        with open(new_filepath, 'wb') as file:
            pickle.dump(new_dict, file)

        print(f"Processed file: {filename} --> Created {new_filename}")

print("All files processed and saved.")
