import mrdja.ransaclpexperiments as experiments
import open3d as o3d
import os
import pickle as pkl
import glob

database_path = "/home/scpmaotj/Stanford3dDataset_v1.2/"
ply_files = glob.glob(database_path + "/**/*.ply", recursive=True)

total_files = len(ply_files)

threshold = 0.02
repetitions = 10
# repetitions = 3
iterations_list = [100, 200, 300, 400, 500, 600]
# iterations_list = [100, 200]
percentage_chosen_lines = 0.2
percentage_chosen_planes = 0.05
seed = 42

for index, filename in enumerate(ply_files):
    filename_only_file = os.path.basename(filename)
    inherited_verbose_string = f"filename {index+1} of {total_files}: {filename_only_file}"
    dict_results = experiments.get_data_comparison_ransac_and_ransaclp(filename = filename, repetitions = repetitions, 
                                                                   iterations_list = iterations_list, threshold = threshold, 
                                                                   cuda = True,
                                                                    percentage_chosen_lines = percentage_chosen_lines, 
                                                                    percentage_chosen_planes = percentage_chosen_planes, 
                                                                    verbosity_level = 1,
                                                                    inherited_verbose_string = inherited_verbose_string,
                                                                    seed = seed)
    
    # save the results as a pickle file in the same folder as the filename file; to do so, just change the extension of the file to pkl
    filename_pkl = filename.replace(".ply", ".pkl")
    # filename is in the form '/root/open3d_data/extract/OfficePointClouds/cloud_bin_51.ply'; convert it to "/tmp/cloud_bin_51.pkl"
    # filename_pkl = filename_pkl.split("/")[-1]
    # filename_pkl = "/tmp/" + filename_pkl
    with open(filename_pkl, 'wb') as f:
        pkl.dump(dict_results, f)