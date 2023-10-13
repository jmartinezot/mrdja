import open3d as o3d
import os
import pickle as pkl
import glob

# database_path = "/tmp/open3d_data/extract/"
database_path = "/tmp/Stanford3dDataset_v1.2/"
ply_files = glob.glob(database_path + "/**/*.ply", recursive=True)
total_files = len(ply_files)

threshold = 0.02
repetitions = 10
# repetitions = 3
iterations_list = [109, 239, 388, 558, 747, 957]

for index, filename in enumerate(ply_files):
    filename_only_file = os.path.basename(filename)
    print(f"filename {index+1} of {total_files}: {filename_only_file}")
    current_dict = dict()
    # load filename as ply and segment plane using o3d
    pcd = o3d.io.read_point_cloud(filename)
    for iterations in iterations_list:
        o3d.utility.random.seed(42)
        all_data_repetitions = []
        all_sum_inliers = 0
        for i in range(repetitions):
            data_repetition = dict()
            plane, inliers = pcd.segment_plane(distance_threshold=threshold, ransac_n=3, num_iterations=iterations)
            all_sum_inliers += len(inliers)
            data_repetition["number_inliers"] = len(inliers)
            data_repetition["plane"] = plane
            data_repetition["plane_iterations"] = iterations
            all_data_repetitions.append(data_repetition)
        current_dict[f"standard_RANSAC_{iterations}"] = all_data_repetitions
        current_dict[f"mean_number_inliers_standard_RANSAC_{iterations}"] = all_sum_inliers / repetitions
    # save the results as a pickle file in the same folder as the filename file; to do so, just change the extension of the file to pkl
    filename_pkl = filename.replace(".ply", ".pkl")
    # filename is in the form '/root/open3d_data/extract/OfficePointClouds/cloud_bin_51.ply'; convert it to "/tmp/cloud_bin_51.pkl"
    # filename_pkl = filename_pkl.split("/")[-1]
    # filename_pkl = "/tmp/" + filename_pkl
    with open(filename_pkl, 'rb') as f:
        old_data = pkl.load(f)
    old_data.update(current_dict)
    with open(filename_pkl, 'wb') as f:
        pkl.dump(old_data, f)
    
