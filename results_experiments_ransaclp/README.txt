The file "results_baseline_S3DIS.pkl" is the results of open3d with 100,000 iterations over the S3DIS database. The code for generating is in the docstring of "get_baseline_S3DIS" in "ransaclpexperiments.py".
The pkl files in Open3D are the results after applying the algorithm over the database of Open3D. The file for generating this data is "experiment_all_files_Open3D_RANSAC_lines_fitting_plane.py"
The pkl files in S3DIS are the results after applying the algorithm over the database of S3DIS. The file for generating this data is "experiment_all_files_S3DIS_RANSAC_lines_fitting_plane.py"
To perform the Friedman test and Nemenyi post-hoc, just run "friedman_and_nemenyi_Open3D.py" and "friedman_and_nemenyi_S3DIS.py".
