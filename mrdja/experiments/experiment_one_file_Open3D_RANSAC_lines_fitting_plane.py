import mrdja.ransaclpexperiments as experiments
import open3d as o3d

dataset = o3d.data.OfficePointClouds()
office_paths = dataset.paths
filename = office_paths[0]

# add timing
import time
start = time.time()

threshold = 0.02
# repetitions = 10
repetitions = 3
# iterations_list = [100, 200, 300, 400, 500, 600]
iterations_list = [100, 200]
percentage_chosen_lines = 0.2
percentage_chosen_planes = 0.05
seed = 42

dict_results = experiments.get_data_comparison_ransac_and_ransaclp(filename = filename, repetitions = repetitions, 
                                                                   iterations_list = iterations_list, threshold = threshold, 
                                                                   cuda = True,
                                                                    percentage_chosen_lines = percentage_chosen_lines, 
                                                                    percentage_chosen_planes = percentage_chosen_planes, verbosity_level = 1, 
                                                                    seed = seed)

end = time.time()
print("Time elapsed: ", end - start)

print(dict_results)
