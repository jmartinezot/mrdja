# import numpy as np
import math
# from ransac_utils import timer_func

def compute_number_iterations(inliers_ratio, alpha):
    return math.log(alpha) / math.log(1 - inliers_ratio ** 3)