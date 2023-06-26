# import numpy as np
import math
# from ransac_utils import timer_func

def compute_number_iterations(inliers_ratio, alpha):
    """
    Computes the minimum number of iterations needed to achieve a given probability of success in RANSAC.

    For success this is understood as the probability of finding at least one sample set free of outliers.

    :param inliers_ratio: Ratio of inliers in the data.
    :type inliers_ratio: float
    :param alpha: Desired probability of success.
    :type alpha: float
    :return: Minimum number of iterations needed.
    :rtype: float
    """
    return math.log(1-alpha) / math.log(1 - inliers_ratio ** 3)