import numpy as np
import math

def first_version(point, B, C):
    cross_product = np.cross(point - B, C - B)
    magnitude_cross_product = np.linalg.norm(cross_product)
    magnitude_C_minus_B = np.linalg.norm(C - B)
    distance = magnitude_cross_product / magnitude_C_minus_B
    return distance

def second_version(point, B, C):
    V = point - B
    W = C - B
    cross_product_x = V[1] * W[2] - V[2] * W[1]
    cross_product_y = V[2] * W[0] - V[0] * W[2]
    cross_product_z = V[0] * W[1] - V[1] * W[0]
    
    magnitude_cross_product = math.sqrt(cross_product_x * cross_product_x + cross_product_y * cross_product_y + cross_product_z * cross_product_z)
    magnitude_C_minus_B = math.sqrt((C[0] - B[0]) * (C[0] - B[0]) + (C[1] - B[1]) * (C[1] - B[1]) + (C[2] - B[2]) * (C[2] - B[2]))
    
    dist = magnitude_cross_product / magnitude_C_minus_B
    return dist

# test if the two versions are the same
np.random.seed(42)
# create a list with dist1 and dist2 for every point
dist1_list = []
dist2_list = []
for _ in range(100):
    point = np.random.rand(3)
    B = np.random.rand(3)
    C = np.random.rand(3)
    dist1 = first_version(point, B, C)
    dist2 = second_version(point, B, C)
    dist1_list.append(dist1)
    dist2_list.append(dist2)
dist1_list = np.array(dist1_list)
dist2_list = np.array(dist2_list)
print(np.allclose(dist1_list, dist2_list))