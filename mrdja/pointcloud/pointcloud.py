import open3d as o3d
from typing import List, Tuple, Dict, Any
import numpy as np

def pointcloud_audit(pcd: o3d.geometry.PointCloud)-> Dict[str, Any]:
    '''
    Audit point cloud.

    :param pcd: Point cloud.
    :type pcd: o3d.geometry.PointCloud
    :return: Dictionary of audit results.
    :rtype: Dict[str, Any]

    :Example:

    ::
    '''
    dict_results = {}
    number_pcd_points = len(pcd.points)
    dict_results["number_pcd_points"] = number_pcd_points
    dict_results["has_normals"] = pcd.has_normals()
    dict_results["has_colors"] = pcd.has_colors()
    dict_results["is_empty"] = pcd.is_empty()
    if not pcd.is_empty():
        # get maximum and minimum values
        np_points = np.asarray(pcd.points)
        dict_results["max_x"] = np.max(np_points[:, 0])
        dict_results["min_x"] = np.min(np_points[:, 0])
        dict_results["max_y"] = np.max(np_points[:, 1])
        dict_results["min_y"] = np.min(np_points[:, 1])
        dict_results["max_z"] = np.max(np_points[:, 2])
        dict_results["min_z"] = np.min(np_points[:, 2])
        # check if all points are finite
        dict_results["all_points_finite"] = np.all(np.isfinite(np_points))
        # check if all points are unique
        dict_results["all_points_unique"] = len(np_points) == len(np.unique(np_points, axis=0))
    return dict_results

def pointcloud_sanitize(pcd: o3d.geometry.PointCloud)->o3d.geometry.PointCloud:
    # check if the pointcloud is empty and then remove nonfinite and duplicate points
    if pcd.is_empty():
        return pcd
    # remove non finite points and associated colors
    pcd_has_colors = pcd.has_colors()
    np_points = np.asarray(pcd.points)
    np_points = np_points[np.isfinite(np_points).all(axis=1)]
    if pcd_has_colors:
        np_colors = np.asarray(pcd.colors)
        np_colors = np_colors[np.isfinite(np_points).all(axis=1)]
    # remove duplicate points and associated colors
    _, unique_indices = np.unique(np_points, axis=0, return_index=True)
    pcd.points = o3d.utility.Vector3dVector(np_points[unique_indices])
    if pcd_has_colors:
        pcd.colors = o3d.utility.Vector3dVector(np_colors[unique_indices])
    return pcd

"""
pcd = o3d.geometry.PointCloud()
dict_results = pointcloud_audit(pcd)
print(dict_results)
pcd.points = o3d.utility.Vector3dVector(np.random.randn(100, 3))
dict_results = pointcloud_audit(pcd)
print(dict_results)
pcd_filename = "/home/scpmaotj/Stanford3dDataset_v1.2/Area_1/conferenceRoom_1/conferenceRoom_1.ply"
pcd = o3d.io.read_point_cloud(pcd_filename)
dict_results = pointcloud_audit(pcd)
print(dict_results)
o3d.visualization.draw_geometries([pcd])
pcd = pointcloud_sanitize(pcd)
dict_results = pointcloud_audit(pcd)
print(dict_results)
o3d.visualization.draw_geometries([pcd])
"""