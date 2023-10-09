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

def get_pointcloud_after_substracting_point_cloud(pcd: o3d.geometry.PointCloud, substract: o3d.geometry.PointCloud,
                                                  threshold: float = 0.05) -> o3d.geometry.PointCloud:
    """
    Substracts one pointcloud from another. It removes all the points of the first pointcloud that are
    closer than *threshold* to some point of the second pointcloud.

    :param pcd: Pointcloud to substract from.
    :type pcd: o3d.geometry.PointCloud
    :param substract: Pointcloud to substract.
    :type substract: o3d.geometry.PointCloud
    :param threshold: If a point of the first pointcloud is closer to some point of the second pointcloud than this value, the point is removed.
    :type threshold: float
    :return: The results after substracting the second pointcloud from the first pointcloud.
    :rtype: o3d.geometry.PointCloud

    :Example:

    ::

        >>> import mrdja.pointcloud as pointcloud
        >>> import open3d as o3d
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=5.0, depth=1.0)
        >>> pcd_1 = mesh_box.sample_points_uniformly(number_of_points = 10000)
        >>> mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=4.0, depth=0.5)
        >>> pcd_2 = mesh_box.sample_points_uniformly(number_of_points = 10000)
        >>> pcd_1.paint_uniform_color([1, 0, 0])
        >>> pcd_2.paint_uniform_color([0, 1, 0])
        >>> pcd_1_minus_pcd_2 = pointcloud.get_pointcloud_after_substracting_point_cloud(pcd_1, pcd_2, threshold = 0.02)
        >>> pcd_1_minus_pcd_2
        PointCloud with 5861 points.
        >>> o3d.visualization.draw_geometries([pcd_1, pcd_2])
        >>> o3d.visualization.draw_geometries([pcd_1_minus_pcd_2])
        >>> pcd_2_minus_pcd_1 = pointcloud.get_pointcloud_after_substracting_point_cloud(pcd_2, pcd_1, threshold = 0.02)
        >>> pcd_2_minus_pcd_1
        PointCloud with 4717 points.
        >>> o3d.visualization.draw_geometries([pcd_2_minus_pcd_1])
    """

    def aux_func(x, y, z):
        [_, _, d] = pcd_tree.search_knn_vector_3d([x, y, z], knn=1)
        return d[0]

    pcd_tree = o3d.geometry.KDTreeFlann(substract)
    points = np.asarray(pcd.points)
    if len(pcd.colors) == 0:
        remaining_points = [point for point in points if
                            aux_func(point[0], point[1], point[2]) > threshold]
        pcd_result = o3d.geometry.PointCloud()
        pcd_result.points = o3d.utility.Vector3dVector(np.asarray(remaining_points))
        return pcd_result
    colors = np.asarray(pcd.colors)
    remaining_points_and_colors = [(point, color) for point, color in zip(points, colors) if
                                   aux_func(point[0], point[1], point[2]) > threshold]
    remaining_points = [item[0] for item in remaining_points_and_colors]
    remaining_colors = [item[1] for item in remaining_points_and_colors]
    pcd_result = o3d.geometry.PointCloud()
    pcd_result.points = o3d.utility.Vector3dVector(np.asarray(remaining_points))
    pcd_result.colors = o3d.utility.Vector3dVector(np.asarray(remaining_colors))
    return pcd_result
   