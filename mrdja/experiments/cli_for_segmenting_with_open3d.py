import open3d as o3d
import argparse
import os

def segment_plane(file_path, num_iterations, threshold):
    print(file_path)
    pcd = o3d.io.read_point_cloud(file_path)

    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                             ransac_n=3,
                                             num_iterations=num_iterations)

    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    inlier_cloud.paint_uniform_color([1, 0, 0])

    combined_pcd = inlier_cloud + outlier_cloud

    o3d.visualization.draw_geometries([combined_pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Plane Segmentation')
    parser.add_argument('--file', required=True, help='Path to the point cloud file')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of RANSAC iterations')
    parser.add_argument('--threshold', type=float, default=0.01, help='Distance threshold for RANSAC')

    args = parser.parse_args()

    segment_plane(args.file, args.iterations, args.threshold)



