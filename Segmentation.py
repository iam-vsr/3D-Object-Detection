def ransac(pcd, iterations=100, tolerance=0.3):
    if pcd.is_empty():
        return pcd, pcd

    plane_model, inliers = pcd.segment_plane(distance_threshold=tolerance,
                                             ransac_n=3,
                                             num_iterations=iterations)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    return inlier_cloud, outlier_cloud

