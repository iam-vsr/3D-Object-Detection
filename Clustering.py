import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def dbscan(pcd, eps=0.45, min_points=7, print_progress=False, debug=False):
    # Convert point cloud to numpy array for faster processing
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        return pcd, np.array([])

    verbosityLevel = o3d.utility.VerbosityLevel.Warning
    if debug:
        verbosityLevel = o3d.utility.VerbosityLevel.Debug
    with o3d.utility.VerbosityContextManager(verbosityLevel):
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))

    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")

    # Avoid unnecessary color assignment for better performance
    if max_label > 0:
        colors = plt.get_cmap("tab20")(labels / max_label)
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcd, labels


