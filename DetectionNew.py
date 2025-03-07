#!/usr/bin/env python3

import time
import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import Downsampling as ds
import Segmentation as sg
import Clustering as cl
import BoundingBoxes as bb
from concurrent.futures import ThreadPoolExecutor

# Global variable to track active markers
active_marker_ids = set()

def ros_to_open3d(ros_cloud):
    # Convert ROS PointCloud2 message to Open3D point cloud
    points = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True)], dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def open3d_to_ros(point_cloud, frame_id="os_sensor"):
    # Convert Open3D point cloud to ROS PointCloud2 message
    points = np.asarray(point_cloud.points, dtype=np.float32)
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
        pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
        pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
    ]
    
    return pc2.create_cloud(header, fields, points)

def create_marker(bbox, marker_id, distance, frame_id="os_sensor"):
    markers = MarkerArray()
    
    # Create a ROS Marker for the bounding box
    box_marker = Marker()
    box_marker.header.frame_id = frame_id
    box_marker.header.stamp = rospy.Time.now()
    box_marker.ns = "detection"
    box_marker.id = marker_id
    box_marker.type = Marker.CUBE
    box_marker.action = Marker.ADD

    # Calculate box dimensions and position
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    extent = max_bound - min_bound
    center = bbox.get_center()

    box_marker.scale.x = extent[0]
    box_marker.scale.y = extent[1]
    box_marker.scale.z = extent[2]

    box_marker.pose.position.x = center[0]
    box_marker.pose.position.y = center[1]
    box_marker.pose.position.z = center[2]

    # Set marker orientation
    box_marker.pose.orientation.w = 1.0  # No rotation for simplicity
    box_marker.pose.orientation.x = 0.0
    box_marker.pose.orientation.y = 0.0
    box_marker.pose.orientation.z = 0.0

    # Set marker color
    box_marker.color.a = 0.5  # Transparency
    box_marker.color.r = 1.0
    box_marker.color.g = 0.0
    box_marker.color.b = 0.0

    markers.markers.append(box_marker)
    
    # Create a text marker to display the distance
    text_marker = Marker()
    text_marker.header.frame_id = frame_id
    text_marker.header.stamp = rospy.Time.now()
    text_marker.ns = "detection"
    text_marker.id = marker_id + 1000  # Ensure a unique ID for text marker
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD

    text_marker.pose.position.x = center[0]
    text_marker.pose.position.y = center[1]
    text_marker.pose.position.z = center[2] + extent[2] / 2 + 0.5  # Position above the bounding box

    text_marker.scale.z = 0.5  # Font size
    text_marker.color.a = 1.0
    text_marker.color.r = 1.0
    text_marker.color.g = 1.0
    text_marker.color.b = 1.0

    text_marker.text = f"{distance:.2f} m"  # Distance text

    markers.markers.append(text_marker)

    return markers

def apply_roi_filter(pcd, x_limits=(-5, 5), y_limits=(-2.5, 2.5), z_limits=(-1.5, 2)):
    # Apply Region of Interest (ROI) filtering based on x, y, z limits
    points = np.asarray(pcd.points)
    mask = (
        (points[:, 0] >= x_limits[0]) & (points[:, 0] <= x_limits[1]) &  # Forward clipping (X)
        (points[:, 1] >= y_limits[0]) & (points[:, 1] <= y_limits[1]) &  # Lateral clipping (Y)
        (points[:, 2] >= z_limits[0]) & (points[:, 2] <= z_limits[1])    # Height clipping (Z)
    )
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    return pcd

def detection_pipeline(pcd, downsample_factor=0.25, iterations=100, tolerance=0.3, eps=0.4, min_points=5):
    # Apply ROI filtering before processing
    pcd = apply_roi_filter(pcd)
    
    # Pipeline optimization: Turn off debug prints to avoid unnecessary I/O operations
    downsample_pcd = ds.downsample(pcd, downsample_factor)

    inlier_pts, outlier_pts = sg.ransac(downsample_pcd, iterations=iterations, tolerance=tolerance)

    outlier_pts, labels = cl.dbscan(outlier_pts, eps=eps, min_points=min_points, print_progress=False)

    bboxes = bb.oriented_bbox(outlier_pts, labels)

    return outlier_pts, inlier_pts, bboxes

def point_cloud_callback(msg):
    global active_marker_ids

    pcd = ros_to_open3d(msg)
    
    # Process detection pipeline asynchronously to reduce callback blocking time
    with ThreadPoolExecutor() as executor:
        future = executor.submit(detection_pipeline, pcd, 0.25, 100, 0.3, 0.4, 5)
        outlier_pts, inlier_pts, bboxes = future.result()

    # Calculate distances from the origin
    distances = [np.linalg.norm(bbox.get_center()) for bbox in bboxes]

    # Convert processed point cloud and bounding boxes to ROS messages
    outlier_msg = open3d_to_ros(outlier_pts)
    inlier_msg = open3d_to_ros(inlier_pts)

    current_marker_ids = set()
    markers = MarkerArray()

    for i, (bbox, distance) in enumerate(zip(bboxes, distances)):
        marker_id = i
        current_marker_ids.add(marker_id)

        # Create markers
        marker = create_marker(bbox, marker_id, distance)
        markers.markers.extend(marker.markers)

    # Remove markers that are no longer active
    markers_to_remove = MarkerArray()
    for marker_id in active_marker_ids - current_marker_ids:
        remove_marker = Marker()
        remove_marker.header.frame_id = "os_sensor"
        remove_marker.header.stamp = rospy.Time.now()
        remove_marker.ns = "detection"
        remove_marker.id = marker_id
        remove_marker.type = Marker.CUBE
        remove_marker.action = Marker.DELETE

        remove_text_marker = Marker()
        remove_text_marker.header.frame_id = "os_sensor"
        remove_text_marker.header.stamp = rospy.Time.now()
        remove_text_marker.ns = "detection"
        remove_text_marker.id = marker_id + 1000  # Text marker ID
        remove_text_marker.type = Marker.TEXT_VIEW_FACING
        remove_text_marker.action = Marker.DELETE

        markers_to_remove.markers.append(remove_marker)
        markers_to_remove.markers.append(remove_text_marker)

    # Publish the results
    outlier_pub.publish(outlier_msg)
    inlier_pub.publish(inlier_msg)
    marker_pub.publish(markers_to_remove)
    marker_pub.publish(markers)

    # Update active marker IDs
    active_marker_ids = current_marker_ids

if __name__ == "__main__":
    rospy.init_node('detection_node')

    # Publishers for processed data
    outlier_pub = rospy.Publisher('/detection/outliers', PointCloud2, queue_size=10)  # Increased queue size for better performance
    inlier_pub = rospy.Publisher('/detection/inliers', PointCloud2, queue_size=10)
    marker_pub = rospy.Publisher('/detection/markers', MarkerArray, queue_size=10)

    # Subscriber for the raw point cloud data from the bag file
    rospy.Subscriber('/p', PointCloud2, point_cloud_callback, queue_size=1)  # Low latency by minimizing queue

    rospy.spin()
