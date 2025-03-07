#!/usr/bin/env python3

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField

class PointCloudClipper:
    def __init__(self):
        rospy.init_node('point_cloud_clipper')
        self.pub = rospy.Publisher('/p', PointCloud2, queue_size=10)
        rospy.Subscriber('/ouster/points', PointCloud2, self.point_cloud_callback)

        # Parameters
        self.lidar_height = 1.1  # Height of the LiDAR above ground
        self.min_z = self.lidar_height - 2.5   # Minimum Z value (1.1 - 0.5)
        self.max_z = self.lidar_height - 0.8   # Maximum Z value (1.1 + 1.0)
        self.ground_clip_z = self.lidar_height - 1.75  # Clip ground below this Z value

    def point_cloud_callback(self, msg):
        header = msg.header
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        
        filtered_points = [
            point for point in points
            if (self.min_z <= point[2] <= self.max_z and
                (point[0]**2 + point[1]**2)**0.5 > 0.75 and
                point[2] > self.ground_clip_z)  # Clip ground below this Z value
        ]

        # Create a new PointCloud2 message
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        filtered_pc_msg = pc2.create_cloud(header, fields, filtered_points)

        # Publish the filtered point cloud
        self.pub.publish(filtered_pc_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        clipper = PointCloudClipper()
        clipper.run()
    except rospy.ROSInterruptException:
        pass
