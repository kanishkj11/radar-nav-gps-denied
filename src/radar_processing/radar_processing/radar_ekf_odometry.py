#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler
from std_msgs.msg import Float64


class RadarEKF(Node):
    def __init__(self):
        super().__init__('radar_ekf_odometry')
        self.icp_x = 0.0
        self.icp_y = 0.0
        self.z = None
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_set = False
        self.first_icp_received = False

        self.mavros_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(Odometry, '/radar/odom',
                                 self.odom_cb, 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self.local_pose_cb, self.mavros_qos)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt',
                                 self.baro_cb, self.mavros_qos)

        self.pub = self.create_publisher(Odometry, '/radar/ekf_odom', 10)
        self.timer = self.create_timer(0.05, self.run)  # 20 Hz
        self.get_logger().info('Radar EKF started (ROS 2)')

    def local_pose_cb(self, msg):
        # Initial lock (for yaw)
        if not self.offset_set and self.first_icp_received:
            q = msg.pose.orientation
            siny = 2 * (q.w * q.z + q.x * q.y)
            cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
            self.yaw = math.atan2(siny, cosy)
            self.offset_set = True

        # Continuously re-align: keep vision tracking GPS while GPS is on
        if self.first_icp_received:
            self.offset_x = msg.pose.position.x - self.icp_x
            self.offset_y = msg.pose.position.y - self.icp_y

        # z from local_position (PX4's own estimate)
        self.z = msg.pose.position.z


    def odom_cb(self, msg):
        self.icp_x = msg.pose.pose.position.x
        self.icp_y = msg.pose.pose.position.y
        self.vx = msg.twist.twist.linear.x
        self.vy = msg.twist.twist.linear.y
        if not self.first_icp_received:
            self.first_icp_received = True

    def baro_cb(self, msg):
        # Use BARO as the z source for vision
        self.z = msg.data

    def run(self):
        if self.z is None:
            return

        if self.offset_set:
            x = self.icp_x + self.offset_x
            y = self.icp_y + self.offset_y
        else:
            x = self.icp_x
            y = self.icp_y

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = self.z
        q = quaternion_from_euler(0, 0, self.yaw)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = self.vy
        self.pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = RadarEKF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
