#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
import numpy as np
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Path, Odometry
from tf_transformations import quaternion_from_euler


class ICPOdometry(Node):
    def __init__(self):
        super().__init__('icp_odometry')

        self.prev_cloud = None
        self.last_transform = np.eye(4)
        self.prev_time = None
        self.first_pub = False

        self.x = self.y = self.z = 0.0
        self.yaw = 0.0
        self.vx = self.vy = 0.0

        self.vel_x = 0.0
        self.vel_y = 0.0
        self.armed = False
        self.mode = 'UNKNOWN'

        self.alpha = 1.0
        self.motion_threshold = 0.02
        self.max_translation = 0.25
        self.voxel_size = 0.15
        self.fitness_min = 0.6
        self.rmse_max = 0.3
        self.icp_fail_count = 0
        self.max_fail_before_reset = 2

        self.sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST)
        self.mavros_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(PointCloud2, '/radar/pointcloud',
                                 self.cloud_cb, self.sensor_qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self.pose_cb, self.mavros_qos)
        self.create_subscription(TwistStamped,
                                 '/mavros/local_position/velocity_local',
                                 self.vel_cb, self.mavros_qos)
        self.create_subscription(State, '/mavros/state',
                                 self.state_cb, 10)

        self.path_pub = self.create_publisher(Path, '/radar/path', 10)
        self.odom_pub = self.create_publisher(Odometry, '/radar/odom', 10)
        self.path = Path()
        self.path.header.frame_id = 'map'

        self.timer = self.create_timer(0.05, self.publish_odom)
        self.get_logger().info('ICP odometry started [ZUPT on state+velocity]')

    def pose_cb(self, msg):
        q = msg.pose.orientation
        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

    def vel_cb(self, msg):
        self.vel_x = msg.twist.linear.x
        self.vel_y = msg.twist.linear.y

    def state_cb(self, msg):
        self.armed = msg.armed
        self.mode = msg.mode

    def ros_to_o3d(self, msg):
        pts = np.array([[p[0], p[1], p[2]]
                        for p in pc2.read_points(msg,
                                                 field_names=('x', 'y', 'z'),
                                                 skip_nans=True)],
                       dtype=np.float64)
        cloud = o3d.geometry.PointCloud()
        if pts.shape[0] == 0:
            return cloud
        cloud.points = o3d.utility.Vector3dVector(pts)
        cloud = cloud.voxel_down_sample(voxel_size=self.voxel_size)
        if len(cloud.points) >= 10:
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.5, max_nn=30))
        return cloud

    def cloud_cb(self, msg):
        current_cloud = self.ros_to_o3d(msg)
        if len(current_cloud.points) < 10:
            return
        if self.prev_cloud is None:
            self.prev_cloud = current_cloud
            self.prev_time = msg.header.stamp
            return

        threshold = 4.0
        try:
            reg = o3d.pipelines.registration.registration_icp(
                self.prev_cloud, current_cloud, threshold,
                self.last_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=40))
        except Exception as e:
            self.get_logger().warn(f'ICP exception: {e}')
            return

        T = reg.transformation
        fitness = reg.fitness
        rmse = reg.inlier_rmse

        if not np.isfinite(T).all():
            self.prev_cloud = current_cloud
            return

        failed = (fitness < self.fitness_min or rmse > self.rmse_max
                  or len(current_cloud.points) < 100)
        if failed:
            self.icp_fail_count += 1
            self.prev_cloud = current_cloud
            if self.icp_fail_count >= self.max_fail_before_reset:
                self.last_transform = np.eye(4)
            return
        self.icp_fail_count = 0

        dx = -T[0, 3]
        dy = -T[1, 3]
        
        dx *= 1.25
        dy *= 1.25

        self.last_transform = T
        self.prev_cloud = current_cloud

        # ---------- ZUPT ----------
        # Rule 1: if not armed, force zero motion
        if not self.armed:
            return

        # Rule 2: if PX4 mode is not OFFBOARD, force zero motion
        if 'OFFBOARD' not in self.mode:
            return

        # Rule 3: if PX4 speed < 0.15 m/s, force zero motion
        speed = math.sqrt(self.vel_x ** 2 + self.vel_y ** 2)
        if speed < 0.15:
            return
        # --------------------------

        if np.linalg.norm([dx, dy]) > self.max_translation:
            self.last_transform = np.eye(4)
            return

        if abs(dx) < self.motion_threshold:
            dx = 0.0
        if abs(dy) < self.motion_threshold:
            dy = 0.0

        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx_map = c * dx - s * dy
        dy_map = s * dx + c * dy
        self.x += dx_map
        self.y += dy_map
        self.vx = 0.9 * self.vx + 0.1 * dx_map
        self.vy = 0.9 * self.vy + 0.1 * dy_map
        if not self.first_pub:
            self.get_logger().info(
                f'First odom: x={self.x:.3f} y={self.y:.3f}')
            self.first_pub = True

    def publish_odom(self):
        stamp = self.get_clock().now().to_msg()
        q = quaternion_from_euler(0, 0, self.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = self.z
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.pose.covariance = [
            0.05, 0, 0, 0, 0, 0,
            0, 0.05, 0, 0, 0, 0,
            0, 0, 0.05, 0, 0, 0,
            0, 0, 0, 0.02, 0, 0,
            0, 0, 0, 0, 0.02, 0,
            0, 0, 0, 0, 0, 0.02]
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.linear.y = self.vy
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = ICPOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
