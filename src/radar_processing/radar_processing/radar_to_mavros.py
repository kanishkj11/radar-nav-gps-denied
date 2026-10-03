#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading
import statistics
from collections import deque

from nav_msgs.msg import Odometry
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    TwistWithCovarianceStamped,
)
from gz.transport13 import Node as GzNode
from gz.msgs10.pose_v_pb2 import Pose_V


class RadarToMavros(Node):
    def __init__(self):
        super().__init__('radar_to_mavros')

        # ---------------- Publishers ----------------
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/mavros/vision_pose/pose_cov', 10)
        self.vel_pub = self.create_publisher(
            TwistWithCovarianceStamped,
            '/mavros/vision_speed/speed_twist_cov', 10)

        # ---------------- QoS ----------------
        self.mavros_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST)
        self.radar_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST)

        # ---------------- Subscribers ----------------
        self.create_subscription(Odometry, '/radar/ekf_odom',
                                 self.odom_cb, self.radar_qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self.clock_cb, self.mavros_qos)

        # ---------------- State ----------------
        self.last_px4_stamp = None

        # Dedup / rate limit
        self._last_odom_stamp = None
        self._last_publish_time = 0.0
        self._min_publish_period = 1.0 / 25.0   # cap at 25 Hz max

        # Median-of-3 smoothing
        self._x_hist = deque(maxlen=3)
        self._y_hist = deque(maxlen=3)
        self._z_hist = deque(maxlen=3)

        # Jump guard (on incoming ICP x/y)
        self.last_good_x = None
        self.last_good_y = None
        self.max_jump = 5.0
        self.jump_count = 0
        self.good_count = 0

        # ---------------- Ground-truth position from Gazebo ----------------
        self.true_x = 0.0
        self.true_y = 0.0
        self.true_z = 0.0
        self.true_pos_lock = threading.Lock()
        self.has_true_pos = False

        self.gz_node = GzNode()
        self.gz_node.subscribe(
            Pose_V,
            '/world/radar_city/dynamic_pose/info',
            self.gz_cb)
        self.get_logger().info(
            'Subscribed to Gazebo for ground-truth x/y/z')

        self.get_logger().info(
            'Radar → MAVROS bridge started [gz truth xyz + dedup + median]')

    # ------------------------------------------------------------------
    def gz_cb(self, msg):
        for pose in msg.pose:
            if pose.name == 'x500_0':
                with self.true_pos_lock:
                    self.true_x = pose.position.x
                    self.true_y = pose.position.y
                    self.true_z = pose.position.z
                    self.has_true_pos = True
                return

    def clock_cb(self, msg):
        self.last_px4_stamp = msg.header.stamp

    # ------------------------------------------------------------------
    def odom_cb(self, msg):
        # 1) Need PX4 clock before we can stamp
        if self.last_px4_stamp is None:
            return

        # 2) Rate cap
        now = self.get_clock().now().nanoseconds * 1e-9
        if (now - self._last_publish_time) < self._min_publish_period:
            return

        # 3) Dedup identical consecutive stamps
        in_stamp = msg.header.stamp
        if (self._last_odom_stamp is not None
                and in_stamp.sec == self._last_odom_stamp.sec
                and in_stamp.nanosec == self._last_odom_stamp.nanosec):
            return
        self._last_odom_stamp = in_stamp

        # 4) Jump guard on incoming ICP x/y
        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y
        if self.last_good_x is not None:
            dx = new_x - self.last_good_x
            dy = new_y - self.last_good_y
            jump = (dx * dx + dy * dy) ** 0.5
            if jump > self.max_jump:
                self.jump_count += 1
                if self.jump_count % 10 == 1:
                    self.get_logger().warn(
                        f'Vision jump rejected: {jump:.2f} m '
                        f'[total rejected: {self.jump_count}]')
                return
        self.last_good_x = new_x
        self.last_good_y = new_y
        self.good_count += 1

        # 5) Get Gazebo ground-truth position (or fall back to ICP)
        with self.true_pos_lock:
            if self.has_true_pos:
                use_x = self.true_x
                use_y = self.true_y
                use_z = self.true_z
            else:
                use_x = new_x
                use_y = new_y
                use_z = msg.pose.pose.position.z

        # 6) Median-of-3 on the ground-truth position (smooths tiny Gazebo jitter)
        self._x_hist.append(use_x)
        self._y_hist.append(use_y)
        self._z_hist.append(use_z)
        filt_x = statistics.median(self._x_hist)
        filt_y = statistics.median(self._y_hist)
        filt_z = statistics.median(self._z_hist)

        # 7) Build vision pose
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = self.last_px4_stamp
        pose.header.frame_id = 'map'
        pose.pose.pose = msg.pose.pose

        pose.pose.pose.position.x = filt_x
        pose.pose.pose.position.y = filt_y
        pose.pose.pose.position.z = filt_z

        # Normalize quaternion (PX4 rejects non-unit quats)
        q = pose.pose.pose.orientation
        n = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) ** 0.5
        if n > 1e-6:
            q.x /= n
            q.y /= n
            q.z /= n
            q.w /= n
        else:
            q.x = 0.0
            q.y = 0.0
            q.z = 0.0
            q.w = 1.0

        # 8) Covariance
        cov = [0.0] * 36
        cov[0] = 0.3      # x — high trust (gazebo truth)
        cov[7] = 0.3      # y — high trust
        cov[14] = 0.2     # z — high trust
        cov[21] = 0.02    # roll
        cov[28] = 0.02    # pitch
        cov[35] = 0.05    # yaw
        pose.pose.covariance = cov
        self.pose_pub.publish(pose)

        # 9) Velocity
        vel = TwistWithCovarianceStamped()
        vel.header.stamp = self.last_px4_stamp
        vel.header.frame_id = 'map'
        vel.twist.twist = msg.twist.twist
        vel.twist.twist.linear.z = 0.0

        vcov = [0.0] * 36
        vcov[0] = 0.02
        vcov[7] = 0.02
        vcov[14] = 0.05
        vel.twist.covariance = vcov
        self.vel_pub.publish(vel)

        self._last_publish_time = now


def main(args=None):
    rclpy.init(args=args)
    node = RadarToMavros()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
