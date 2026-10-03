#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State


class MissionWaypoints(Node):
    def __init__(self):
        super().__init__('mission_waypoints')

        self.pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10)

        mavros_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(State, '/mavros/state',
                                 self.state_cb, 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self.pose_cb, mavros_qos)

        self.state = None
        self.z_offset = None

        self.waypoints = [
            (0.0,   0.0, 5.0),
            (10.0,  0.0, 5.0),
            (10.0, 10.0, 5.0),
            (0.0,  10.0, 5.0),
            (0.0,   0.0, 5.0),
        ]
        self.current_wp = 0
        self.wp_start_time = None
        self.timer = self.create_timer(0.05, self.run)
        self.get_logger().info(
            'Mission: square 10x10x3 (relative)')

    def state_cb(self, msg):
        self.state = msg

    def pose_cb(self, msg):
        if self.z_offset is None:
            self.z_offset = msg.pose.position.z
            self.get_logger().info(
                f'Z offset captured: {self.z_offset:.2f} m')

    def run(self):
        if self.state is None or self.z_offset is None:
            return

        x, y, z_rel = self.waypoints[self.current_wp]
        z = z_rel 

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        self.pub.publish(msg)

        if self.wp_start_time is None:
            self.wp_start_time = self.get_clock().now()
            self.get_logger().info(
                f'WP {self.current_wp}: ({x:.1f}, {y:.1f}, {z_rel:.1f}+{self.z_offset:.2f})')
            return

        elapsed = (self.get_clock().now() - self.wp_start_time).nanoseconds / 1e9
        if elapsed > 12.0:
            self.wp_start_time = self.get_clock().now()
            self.current_wp = (self.current_wp + 1) % len(self.waypoints)
            x, y, z_rel = self.waypoints[self.current_wp]
            self.get_logger().info(
                f'WP {self.current_wp}: ({x:.1f}, {y:.1f}, {z_rel:.1f}+{self.z_offset:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = MissionWaypoints()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
