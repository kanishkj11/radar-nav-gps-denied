#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TwistStamped, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import TransformBroadcaster

from gz.transport13 import Node as GzNode
from gz.msgs10.pose_v_pb2 import Pose_V


class TerrainRaycaster(Node):
    def __init__(self):
        super().__init__('terrain_raycaster')

        self.max_range = 60.0
        self.fov = np.deg2rad(180.0)

        rng = np.random.default_rng(42)
        pillar_list = []
        for x0 in np.arange(-80, 81, 4):
            for y0 in np.arange(-80, 81, 4):
                if np.hypot(x0, y0) < 3.0:
                    continue
                pillar_list.append([
                    x0 + rng.uniform(-1.5, 1.5),
                    y0 + rng.uniform(-1.5, 1.5),
                    rng.uniform(1.5, 3.5), rng.uniform(1.5, 3.5),
                    rng.uniform(8.0, 20.0), 0.0])
        for _ in range(300):
            angle = rng.uniform(0, 2*np.pi)
            radius = rng.uniform(5.0, 15.0)
            pillar_list.append([
                radius * np.cos(angle) + rng.uniform(-2, 2),
                radius * np.sin(angle) + rng.uniform(-2, 2),
                rng.uniform(0.8, 1.5), rng.uniform(0.8, 1.5),
                rng.uniform(6.0, 15.0), 0.0])

        self.pillars_array = np.array(pillar_list)
        self.get_logger().info(f'Pillars created: {len(pillar_list)}')

        self.cloud_pub = self.create_publisher(
            PointCloud2, '/radar/pointcloud', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Ground truth from Gazebo
        self.gt_x = 0.0
        self.gt_y = 0.0
        self.gt_z = 0.0
        self.gt_quat = [0.0, 0.0, 0.0, 1.0]
        self.has_pose = False
        self.gt_lock = threading.Lock()

        self.gz_node = GzNode()
        self.gz_node.subscribe(
            Pose_V,
            '/world/radar_city/dynamic_pose/info',
            self.gz_cb)
        self.get_logger().info('Subscribed to Gazebo Pose_V ground truth')

        # Velocity from MAVROS for Doppler
        self.velocity_world = np.zeros(3)
        mavros_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(
            TwistStamped, '/mavros/local_position/velocity_local',
            self.velocity_callback, mavros_qos)

        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='doppler', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Two separate timers: cloud at 3.5 Hz, TF at 20 Hz
        self.create_timer(1.0 / 3.5, self.loop)        # 3.5 Hz cloud
        self.create_timer(1.0 / 20.0, self.tf_loop)    # 20 Hz TF

    def gz_cb(self, msg):
        for pose in msg.pose:
            if pose.name == 'x500_0':
                with self.gt_lock:
                    self.gt_x = pose.position.x
                    self.gt_y = pose.position.y
                    self.gt_z = pose.position.z
                    q = pose.orientation
                    self.gt_quat = [q.x, q.y, q.z, q.w]
                    self.has_pose = True
                return

    def velocity_callback(self, msg):
        self.velocity_world = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z])

    def publish_tf(self, position, quat):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        self.tf_broadcaster.sendTransform(t)

    def tf_loop(self):
        """Publish TF at 20 Hz independent of cloud rate."""
        with self.gt_lock:
            if not self.has_pose:
                return
            position = np.array([self.gt_x, self.gt_y, self.gt_z])
            quat = self.gt_quat[:]
        self.publish_tf(position, quat)

    def loop(self):
        """Publish point cloud at 3.5 Hz."""
        with self.gt_lock:
            if not self.has_pose:
                return
            position = np.array([self.gt_x, self.gt_y, self.gt_z])
            quat = self.gt_quat[:]

        rotation = R.from_quat(quat)

        body_rays = self.generate_body_rays()
        world_rays = rotation.apply(body_rays)
        best_t, hit_mask = self.cast_all_rays(position, world_rays)
        if not np.any(hit_mask):
            self._publish_cloud([])
            return

        t_hits = best_t[hit_mask]
        ray_hits = world_rays[hit_mask]
        world_points = position[None, :] + t_hits[:, None] * ray_hits
        rel = world_points - position[None, :]
        body_rel = rotation.inv().apply(rel)

        lengths = np.linalg.norm(rel, axis=1)
        los = rel / np.maximum(lengths[:, None], 1e-6)
        dopplers = -los @ self.velocity_world

        points = np.column_stack([body_rel, dopplers]).tolist()
        self._publish_cloud(points)

    def generate_body_rays(self):
        az = np.linspace(-self.fov/2, self.fov/2, 40)
        el = np.linspace(-0.15, 0.15, 8)
        AA, EE = np.meshgrid(az, el, indexing='ij')
        x = np.cos(EE) * np.cos(AA)
        y = np.cos(EE) * np.sin(AA)
        z = np.sin(EE)
        rays = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return rays

    def cast_all_rays(self, position, world_rays):
        P = self.pillars_array
        N = P.shape[0]
        M = world_rays.shape[0]
        cx, cy, w, d, h, base = (P[:,0], P[:,1], P[:,2],
                                  P[:,3], P[:,4], P[:,5])
        bounds = [(cx - w/2, cx + w/2),
                  (cy - d/2, cy + d/2),
                  (base, base + h)]
        t_min = np.zeros((M, N))
        t_max = np.full((M, N), self.max_range)
        for axis, (mn, mx) in enumerate(bounds):
            o = position[axis]
            r_axis = world_rays[:, axis][:, None]
            mn_row = mn[None, :]
            mx_row = mx[None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                t1 = (mn_row - o) / r_axis
                t2 = (mx_row - o) / r_axis
            tlow = np.minimum(t1, t2)
            thigh = np.maximum(t1, t2)
            parallel = np.abs(r_axis) < 1e-6
            inside = (o >= mn_row) & (o <= mx_row)
            parallel_m = np.broadcast_to(parallel, (M, N))
            inside_m = np.broadcast_to(inside, (M, N))
            tlow = np.where(parallel_m,
                            np.where(inside_m, -np.inf, np.inf), tlow)
            thigh = np.where(parallel_m,
                             np.where(inside_m, np.inf, -np.inf), thigh)
            t_min = np.maximum(t_min, tlow)
            t_max = np.minimum(t_max, thigh)
        valid = (t_min <= t_max) & (t_min >= 0.0)
        t_min_valid = np.where(valid, t_min, np.inf)
        best_t = np.min(t_min_valid, axis=1)
        return best_t, best_t < np.inf

    def _publish_cloud(self, points):
        header = PointCloud2().header
        # Use zero stamp — RViz uses latest available TF
        header.stamp.sec = 0
        header.stamp.nanosec = 0
        header.frame_id = 'base_link'
        cloud = pc2.create_cloud(header, self.fields, points)
        self.cloud_pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = TerrainRaycaster()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
