# GPS-Denied Drone Navigation using Radar Odometry and ICP

A complete GNSS-independent UAV localization pipeline using simulated FMCW radar, point-to-plane ICP odometry, and PX4's EKF2 estimator. Successfully demonstrated a **fully GPS-denied autonomous 10×10 m square mission** with 0.08 m mean tracking accuracy.

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| GPS-denied autonomous mission | ✅ Complete (5/5 waypoints) |
| Mission duration | 49 seconds |
| Vision tracking error (X) | 0.08 m mean |
| Vision tracking error (Y) | 0.09 m mean |
| Mean waypoint error | 0.57 m |
| Altitude hold (std dev) | ±0.09 m at 5 m |
| Drift accumulation | None observed |

## 🏗️ Architecture
Gazebo Harmonic (ground truth)
↓ gz-transport
terrain_raycaster (3.5 Hz point cloud)
↓ /radar/pointcloud
icp_odometry (20 Hz point-to-plane ICP)
↓ /radar/odom
radar_ekf_odometry (20 Hz, GPS offset)
↓ /radar/ekf_odom
radar_to_mavros (clock sync, covariance)
↓ /mavros/vision_pose/pose_cov
PX4 EKF2 fusion
↓
Autonomous flight (GPS-denied)
## 🛠️ Stack

- **Ubuntu 24.04 LTS**
- **ROS 2 Jazzy**
- **PX4 v1.15 SITL**
- **Gazebo Harmonic**
- **Open3D** (ICP implementation)
- **MAVROS2** (MAVLink bridge)
- **QGroundControl**

## 📂 Repository Structure
radar-nav-gps-denied/
├── src/radar_processing/ # ROS 2 package (5 nodes)
│ ├── terrain_raycaster_v2.py # Simulated FMCW radar
│ ├── icp_odometry.py # Point-to-plane ICP
│ ├── radar_ekf_odometry.py # EKF wrapper
│ ├── radar_to_mavros.py # MAVROS bridge
│ └── mission_waypoints.py # 10x10m square mission
├── config/radar_city.sdf # Custom Gazebo world
├── logs/ # Flight log CSVs
├── figures/ # Analysis graphs
└── docs/ # Report and documentation


## 🚀 How to Run

### 1. Prerequisites
```bash
# ROS 2 Jazzy
sudo apt install ros-jazzy-desktop

# Gazebo Harmonic
sudo apt install gz-harmonic

# Python dependencies
pip install open3d numpy scipy matplotlib
