from setuptools import find_packages, setup

package_name = 'radar_processing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kanishk-j',
    maintainer_email='kanishk-j@todo.todo',
    description='Radar processing nodes',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'terrain_raycaster  = radar_processing.terrain_raycaster_v2:main',
            'icp_odometry       = radar_processing.icp_odometry:main',
            'radar_ekf_odometry = radar_processing.radar_ekf_odometry:main',
            'radar_to_mavros    = radar_processing.radar_to_mavros:main',
            'mission_waypoints  = radar_processing.mission_waypoints:main',
        ],
    },
)
