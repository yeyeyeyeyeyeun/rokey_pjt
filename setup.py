from setuptools import find_packages, setup

package_name = 'rokey_pjt'

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
    maintainer='rokey',
    maintainer_email='rokey@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_checker = rokey_pjt.depth_checker:main',
			'depth_checker_click = rokey_pjt.depth_checker_click:main',
            'yolo_depth_checker = rokey_pjt.3_tb4_yolo_bbox_depth_checker:main',
            'tf_trans = rokey_pjt.4_tb4_tf_transform:main',
            'test_node = rokey_pjt.5_test:main',
            'object_xyz_marker = rokey_pjt.6_rviz_marking:main',
            'nav_to_pose_sc = rokey_pjt.7_nav_to_pose_sc:main',
            'nav_through_poses_sc = rokey_pjt.8_nav_through_poses_sc:main',
            'follow_waypoints_sc = rokey_pjt.9_follow_waypoints_sc:main',
        ],
    },
)
