from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'bluerov_sensors'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('bluerov_sensors/launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='BlueROV2 MAVROS sensor data collection node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'bluerov_sensor_node = bluerov_sensors.bluerov_sensor_node:main',
        ],
    },
)
