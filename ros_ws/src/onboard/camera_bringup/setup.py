import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'camera_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # XML LAUNCH FILES:
        (os.path.join('share', package_name, 'launch'), glob('launch/*.xml')),
        # XML and YAML CONFIG FILES:
        (os.path.join('share', package_name, 'config'), glob('config/*.xml') + glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sealbot',
    maintainer_email='nolanknight2026@u.northwestern.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['tests'],
    entry_points={
        'console_scripts': [
        ],
    },
)
