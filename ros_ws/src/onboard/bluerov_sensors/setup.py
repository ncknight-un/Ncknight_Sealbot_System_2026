from glob import glob
from setuptools import setup

package_name = "bluerov_sensors"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sealbot",
    maintainer_email="nolanknight133@icloud.com",
    description="BlueROV2 MAVROS sensor relay and status node",
    license="MIT",
    entry_points={
        "console_scripts": [
            "bluerov_sensor_node = bluerov_sensors.bluerov_sensor_node:main",
        ],
    },
)
