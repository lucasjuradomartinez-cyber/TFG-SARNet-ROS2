from setuptools import setup
import os
from glob import glob

package_name = 'sarnet_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Esto es para que ROS2 encuentre tu carpeta de pesos al instalar
        (os.path.join('share', package_name, 'weights'), glob('sarnet_py/weights/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tfg_rfc',
    maintainer_email='tfg_rfc@todo.com',
    description='Segmentacion SARNet en ROS2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segmentation_node = sarnet_py.segmentation_node:main',
            'camera_simulator = sarnet_py.camera_simulator:main',
            'zed_segmentation_node = sarnet_py.zed_segmentation_node:main',
        ],
    },
)
