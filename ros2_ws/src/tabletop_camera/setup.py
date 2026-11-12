from setuptools import find_packages, setup

package_name = 'tabletop_camera'

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
    maintainer='vdavid21',
    maintainer_email='davidvincze21@gmail.com',
    description='ROS 2 node that publishes images from a CSI camera on NVIDIA Jetson.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_node = tabletop_camera.camera_node:main'
        ],
    },
)
