from setuptools import find_packages, setup

package_name = 'py_pubsub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'ament_package', 'numpy'],  # Add any other dependencies
    zip_safe=True,
    maintainer='moonjung',
    maintainer_email='moonjung42@naver.com',
    description='Examples of minimal publisher/subscriber using rclpy',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = py_pubsub.Image2Plan:main',
            'listener = py_pubsub.Plan2WP:main',
        ],
    },
)

