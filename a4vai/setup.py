from setuptools import find_packages, setup

package_name = 'a4vai'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kestrel',
    maintainer_email='kestrel@inha.edu',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node_att_ctrl = a4vai.pathFollowing.node_att_ctrl:main',
            'node_MPPI_output = a4vai.pathFollowing.node_MPPI_output:main',
            'test_wayoint = a4vai.pathFollowing.get_waypoint:main'
        ],
    },
)
