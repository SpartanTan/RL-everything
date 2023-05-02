from setuptools import setup

package_name = 'gym_pubsub'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhicun',
    maintainer_email='tanzc9866@126.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'gymtalker = gym_pubsub.publisher_member_function:main',
                'gymlistener = gym_pubsub.subscriber_member_function:main',
        ],
    },
)
