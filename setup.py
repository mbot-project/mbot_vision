from setuptools import setup, find_packages

setup(
    name='mbot_vision',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'flask',
        'ultralytics[export]' # for cone detection, details see README
    ],
)
