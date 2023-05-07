from setuptools import setup, find_packages
import os.path as path

dir = path.dirname(path.abspath(__file__))
with open(path.join(dir, 'requirements.txt'), 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='recommendation_models',
    version='0.0.0',
    description='recommendation_models',
    packages=find_packages(),
    install_requires = requirements,
    
)