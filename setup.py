import os
from setuptools import find_packages
from setuptools import setup

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    readme_file = f.read()

setup(name='safemotions',
      version='0.2.1',
      packages=find_packages(),
      include_package_data=True,
      author='Jonas C. Kiemel',
      author_email='jonas.kiemel@kit.edu',
      url='https://github.com/translearn/safemotions',
      description='Learning Collision-free and Torque-limited Robot Trajectories based on Alternative Safe Behaviors.',
      long_description=readme_file,
      long_description_content_type='text/markdown',
      python_requires='>=3.5, <3.9',
      install_requires=[
          'numpy',
          'klimits',
          'matplotlib',
          'pybullet',
          'gym',
          'ray[tune]==0.8.4',
          'ray[rllib]==0.8.4',
          'tensorflow'
      ],
      )
