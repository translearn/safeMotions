import os
import re
from setuptools import find_packages
from setuptools import setup


def get_version():
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'safemotions', '__init__.py'), encoding='utf-8') as f:
        init_file = f.read()
        version = re.search(r"__version__\W*=\W*'([^']+)'", init_file)
        return version.group(1) if version is not None else '0.0.0'


with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    readme_file = f.read()

setup(name='safemotions',
      version=get_version(),
      packages=find_packages(),
      include_package_data=True,
      author='Jonas C. Kiemel',
      author_email='jonas.kiemel@kit.edu',
      url='https://github.com/translearn/safemotions',
      description='Learning Collision-free and Torque-limited Robot Trajectories based on Alternative Safe Behaviors.',
      long_description=readme_file,
      long_description_content_type='text/markdown',
      license='MIT',
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Intended Audience :: Developers",
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9'],
      python_requires='>=3.5',
      install_requires=[
          'numpy',
          'klimits>=1.1.1',
          'matplotlib',
          'pybullet',
          'gym',
          'Pillow'
      ],
      extras_require={'train': ['aiohttp==3.7.4.post0', 'aiohttp-cors==0.7.0', 'aioredis==1.3.1', 'redis==3.5.3', 'prometheus-client==0.11.0',
                                'ray[default]==1.4.1', 'ray[rllib]==1.4.1', 'tensorflow']}
      )
