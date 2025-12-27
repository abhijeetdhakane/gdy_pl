#!/usr/bin/env python
"""
GDyNet-Ferro: Graph Dynamical Network for Molecular Dynamics Analysis

A PyTorch implementation of Graph Dynamical Networks trained with VAMP
(Variational Approach for Markov Processes) loss for analyzing molecular
dynamics trajectories and identifying slow dynamical features in ferroelectrics.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='gdynet-ferro',
    version='2.0.0',
    author='Abhijeet Dhakane',
    author_email='adhakane@vols.utk.edu',
    description='A Graph Dynamical Neural Network Approach for Decoding Dynamical States in Ferroelectrics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/abhijeetdhakane/gdy_pl',
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'notebooks']),
    package_data={
        'config': ['*.py'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  # Update if different
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'isort>=5.12.0',
        ],
        'wandb': [
            'wandb>=0.15.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'gdynet-train=trainer:main',
        ],
    },
    keywords=[
        'graph neural networks',
        'molecular dynamics',
        'VAMP',
        'variational approach',
        'deep learning',
        'PyTorch',
        'ferroelectrics',
        'dynamical systems',
    ],
    project_urls={
        'Documentation': 'https://github.com/abhijeetdhakane/gdy_pl/blob/main/README.md',
        'Source': 'https://github.com/abhijeetdhakane/gdy_pl',
        'Tracker': 'https://github.com/abhijeetdhakane/gdy_pl/issues',
    },
)
