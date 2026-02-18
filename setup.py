#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for Voxel Cluster Density Attention Network (VCDA-Net) Package
Brain Age Prediction with Density Attention-Enhanced Features

⚠️ IMPORTANT: This package uses SALIENCY MAPS, not GradCAM!
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding='utf-8').splitlines()
requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="vcda-net",
    version="4.0.2",
    author="Anonymous",
    author_email="anonymous@review.com",
    description="Voxel Cluster Density Attention Network (VCDA-Net) - Brain Age Prediction with Density Attention-Enhanced Features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://anonymous.review/vcda-net",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Package data - include model checkpoints and configs
    package_data={
        'vcda_net': [
            'checkpoints/*.pth',
            'configs/*.yaml',
            'configs/*.json',
        ],
    },
    include_package_data=True,
    
    # Dependencies
    install_requires=requirements,
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'vcda-net-train=vcda_net.training.train:main',
            'vcda-net-predict=vcda_net.predict:main',
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    
    # Keywords
    keywords=[
        'brain age prediction',
        'medical imaging',
        'deep learning',
        'density attention map',
        'activation map',
        'graph neural networks',
        'transformer',
        'MRI analysis',
    ],
    
    # Project URLs
    project_urls={
        'Documentation': 'https://anonymous.review/vcda-net/wiki',
        'Source': 'https://anonymous.review/vcda-net',
        'Bug Reports': 'https://anonymous.review/vcda-net/issues',
    },
    
    # License
    license='MIT',
    
    # Zip safe
    zip_safe=False,
)
