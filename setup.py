# Copyright (c) 2025 Pan Yinghong
# All rights reserved.

import os
from setuptools import find_packages, setup

# Package metadata
NAME = "Brain-SAM"
VERSION = "1.0.0"
DESCRIPTION = "Brain-SAM"  
URL ="https://github.com/panyinghong/Brain-SAM"
AUTHOR = "Pan Yinghong"
LICENSE = "MIT"

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()


REQUIRED_PACKAGES = [
    "torch>=2.3.1",
    "torchvision>=0.18.1",
    "numpy>=1.26.4",
    "nibabel>=5.3.2",
    "SimpleITK>=2.4.1",
    "scipy>=1.15.2",
    "scikit-image>=0.25.2",
    "tqdm>=4.67.1",
    "pillow>=11.1.0",
    "pandas>=2.2.3",
]


EXTRA_PACKAGES = {
    "dev": [
        "pytest>=7.0.0",
        "black>=22.0.0",
    ],
}

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    python_requires=">=3.10.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
