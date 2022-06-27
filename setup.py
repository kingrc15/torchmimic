import os
import re
import setuptools


# for simplicity we actually store the version in the __version__ attribute in the source
here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, 'torchmimic', '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

required = [
    'scikit-learn',
    'numpy'
    'torch',
    'wandb',
]

setuptools.setup(
    name="torchmimic",
    version=version,
    author="Ryan King",
    author_email="kingrc15@tamu.edu",
    description="MIMIC Benchmark in PyTorch.",
    url="https://github.com/stmilab/mimic-benchmark-pytorch",
    packages=setuptools.find_packages(),
    install_requires=required, # TODO
    python_requires='~=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)