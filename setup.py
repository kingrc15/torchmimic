import os
import re
import setuptools

try:
    # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:
    # for pip <= 9.0.3
    from pip.req import parse_requirements

# for simplicity we actually store the version in the __version__ attribute in the source
here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, "torchmimic", "__init__.py")) as f:
    meta_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M
    )
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

required = [
    "torch",
    "scikit-learn",
    "numpy",
    "wandb",
]


def read(fname="README.md"):
    with open(
        os.path.join(os.path.dirname(__file__), fname), encoding="utf-8"
    ) as cfile:
        return cfile.read()


setuptools.setup(
    name="torchmimic",
    version=version,
    author="Ryan King",
    author_email="kingrc15@tamu.edu",
    description="MIMIC Benchmark in PyTorch.",
    license="MIT",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/kingrc15/torchmimic",
    packages=setuptools.find_packages(),
    package_data={
        "torchmimic": [
            "data/discretizers/discretizer_config.json",
            "data/normalizers/*.normalizer",
        ]
    },
    include_package_data=True,
    install_requires=required,
    extras_require={
        "local": ["pytest", "black", "setuptools", "sphinx"],
    },
    python_requires="~=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
