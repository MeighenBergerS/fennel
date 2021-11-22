#!/usr/bin/env python

import pathlib
from setuptools import setup

# Parent directory
HERE = pathlib.Path(__file__).parent

# The readme file
README = (HERE / "README.md").read_text()

setup(
    name="fennel-seed",
    version="1.0",
    description="Light-yields for tracks, and cascades",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Stephan Meighen-Berger",
    author_email="stephan.meighenberger@gmail.com",
    url='https://github.com/MeighenBergerS/fennel',
    license="MIT",
    install_requires=[
        "PyYAML",
        "numpy",
        "scipy",
    ],
    extras_require={
        "interactive": ["nbstripout", "matplotlib", "jupyter", "tqdm"],
    },
    packages=["fennel"],
    package_data={'fennel': ["data/*.pkl"]},
    include_package_data=True
)
