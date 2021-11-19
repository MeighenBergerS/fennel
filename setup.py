#!/usr/bin/env python

from setuptools import setup

setup(
    name="Fennel",
    version="1.0",
    description="Light-yields for tracks, and cascades",
    author="Stephan Meighen-Berger",
    author_email="stephan.meighenberger@gmail.com",
    install_requires=[
        "PyYAML",
        "numpy",
        "scipy",
    ],
    extras_require={
        "interactive": ["nbstripout", "matplotlib", "jupyter", "tqdm"],
    },
    packages=["fennel"]
)
