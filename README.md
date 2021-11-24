# Fennel

Authors:

1. Stephan Meighen-Berger, developed the Fennel Code

## Table of contents

1. [Introduction](#introduction)

2. [Citation](#citation)

3. [Documentation](#documentation)

4. [Installation](#installation)

## Introduction <a name="introduction"></a>

Welcome to Fennel!

![Logo](images/Fennel.png)

A python package to simulate the light production of particles.
It calculates the light emissions from cascades and tracks.

## Citation <a name="citation"></a>

This packages includes/uses distributions developed in

Leif Rädel, Christopher Wiebusch,\
*Calculation of the Cherenkov light yield from low energetic secondary particles accompanying high-energy muons in ice and water with Geant4 simulations*,\
Astroparticle Physics,
Volume 38,
2012,
Pages 53-67,
ISSN 0927-6505,\
https://doi.org/10.1016/j.astropartphys.2012.09.008.
(https://www.sciencedirect.com/science/article/pii/S0927650512001831)

and

https://www.institut3b.physik.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaaapwhjz

Please cite this [software](https://github.com/MeighenBergerS/fennel) using
```
@software{fennel2021@github,
  author = {Stephan Meighen-Berger},
  title = {{Fennel}: Light from tracks and cascades,
  url = {https://github.com/MeighenBergerS/fennel},
  version = {1.1.1},
  year = {2021},
}
```

and their work when using this package.

## Documentation <a name="documentation"></a>

The package provides automatically generated documentation under 
https://meighenbergers.github.io/fennel/

## Installation <a name="installation"></a>

Install using pip:
```python
pip install fennel_seed
```
[The PyPi webpage](https://pypi.org/project/fennel-seed/)

Other installation methods:
To install please clone the (repository)[https://github.com/MeighenBergerS/fennel] or download the latest release. Then
follow the instructions given in INSTALL.txt.
Note this should install all necessary components.

Or install using the setup.py

Please note that JAX is not included in the basic installation.
To use this option, please also install JAX or install fennel using:
Install using pip:
```python
pip install fennel_seed[jax]
```

To be able to run the example notebook use
Install using pip:
```python
pip install fennel_seed[interactive]
```
