# Fennel

Authors:

1. Stephan Meighen-Berger, developed the Fennel Code

## Table of contents

1. [Introduction](#introduction)

2. [Citation](#citation)

3. [Documentation](#documentation)

4. [Installation](#installation)

5. [Beta](#beta)

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
@software{fennel2022@github,
  author = {Stephan Meighen-Berger},
  title = {{Fennel}: Light from tracks and cascades,
  url = {https://github.com/MeighenBergerS/fennel},
  version = {1.3.3},
  year = {2022},
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
```python
pip install fennel_seed[jax]
```

For the cpu version of jax use:
```python
pip install fennel_seed[cpu]
```

To be able to run the example notebook use:
```python
pip install fennel_seed[interactive]
```

## Beta <a name="beta"></a>
Fennel offers a few subprojects which are currently still in beta. While these projects work, they have as of yet not been designed for usability. Currently available subprojects are available offer the GitHub repository (not pip!). Subprojects are:

1. Jfennel: A Julia implementation of Fennel. This is its own branch in the repository and still requires further work and cross-checks.

2. Seed: An interface to the Geant4 code used for the parametrization. The code itself offers a Python interface for ease of use. To use this code, a Geant4 installation is required. Currently this module has only been tested in a Linux environment. The code itself is contained in the seed folder and includes some examples in the notebooks folder on how to use it.
