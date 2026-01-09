# -*- coding: utf-8 -*-
"""
Fennel: Light-yield simulation for particle physics.

This package implements the Aachen parametrization for calculating
Cherenkov light yields from particle tracks and cascades in transparent media.

Main Components
---------------
Fennel : Main interface class for light yield calculations
config : Global configuration dictionary

New in v2.0
-----------
Result containers (TrackYieldResult, EMYieldResult, HadronYieldResult):
    Structured result objects that make it easier to work with calculation outputs

Enhanced API methods (*_v2):
    Improved versions with better validation and result containers

Convenience methods:
    quick_track(), quick_cascade(), calculate() for simplified usage

Validation utilities:
    ValidationError for clear, helpful error messages

Examples
--------
>>> from fennel import Fennel, config
>>> config["general"]["random state seed"] = 42
>>> fennel = Fennel()

Classic API (still supported):
>>> dcounts, angles = fennel.track_yields(energy=100.0, wavelengths=[400.0])

Enhanced v2.0 API:
>>> result = fennel.track_yields_v2(100.0)
>>> print(result)  # TrackYieldResult with structured output
>>> total_photons = np.trapezoid(result.dcounts, wavelengths)

Convenience methods:
>>> result = fennel.quick_track(100.0)  # Minimal parameters
>>> result = fennel.calculate(100.0, particle=11)  # Auto-detect type
"""

from .config import config
from .fennel import Fennel
from .results import EMYieldResult, HadronYieldResult, TrackYieldResult
from .validation import ValidationError

__all__ = [
    "Fennel",
    "config",
    # v2.0 result containers
    "TrackYieldResult",
    "EMYieldResult",
    "HadronYieldResult",
    # v2.0 validation
    "ValidationError",
]

# Version of the fennel package
__version__ = "2.0.0"
__author__ = "Stephan Meighen-Berger"
