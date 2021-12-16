# -*- coding: utf-8 -*-
# Name: config.py
# Authors: Stephan Meighen-Berger
# Config file for the fennel package.

import logging
from typing import Dict, Any
import yaml
import numpy as np

_baseconfig: Dict[str, Any]

_baseconfig = {
    ###########################################################################
    # General inputs
    ###########################################################################
    "general": {
        # Random state seed
        "random state seed": 1337,
        # Enable logger and config dump
        "enable logging": False,
        # Output level
        'debug level': logging.ERROR,
        # Note the paths need to be set appropiately for your system
        # Location of logging file handler
        "log file handler": "../run/fennel.log",
        # Dump experiment config to this location
        "config location": "../run/fennel.txt",
        # JAX switch -- If true Jax functions will be used
        # Please note that JAX does not support Windows
        "jax": False,
    },
    ###########################################################################
    # Scenario
    ###########################################################################
    "scenario": {
        "medium": "water",  # The background medium
        "parametrization": "aachen"
    },
    ###########################################################################
    # Particle data
    ###########################################################################
    "mu-": {
        # Mass
        "mass": 0.1056583755,
        # Standard Track Length
        "standard track length": 10.,  # Value given in cm
    },
    "mu+": {
        # Mass
        "mass": 0.1056583755,
        # Standard Track Length
        "standard track length": 10.,  # Value given in cm
    },
    ###########################################################################
    # PDG ID Lib
    ###########################################################################
    "pdg id": {
        11: "e-",
        -11: "e+",
        12: "nue",
        -12: "anti_nue",
        13: "mu-",
        -13: "mu+",
        14: "numu",
        -14: "anti_numu",
        15: "tau-",
        -15: "tau+",
        16: "nutau",
        -16: "anti_nutau",
        22: "gamma",
        211: "pi+",
        -211: "pi-",
        130: "KL0",
        2212: "p+",
        -2212: "p-",
        2112: "n",
    },
    ###########################################################################
    # Mediums
    ###########################################################################
    "mediums": {
       "water": {
           "refractive index": 1.333,
           "density": 1.0,  # in g/cm^-3,
           "radiation length": 36.08,  # g cm^-2
       },
       "ice": {
           "refractive index": 1.309,
           "density": 0.9180,  # in g/cm^-3,
           "radiation length": 36.08,  # g cm^-2
       },
    },
    ###########################################################################
    # Simulation aspects
    ###########################################################################
    "simulation": {
        "track particles": [13, -13],
        "track interactions": [
            "ionization", "pair", "brems", "nuclear", "total"
            ],
        "em particles": [11, -11, 22],
        "hadron particles": [211, -211, 130, 2212, -2212, 2112]
    },
    ###########################################################################
    # Track
    ###########################################################################
    "track": {
        "additional track water": {
            "ionization": {
                "lambda": 0.2489,
                "kappa": 0.0030
            },
            "pair": {
                "lambda": -0.0626,
                "kappa": 0.0175
            },
            "brems": {
                "lambda": 0.0004,
                "kappa": 0.
            },
            "nuclear": {
                "lambda": 0.0005,
                "kappa": 0.
            },
            "total": {
                "lambda": 0.1880,
                "kappa": 0.0206
            },
        },
        "additional track ice": {
            "ionization": {
                "lambda": 0.2247,
                "kappa": 0.0030
            },
            "pair": {
                "lambda": -0.0612,
                "kappa": 0.0174
            },
            "brems": {
                "lambda": 0.0004,
                "kappa": 0.
            },
            "nuclear": {
                "lambda": 0.0005,
                "kappa": 0.
            },
            "total": {
                "lambda": 0.1842,
                "kappa": 0.0204
            },
        },
        "angular distribution": {
            "a pars": [0.34485, 0.03145],  # sr^-1
            "b pars": [-3.04160, -0.07193],
            "c pars": [0.69937, -0.01421],
        },
    },
    ###########################################################################
    # EM Cascade
    ###########################################################################
    # TODO: Move this to a seperate data file
    "em cascade": {
        "track parameters": {
            11: {
                "alpha": 532.07078881,  # cm GeV^-1
                "beta": 1.00000211,
                "alpha dev": 5.78170887,
                "beta dev": 0.5
            },
            -11: {
                "alpha": 532.11320598,  # cm GeV^-1
                "beta": 0.99999254,
                "alpha dev": 5.73419669,
                "beta dev": 0.5
            },
            22: {
                "alpha": 532.08540905,  # cm GeV^-1
                "beta": 0.99999877,
                "alpha dev": 5.78170887,
                "beta dev": 5.66586567
            },
        },
        "longitudinal parameters": {
            11: {
                "alpha": 2.01849,
                "beta": 1.45469,
                "b": 0.63207
            },
            -11: {
                "alpha": 2.00035,
                "beta": 1.45501,
                "b": 0.63008
            },
            22: {
                "alpha": 2.83923,
                "beta": 1.34031,
                "b": 0.64526
            },
        },
        "angular distribution": {
            11: {
                "a": 4.27033,  # sr^-1
                "b": -6.02527,
                "c": 0.29887,
                "d": -0.00103,  # sr^-1
            },
            -11: {
                "a": 4.27725,  # sr^-1
                "b": -6.02430,
                "c": 0.29856,
                "d": -0.00104,  # sr^-1
            },
            22: {
                "a": 4.25716,  # sr^-1
                "b": -6.02421,
                "c": 0.29926,
                "d": -0.00101,  # sr^-1
            },
        }
    },
    ###########################################################################
    # Hadron Cascade
    ###########################################################################
    # TODO: Move this to a seperate data file
    "hadron cascade": {
        "track parameters": {
            211: {
                "alpha": 333.55182722,  # cm GeV^-1
                "beta": 1.03662217,
                "alpha dev": 119.20455395,
                "beta dev": 0.80772057
            },
            -211: {
                "alpha": 335.84489578,  # cm GeV^-1
                "beta": 1.03584394,
                "alpha dev": 122.50188073,
                "beta dev": 0.80322520
            },
            130: {
                "alpha": 326.00450524,  # cm GeV^-1
                "beta": 1.03931457,
                "alpha dev": 121.41970572,
                "beta dev": 0.80779629
            },
            2212: {
                "alpha": 287.37183922,  # cm GeV^-1
                "beta": 1.05172118,
                "alpha dev": 88.04581378,
                "beta dev": 0.82445572
            },
            -2212: {
                "alpha": 303.33074914,  # cm GeV^-1
                "beta": 1.04322206,
                "alpha dev": 113.23088104,
                "beta dev": 0.77134060
            },
            2112: {
                "alpha": 278.43854660,  # cm GeV^-1
                "beta": 1.05582906,
                "alpha dev": 93.22787137,
                "beta dev": 0.81776503
            },
        },
        "em fraction": {
            211: {
                "Es": 0.15591,  # GeV
                "f0": 0.27273,
                "m": 0.15782,
                "sigma0": 0.40626,
                "gamma": 1.01771,
            },
            -211: {
                "Es": 0.13397,  # GeV
                "f0": 0.28735,
                "m": 0.15341,
                "sigma0": 0.43354,
                "gamma": 1.05561,
            },
            130: {
                "Es": 0.21684,  # GeV
                "f0": 0.26987,
                "m": 0.16365,
                "sigma0": 0.37875,
                "gamma": 0.97048,
            },
            2212: {
                "Es": 0.31944,  # GeV
                "f0": 0.10871,
                "m": 0.17921,
                "sigma0": 0.28550,
                "gamma": 0.93413,
            },
            -2212: {
                "Es": 0.31944,  # GeV
                "f0": 0.10871,
                "m": 0.17921,
                "sigma0": 0.47001,
                "gamma": 1.25674,
            },
            2112: {
                "Es": 0.25998,  # GeV
                "f0": 0.02775,
                "m": 0.18505,
                "sigma0": 0.27042,
                "gamma": 0.89796,
            },
        },
        "longitudinal parameters": {
            211: {
                "alpha": 1.81098,
                "beta": 0.90572,
                "b": 0.34347,
            },
            -211: {
                "alpha": 1.81430,
                "beta": 0.90165,
                "b": 0.34131,
            },
            130: {
                "alpha": 1.99751,
                "beta": 0.80628,
                "b": 0.35027,
            },
            2212: {
                "alpha": 1.62345,
                "beta": 0.90875,
                "b": 0.35871,
            },
            -2212: {
                "alpha": 1.88676,
                "beta": 0.78825,
                "b": 0.35063,
            },
            2112: {
                "alpha": 1.78137,
                "beta": 0.87687,
                "b": 0.35473,
            },
        },
        "angular distribution": {
            211: {
                "a pars": [0.25877, 1.05372],  # sr^-1
                "b pars": [-3.34355, 0.22303],
                "c pars": [0.70633, -0.34407],
                "d pars": [0.08572, -1.90632],  # sr^-1
            },
            -211: {
                "a pars": [0.25915, 1.05539],  # sr^-1
                "b pars": [-3.29885, 0.22989],
                "c pars": [0.71082, -0.34857],
                "d pars": [0.11207, -2.05247],  # sr^-1
            },
            130: {
                "a pars": [0.25015, 1.06819],  # sr^-1
                "b pars": [-3.33393, 0.22403],
                "c pars": [0.76039, -0.38042],
                "d pars": [0.14898, -2.19057],  # sr^-1
            },
            2212: {
                "a pars": [0.13966, 1.30159],  # sr^-1
                "b pars": [-2.82378, 0.29381],
                "c pars": [0.91092, -0.45380],
                "d pars": [0.13845, -2.02526],  # sr^-1
            },
            -2212: {
                "a pars": [0.08111, 1.52203],  # sr^-1
                "b pars": [-2.47748, 0.34737],
                "c pars": [1.16940, -0.56291],
                "d pars": [0.18410, -2.07564],  # sr^-1
            },
            2112: {
                "a pars": [0.11829, 1.37902],  # sr^-1
                "b pars": [-2.75135, 0.30581],
                "c pars": [0.99563, -0.49587],
                "d pars": [0.18446, -2.16233],  # sr^-1
            },
        },
    },
    ###########################################################################
    # Advanced
    ###########################################################################
    "advanced": {
        # Energy threshold for continuous losses
        "threshold E": 0.5,  # In GeV
        "energy grid": np.logspace(0., 9, 91),
        "fine structure": 0.0072973525693,
        "particle charge": 1.,  # Charge of the particle of interest
        "wavelengths": np.linspace(350., 500., 100),  # in nm
        "track length": 1.,  # in cm
        "angles": np.linspace(0., 180., 100),  # in degrees
        # The grid used for long profiling. Given in cm
        "z grid": np.linspace(0., 1e4, int(1e4)),
        # Location and name of the definitions file
        "generated definitions": "generated_definitions.pyx",
    },
}


class ConfigClass(dict):
    """ The configuration class. This is used
    by the package for all parameter settings. If something goes wrong
    its usually here.

    Parameters
    ----------
    config : dic
        The config dictionary

    Returns
    -------
    None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: Update this
    def from_yaml(self, yaml_file: str) -> None:
        """ Update config with yaml file

        Parameters
        ----------
        yaml_file : str
            path to yaml file

        Returns
        -------
        None
        """
        yaml_config = yaml.load(open(yaml_file), Loader=yaml.SafeLoader)
        self.update(yaml_config)

    # TODO: Update this
    def from_dict(self, user_dict: Dict[Any, Any]) -> None:
        """ Creates a config from dictionary

        Parameters
        ----------
        user_dict : dic
            The user dictionary

        Returns
        -------
        None
        """
        self.update(user_dict)


config = ConfigClass(_baseconfig)
