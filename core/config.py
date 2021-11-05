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
        # Output level
        'debug level': logging.ERROR,
        # Location of logging file handler
        "log file handler": "../run/fennel.log",
        # Dump experiment config to this location
        "config location": "../run/fennel.txt",
    },
    ###########################################################################
    # Scenario
    ###########################################################################
    "scenario": {
        "medium": "water",  # The background medium
    },
    ###########################################################################
    # Muon
    ###########################################################################
    "mu-": {
        # Mass
        "mass": 0.1056583755,
        # Standard Track Length
        "standard track length": 10.,  # Value given in cm
    },
    ###########################################################################
    # PDG ID Lib
    ###########################################################################
    "pdg id": {
        11: {"e"},
        12: {"nue"},
        13: {"mu"},
        14: {"numu"},
        15: {"tau"},
        16: {"nutau"},
        22: {"gamma"},
        211: {"pi"},
        130: {"KL0"},
        2212: {"p"},
        2112: {"n"},
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
    # EM Cascade
    ###########################################################################
    # TODO: Move this to a seperate data file
    "em cascade": {
        "track parameters": {
            "e-":{
                "alpha": 532.07078881,  # cm GeV^-1
                "beta": 1.00000211,
                "alpha dev": 5.78170887,
                "beta dev": 0.5
            },
            "e+":{
                "alpha": 532.11320598,  # cm GeV^-1
                "beta": 0.99999254,
                "alpha dev": 5.73419669,
                "beta dev": 0.5
            },
            "gamma":{
                "alpha": 532.08540905,  # cm GeV^-1
                "beta": 0.99999877,
                "alpha dev": 5.78170887,
                "beta dev": 5.66586567
            },
        },
        "longitudinal parameters": {
            "e-": {
                "alpha": 2.01849,
                "beta": 1.45469,
                "b": 0.63207
            },
            "e+": {
                "alpha": 2.00035,
                "beta": 1.45501,
                "b": 0.63008
            },
            "gamma": {
                "alpha": 2.83923,
                "beta": 1.34031,
                "b": 0.64526
            },
        },
        "angular distribution": {
            "e-": {
                "a": 4.27033,  # sr^-1
                "b": -6.02527,
                "c": 0.29887,
                "d": -0.00103,  # sr^-1
            },
            "e+": {
                "a": 4.27725,  # sr^-1
                "b": -6.02430,
                "c": 0.29856,
                "d": -0.00104,  # sr^-1
            },
            "gamma": {
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
            "pi+":{
                "alpha": 333.55182722,  # cm GeV^-1
                "beta": 1.03662217,
                "alpha dev": 119.20455395,
                "beta dev": 0.80772057
            },
            "pi-":{
                "alpha": 335.84489578,  # cm GeV^-1
                "beta": 1.03584394,
                "alpha dev": 122.50188073,
                "beta dev": 0.80322520
            },
            "KL0":{
                "alpha": 326.00450524,  # cm GeV^-1
                "beta": 1.03931457,
                "alpha dev": 121.41970572,
                "beta dev": 0.80779629
            },
            "p+":{
                "alpha": 287.37183922,  # cm GeV^-1
                "beta": 1.05172118,
                "alpha dev": 88.04581378,
                "beta dev": 0.82445572
            },
            "p-":{
                "alpha": 303.33074914,  # cm GeV^-1
                "beta": 1.04322206,
                "alpha dev": 113.23088104,
                "beta dev": 0.77134060
            },
            "n":{
                "alpha": 278.43854660,  # cm GeV^-1
                "beta": 1.05582906,
                "alpha dev": 93.22787137,
                "beta dev": 0.81776503
            },
        },
        "longitudinal parameters": {
            "e-": {
                "alpha": 2.01849,
                "beta": 1.45469,
                "b": 0.63207
            },
            "e+": {
                "alpha": 2.00035,
                "beta": 1.45501,
                "b": 0.63008
            },
            "gamma": {
                "alpha": 2.83923,
                "beta": 1.34031,
                "b": 0.64526
            },
        },
        "angular distribution": {
            "e-": {
                "a": 4.27033,  # sr^-1
                "b": -6.02527,
                "c": 0.29887,
                "d": -0.00103,  # sr^-1
            },
            "e+": {
                "a": 4.27725,  # sr^-1
                "b": -6.02430,
                "c": 0.29856,
                "d": -0.00104,  # sr^-1
            },
            "gamma": {
                "a": 4.25716,  # sr^-1
                "b": -6.02421,
                "c": 0.29926,
                "d": -0.00101,  # sr^-1
            },
        }
    },
    ###########################################################################
    # Advanced
    ###########################################################################
    "advanced": {
        # Energy threshold for continuous losses
        "threshold E": 0.5,  # In GeV
        "energy grid": np.logspace(0., 9, 91),
        "Cherenkov distro": "symmetric"
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
