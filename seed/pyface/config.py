# -*- coding: utf-8 -*-
# Name: config.py
# Authors: Stephan Meighen-Berger
# Config file for the pyfaceö package.

from typing import Dict, Any
import yaml

_baseconfig: Dict[str, Any]

_baseconfig = {
    ###########################################################################
    # General inputs
    ###########################################################################
    "general": {
        # If files should be dumped or not
        "dump": False,
        # Random state seed
        "config location": "../run/pyface.txt",
        # Location of the output files
        "build directory": "../build/",
        "working_directory": "../build/",
        "output_location": '../run/',
        # Bash commands to run Geant4
        # Note you need to have run cmake once befor for this to work
        "bash commands": [
            "cmake ..",
            "make",
            "./showers -m py_run.mac"
        ],
        # The mac file and its location
        "mac file": '../py_run.mac'
    },
    ###########################################################################
    # Scenario
    ###########################################################################
    "scenario": {
        # Energy of the injected particle
        "energy": "1 TeV",
        # Species of the injected particle
        "particle": "proton",
        # Production cut for the secondaries
        "production_cut": "0.1 cm",
        # Number of events to generate
        "events": "100",
        # After how many steps to print out an intermediate result
        "progress_printing": "10",
        # Number of threads used in the simulation
        # Note Geant4 has to be compiled with multithreading for this!
        "threads": "4",
        # Switch to recompile the code or not. Turning off improves the
        # simulation time, but should only be used by people who know what
        # they are doing
        "recompile": True
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
