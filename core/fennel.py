# -*- coding: utf-8 -*-
# Name: fennel.py
# Authors: Stephan Meighen-Berger
# Main interface to the fennel model. Calculates the light yields using the
# Aachen parametrization from
# https://www.institut3b.physik.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaaapwhjz
# which is Leif Raedel's Master thesis

# Imports
# Native modules
import logging
import sys
import numpy as np
import yaml
# -----------------------------------------
# Package modules
from .config import config
from .particle import Particle

# unless we put this class in __init__, __name__ will be contagion.contagion
_log = logging.getLogger("fennel")


class Fennel(object):
    """
    class: Fennel
    Interace to the fennel package. This class
    stores all methods required to run the simulation
    of the particle light yields
    Parameters
    ----------
    config : dic
        Configuration dictionary for the simulation
    
    Returns
    -------
    None
    """
    def __init__(self, userconfig=None):
        """
        function: __init__
        Initializes the class fennel.
        Here all run parameters are set.
        Parameters
        ----------
        config : dic
            Configuration dictionary for the simulation
        
        Returns
        -------
        None
        """
        # Inputs
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)

        # Create RandomState
        if config["general"]["random state seed"] is None:
            _log.warning("No random state seed given, constructing new state")
            rstate = np.random.RandomState()
        else:
            rstate = np.random.RandomState(
                config["general"]["random state seed"]
            )
        config["runtime"] = {"random state": rstate}

        # Logger
        # creating file handler with debug messages
        fh = logging.FileHandler(
            config["general"]["log file handler"], mode="w"
        )
        fh.setLevel(logging.DEBUG)
        # console logger with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(config["general"]["debug level"])

        # Logging formatter
        fmt = "%(levelname)s: %(message)s"
        fmt_with_name = "[%(name)s] " + fmt
        formatter_with_name = logging.Formatter(fmt=fmt_with_name)
        fh.setFormatter(formatter_with_name)
        # add class name to ch only when debugging
        if config["general"]["debug level"] == logging.DEBUG:
            ch.setFormatter(formatter_with_name)
        else:
            formatter = logging.Formatter(fmt=fmt)
            ch.setFormatter(formatter)

        _log.addHandler(fh)
        _log.addHandler(ch)
        _log.setLevel(logging.DEBUG)
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Welcome to Fennel!')
        _log.info('This package will help you model light yields')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Creating a particle...')
        # Life creation
        self._particle = Particle()
        _log.info('Creation finished')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('To run the simulation use the sim method')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')

    def sim(self):
        """ Calculates the light yields depending on input

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        # A new simulation
        _log.debug(
            "Dumping run settings into %s",
            config["general"]["config location"],
        )
        with open(config["general"]["config location"], "w") as f:
            yaml.dump(config, f)
        _log.debug("Finished dump")
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info("Have a great day and until next time!")
        _log.info('          /"*._         _')
        _log.info("      .-*'`    `*-.._.-'/")
        _log.info('    < * ))     ,       ( ')
        _log.info('     `*-._`._(__.--*"`.\ ')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        # Closing log
        logging.shutdown()

    @property
    def statistics(self):
        """ Getter functions for the simulation results
        from the simulation

        Parameters
        ----------
        None

        Returns
        -------
        statistics : dic
            Stores the results from the simulation
        """
        return self._statistics
