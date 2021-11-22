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
from .tracks import Track
from .em_cascades import EM_Cascade
from .hadron_cascades import Hadron_Cascade
from .photons import Photon

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
        # Logging formatter
        fmt = "%(levelname)s: %(message)s"
        fmt_with_name = "[%(name)s] " + fmt
        formatter_with_name = logging.Formatter(fmt=fmt_with_name)
        # creating file handler with debug messages
        if config["general"]["enable logging"]:
            fh = logging.FileHandler(
                config["general"]["log file handler"], mode="w"
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter_with_name)
            _log.addHandler(fh)
        # console logger with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(config["general"]["debug level"])
        # add class name to ch only when debugging
        if config["general"]["debug level"] == logging.DEBUG:
            ch.setFormatter(formatter_with_name)
        else:
            formatter = logging.Formatter(fmt=fmt)
            ch.setFormatter(formatter)
        _log.addHandler(ch)
        _log.setLevel(logging.DEBUG)
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Welcome to Fennel!')
        _log.info('This package will help you model light yields')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Creating particles...')
        self._particles = {}
        for particle_id in config["pdg id"].keys():
            # Particle creation
            self._particles[particle_id] = Particle(particle_id)
        _log.info('Creation finished')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Creating a track...')
        # Track creation
        self._track = Track()
        _log.info('Creation finished')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Creating an em cascade...')
        # EM cascade creation
        self._em_cascade = EM_Cascade()
        _log.info('Creation finished')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Creating a hadron cascade...')
        # Hadron cascade creation
        self._hadron_cascade = Hadron_Cascade()
        _log.info('Creation finished')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Creating a photon...')
        # Hadron cascade creation
        self._photon = Photon(
            self._particles, self._track,
            self._em_cascade, self._hadron_cascade
        )
        _log.info('Creation finished')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('To run the simulation use the sim method')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')

    def close(self):
        """ Wraps up the program

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
        if config["general"]["enable logging"]:
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
        _log.info("     `*-._`._(__.--*.---. ")
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        # Closing log
        logging.shutdown()

    def track_yields(
            self,
            energy: float, deltaL: float, interaction='total',
            function=False):
        """ Fetcher function for a specific particle and energy. This is for
        tracks and currently only for muons and symmetric distros

        Parameters
        ----------
        energy : float
            The energy of the particle
        deltaL : float
            The step size for the current track length in cm
        interaction : str
            Optional: The interaction(s) which should produce the light
        function : bool
            returns the functional form instead of the evaluation

        Returns
        -------
        counts : float/function
            The photon counts
        angles : np.array/function
            The angular distribution
        """
        return self._photon._track_fetcher(
            energy, deltaL, interaction, function
        )

    def em_yields(
            self, energy: float, particle: int,
            mean=True, function=False):
        """ Fetcher function for a specific particle and energy. This is for
        em cascades and currently only symmetric distros

        Parameters
        ----------
        energy : float
            The energy of the particle
        particle : int
            The particle of interest with its pdg id
        mean : bool
            Optional: Switch to use either the mean value or a sample
        function : bool
            Optional: Switches between the functional and explicit forms

        Returns
        -------
        counts : float
            The photon counts
        long_profile : np.array
            The distribution along the shower axis
        angles : np.array
            The angular distribution
        """
        return self._photon._em_cascade_fetcher(
            energy, particle, mean, function
        )

    def hadron_yields(
            self, energy, particle,
            mean, function=False):
        """ Fetcher function for a specific particle and energy. This is for
        hadron cascades and currently only symmetric distros

        Parameters
        ----------
        energy : float
            The energy of the particle
        particle : int
            The particle of interest with its pdg id
        mean : bool
            Optional: Switch to use either the mean value or a sample
        function : bool
            Optional: Switches between the functional and explicit forms

        Returns
        -------
        counts : float
            The photon counts
        long_profile : np.array
            The distribution along the shower axis
        em_fraction : np.array
            The amount of em in the shower
        angles : np.array
            The angular distribution
        """
        return self._photon._hadron_cascade_fetcher(
            energy, particle, mean, function
        )
