# -*- coding: utf-8 -*-
# Name: particle.py
# Authors: Stephan Meighen-Berger
# Containts the particle class definitions

import logging
from .config import config


_log = logging.getLogger(__name__)


class Particle(object):
    """Constructs the particle object.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    """
    def __init__(self, pdg_id: int):
        """Constructs the particle.

        Parameters
        ----------
        pdg_id : int
            The pdg number of the particle

        Returns
        -------
        None

        Raises
        ------
        """
        _log.info('Constructing a particle')
        self._pdg_id = pdg_id
        # Naming conventions PDG Monte Carlo scheme
        if pdg_id > 0:
            self._name = config["pdg id"][pdg_id]
            _log.debug("The temporary name is " + self._name)
            if self._name == "gamma":
                self._name = self._name
            elif self._name == "n":
                self._name = self._name
            elif self._name == "KL0":
                self._name = self._name
            elif self._name[:2] == "nu":
                self._name = self._name
            elif pdg_id > 100:
                self._name = self._name + "+"
            else:
                self._name = self._name + "-"
        else:
            _log.debug("The temporary name is " + self._name)
            self._name = config["pdg id"][pdg_id]
            if self._name[:2] == "nu":
                self._name = "anti_" + self._name
            elif pdg_id < -100:
                self._name = self._name + "-"
            else:
                self._name = self._name + "+"
        _log.debug("The final name is " + self._name)
        self._energies = config["advanced"]["energy grid"]
        if self._name[:2] ==  "mu":
            self._mass = config[self._name]["mass"]
            self._std_track = config[self._name]["standard track length"]
