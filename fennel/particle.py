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
        if not config["general"]["enable logging"]:
            _log.disabled = True
        _log.info('Constructing a particle')
        self._pdg_id = pdg_id
        # Naming conventions PDG Monte Carlo scheme
        self._name = config["pdg id"][pdg_id]
        _log.debug("The final name is " + self._name)
        self._energies = config["advanced"]["energy grid"]
        # TODO: Add masses for all particles
        # Masses of the muons
        if self._name[:2] == "mu":
            self._mass = config[self._name]["mass"]
            self._std_track = config[self._name]["standard track length"]
