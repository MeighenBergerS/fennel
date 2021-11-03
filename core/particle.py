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
    def __init__(self):
        """Constructs the particle.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        """
        _log.debug('Constructing particle')
