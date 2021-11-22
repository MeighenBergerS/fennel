# -*- coding: utf-8 -*-
# Name: tracks.py
# Authors: Stephan Meighen-Berger
# Constructs a track, defined by the emission of photons

import logging
import numpy as np
from .config import config
# Checking if jax should be used
if config["general"]["jax"]:
    import jax


_log = logging.getLogger(__name__)


class Track(object):
    """Constructs the track object.

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
        """Constructs the track.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Other distribution type for emission angles is not implemented
        """
        _log.debug('Constructing a track object')
        self._medium = config["scenario"]["medium"]
        self._n = config["mediums"][self._medium]["refractive index"]
        if config["advanced"]["Cherenkov distro"] == "symmetric":
            self.cherenkov_angle_distro = self._symmetric_angle_distro_fetcher
        else:
            ValueError("Distribution type not implemented!")

    def additional_track_ratio_fetcher(
            self, E, interaction: str) -> np.array:
        """ Calculates the ratio between the additional track length
        and the original for a single energy

        Parameters
        ----------
        E : float/np.array
            The energy of the particle in GeV
        interaction : str
            Name of the interaction

        Returns
        -------
        ratio : float
            The resulting ratio
        """
        params = config["track"]["additional track " + self._medium][
            interaction
        ]
        lambd = params["lambda"]
        kappa = params["kappa"]
        ratio = (
            lambd + kappa * np.log(E)
        )
        return ratio

    def _symmetric_angle_distro_fetcher(
            self,
            phi: np.array, n: float,
            E) -> np.array:
        # TODO: Add asymmetry function
        """ Calculates the symmetric angular distribution of the Cherenkov
        emission for a single energy. The error should lie below 10%

        Parameters
        ----------
        phi : np.array
            The angles of interest in degrees
        n : float
            The refractive index
        E : float/np.array
            The energy of interest

        Returns
        -------
        distro : np.array
            The distribution of emitted photons given the angle. The
            result is a 2d array with the first axis for the angles and
            the second for the energies.
        """
        a, b, c = self._energy_dependence_angle_pars(E)
        distro = np.array([
            (a * np.exp(b * np.abs(
                1. / n - np.cos(np.deg2rad(phi_val)))**c
            ))
            for phi_val in phi
        ])

        return distro

    def _energy_dependence_angle_pars(
            self, E):
        """ Parametrizes the energy dependence of the angular distribution
        parameters

        Parameters
        ----------
        E : float / np.array
            The energies of interest

        Returns
        -------
        a : np.array
            The first parameter values for the given energies
        b : np.array
            The second parameter values for the given energies
        c : np.array
            The third parameter values for the given energies
        """
        params = config["track"]["angular distribution"]
        a_pars = params["a pars"]
        b_pars = params["b pars"]
        c_pars = params["c pars"]
        a = a_pars[0] * (np.log(E)) * a_pars[1]
        b = b_pars[0] * (np.log(E)) * b_pars[1]
        c = c_pars[0] * (np.log(E)) * c_pars[1]
        return a, b, c
