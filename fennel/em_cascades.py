# -*- coding: utf-8 -*-
# Name: em_cascades.py
# Authors: Stephan Meighen-Berger
# Constructs an electromagnetic cascade, defined by the emitted photons

import logging
import numpy as np
from scipy.special import gamma as gamma_func
from .config import config
from .particle import Particle


_log = logging.getLogger(__name__)


class EM_Cascade(object):
    """Constructs the em cascade object.

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
        """Constructs the em cascade.

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
        _log.debug('Constructing an em cascade object')
        self._medium = config["mediums"][config["scenario"]["medium"]]
        self._n = self._medium["refractive index"]
        self._radlength = self._medium["radiation length"]
        self._Lrad = self._radlength / self._medium["density"]
        if config["advanced"]["Cherenkov distro"] == "symmetric":
            self.cherenkov_angle_distro = self._symmetric_angle_distro
        else:
            ValueError("Distribution type not implemented!")

    def track_lengths_fetcher(
            self, E, particle: Particle):
        """ Parametrization for the energy dependence of the tracks. This is
        the fetcher function for a single energy

        Parameters
        ----------
        E : float/np.array
            The energy of interest
        particle : Particle
            The particle of interest

        Returns
        -------
        track_length : np.array
            The track lengths for different energies
        track_length_dev : np.array
            The track lengths deviations for different energies
        """
        params = config["em cascade"]["track parameters"][particle._name]
        alpha = params["alpha"]
        beta = params["beta"]
        alpha_dev = params["alpha dev"]
        beta_dev = params["beta dev"]
        track_length = alpha * E**beta
        track_length_dev = alpha_dev * E**beta_dev
        return track_length, track_length_dev

    def _log_profile_func_fetcher(
            self, E, z: np.array, particle: Particle,
            ) -> np.array:
        """ Parametrization of the longitudinal profile for a single energy.
        This still needs work

        Parameters
        ----------
        E : float/np.array
            The energy in GeV
        z : np.array
            The cascade depth in cm
        particle : Particle
            The particle of interest

        Returns
        -------
        res : np.array
            Is equal to l^(-1) * dl/dt. The result will be 2 dimensional, with
            cm defined along the first axis and energies along the second
        """
        t = z / self._Lrad
        b = self._b_energy_fetch(particle)
        a = self._a_energy_fetch(E, particle)
        # res = np.array([
        #     t_val * self._b_energy_fetch(particle) *
        #     gamma.pdf(t_val * self._b_energy_fetch(particle),
        #               self._a_energy_fetch(E, particle))
        #     for t_val in t
        # ])
        # gamma.pdf seems far slower than the explicit implementation
        a = np.array(a).flatten()
        res = np.array([
            t * b * (
                (t * b)**(a_val - 1.) * np.exp(-(t*b)) / gamma_func(a_val)
            ) for a_val in a
        ])
        return res

    def _a_energy_fetch(self, E: float, particle: Particle) -> np.array:
        """ Parametrizes the energy dependence of the a parameter for the
        longitudinal profiles. This is for a single energy.

        Parameters
        ----------
        E : float
            The energy in GeV
        particle : Particle
            The particle of interest

        Returns
        -------
        a : np.array
            The values for the energies of interest
        """
        params = config["em cascade"]["longitudinal parameters"][
            particle._name]
        alpha = params["alpha"]
        beta = params["beta"]
        a = alpha + beta * np.log10(E)
        return a

    def _b_energy_fetch(self, particle: Particle) -> np.array:
        """ Parametrizes the energy dependence of the b parameter for the
        longitudinal profiles. Currently assumed to be constant.
        This is for a single energy.

        Parameters
        ----------
        particle : Particle
            The particle of interest

        Returns
        -------
        b : np.array
            The values for the energies of interest
        """
        params = config["em cascade"]["longitudinal parameters"][
            particle._name]
        b = params["b"]
        return b

    def _symmetric_angle_distro(
            self,
            phi: np.array, n: float,
            particle: Particle) -> np.array:
        # TODO: Add asymmetry function
        # TODO: Add changes with shower depth
        """ Calculates the symmetric angular distribution of the Cherenkov
        emission. The error should lie below 10%

        Parameters
        ----------
        phi : np.array
            The angles of interest in degrees
        n : float
            The refractive index
        particle : Particle
            The particle of interest

        Returns
        -------
        distro : np.array
            The distribution of emitted photons given the angle
        """
        params = config["em cascade"]["angular distribution"][particle._name]
        a = params["a"]
        b = params["b"]
        c = params["c"]
        d = params["d"]
        distro = (a * np.exp(b * np.abs(
            1. / n - np.cos(np.deg2rad(phi)))**c
        ) + d)
        return distro
