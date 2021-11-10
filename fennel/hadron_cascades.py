# -*- coding: utf-8 -*-
# Name: hadron_cascades.py
# Authors: Stephan Meighen-Berger
# Constructs a hadron cascade, defined by the emitted photons

import logging
import numpy as np
from scipy.stats import gamma
from scipy.special import gamma as gamma_func
from .config import config
from .particle import Particle


_log = logging.getLogger(__name__)


class Hadron_Cascade(object):
    """Constructs the hadron cascade object.

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
        """Constructs the hadron cascade.

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
        _log.debug('Constructing a hadron cascade object')
        self._medium = config["mediums"][config["scenario"]["medium"]]
        self._n = self._medium["refractive index"]
        self._radlength = self._medium["radiation length"]
        self._Lrad = self._radlength / self._medium["density"]
        if config["advanced"]["Cherenkov distro"] == "symmetric":
            self.cherenkov_angle_distro = self._symmetric_angle_distro
        else:
            ValueError("Distribution type not implemented!")

    def track_lengths(
            self, particle: Particle):
        """ Parametrization for the energy dependence of the tracks

        Parameters
        ----------
        particle : Particle
            The particle of interest

        Returns
        -------
        track_length : np.array
            The track lengths for different energies
        track_length_dev : np.array
            The track lengths deviations for different energies
        """
        E = particle._energies
        params = config["hadron cascade"]["track parameters"][particle._name]
        alpha = params["alpha"]
        beta = params["beta"]
        alpha_dev = params["alpha dev"]
        beta_dev = params["beta dev"]
        track_length = alpha * E**beta
        track_length_dev = alpha_dev * E**beta_dev
        return track_length, track_length_dev

    def track_lengths_fetcher(
            self, E, particle: Particle):
        """ Parametrization for the energy dependence of the tracks

        Parameters
        ----------
        E : float/np.array
            The energy of interest in GeV
        particle : Particle
            The particle of interest

        Returns
        -------
        track_length : np.array
            The track lengths for different energies
        track_length_dev : np.array
            The track lengths deviations for different energies
        """
        params = config["hadron cascade"]["track parameters"][particle._name]
        alpha = params["alpha"]
        beta = params["beta"]
        alpha_dev = params["alpha dev"]
        beta_dev = params["beta dev"]
        track_length = alpha * E**beta
        track_length_dev = alpha_dev * E**beta_dev
        return track_length, track_length_dev

    def em_fraction(self, particle: Particle):
        """ Parametrization of the EM contribution in a hadronic shower

        Parameters
        ----------
        particle : Particle
            The particle of interest

        Returns
        -------
        em_fraction : np.array
            The fraction for the given energies
        em_fraction_sd : np.array
            The standard deviation
        """
        E = particle._energies  # This is the initial energy of the cascade
        params = config["hadron cascade"]["em fraction"][particle._name]
        Es = params["Es"]
        f0 = params["f0"]
        m = params["m"]
        sigma0 = params["sigma0"]
        gamma = params["gamma"]
        em_fraction = 1. - (1. - f0)*(E / Es)**(-m)
        em_fraction_sd = sigma0 * np.log(E)**(-gamma)
        return em_fraction, em_fraction_sd

    def em_fraction_fetcher(self, E, particle: Particle):
        """ Parametrization of the EM contribution in a hadronic shower

        Parameters
        ----------
        E : float/np.array
            The energy of interest in GeV
        particle : Particle
            The particle of interest

        Returns
        -------
        em_fraction : np.array
            The fraction for the given energies
        em_fraction_sd : np.array
            The standard deviation
        """
        params = config["hadron cascade"]["em fraction"][particle._name]
        Es = params["Es"]
        f0 = params["f0"]
        m = params["m"]
        sigma0 = params["sigma0"]
        gamma = params["gamma"]
        em_fraction = 1. - (1. - f0)*(E / Es)**(-m)
        em_fraction_sd = sigma0 * np.log(E)**(-gamma)
        return em_fraction, em_fraction_sd

    def _log_profile_func(
            self, z: np.array, particle: Particle,
            ) -> np.array:
        """ Parametrization of the longitudinal profile. This still needs work

        Parameters
        ----------
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
        res = np.array([
            t_val * self._b_energy(particle) *
            gamma.pdf(t_val * self._b_energy(particle),
                      self._a_energy(particle))
            for t_val in t
        ])
        return res

    def _a_energy(self, particle: Particle) -> np.array:
        """ Parametrizes the energy dependence of the a parameter for the
        longitudinal profiles

        Parameters
        ----------
        particle : Particle
            The particle of interest

        Returns
        -------
        a : np.array
            The values for the energies of interest
        """
        E = particle._energies
        params = config["hadron cascade"]["longitudinal parameters"][
            particle._name]
        alpha = params["alpha"]
        beta = params["beta"]
        a = alpha + beta * np.log10(E)
        return a

    def _b_energy(self, particle: Particle) -> np.array:
        """ Parametrizes the energy dependence of the b parameter for the
        longitudinal profiles. Currently assumed to be constant

        Parameters
        ----------
        particle : Particle
            The particle of interest

        Returns
        -------
        b : np.array
            The values for the energies of interest
        """
        E = particle._energies
        params = config["hadron cascade"]["longitudinal parameters"][
            particle._name]
        b = params["b"]
        return b * np.ones(len(E))

    def _log_profile_func_fetcher(
            self, E, z: np.array, particle: Particle,
            ) -> np.array:
        """ Parametrization of the longitudinal profile. This still needs work

        Parameters
        ----------
        E : float/np.array
            The energy of interest in GeV
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
        b = self._b_energy_fetcher(particle)
        a = self._a_energy_fetcher(E, particle)
        a = np.array([a]).flatten()
        # gamma.pdf seems far slower than the explicit implementation
        res = np.array([
            t * b * (
                (t * b)**(a_val - 1.) * np.exp(-(t*b)) / gamma_func(a_val)
            ) for a_val in a
        ])
        return res

    def _a_energy_fetcher(self, E: float, particle: Particle) -> np.array:
        """ Parametrizes the energy dependence of the a parameter for the
        longitudinal profiles

        Parameters
        ----------
        E : float
            The energy of interest in GeV
        particle : Particle
            The particle of interest

        Returns
        -------
        a : np.array
            The values for the energies of interest
        """
        params = config["hadron cascade"]["longitudinal parameters"][
            particle._name]
        alpha = params["alpha"]
        beta = params["beta"]
        a = alpha + beta * np.log10(E)
        return a

    def _b_energy_fetcher(self, particle: Particle) -> np.array:
        """ Parametrizes the energy dependence of the b parameter for the
        longitudinal profiles. Currently assumed to be constant

        Parameters
        ----------
        E : float
            The energy of interest in GeV
        particle : Particle
            The particle of interest

        Returns
        -------
        b : np.array
            The values for the energies of interest
        """
        params = config["hadron cascade"]["longitudinal parameters"][
            particle._name]
        b = params["b"]
        return b

    def _symmetric_angle_distro(
            self,
            phi: np.array, n: float,
            particle: Particle) -> np.array:
        # TODO: Add asymmetry function
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
            The distribution of emitted photons given the angle. The
            result is a 2d array with the first axis for the angles and
            the second for the energies.
        """
        E = particle._energies
        a, b, c, d = self._energy_dependence_angle_pars(E, particle)
        distro = np.array([
            (a * np.exp(b * np.abs(
                1. / n - np.cos(np.deg2rad(phi_val)))**c
            ) + d)
            for phi_val in phi
        ])

        return np.nan_to_num(distro)

    def _symmetric_angle_distro_fetcher(
            self, E,
            phi: np.array, n: float,
            particle: Particle) -> np.array:
        # TODO: Add asymmetry function
        """ Calculates the symmetric angular distribution of the Cherenkov
        emission. The error should lie below 10%

        Parameters
        ----------
        E : float/np.array
            The energy of interest in GeV
        phi : np.array
            The angles of interest in degrees
        n : float
            The refractive index
        particle : Particle
            The particle of interest

        Returns
        -------
        distro : np.array
            The distribution of emitted photons given the angle. The
            result is a 2d array with the first axis for the angles and
            the second for the energies.
        """
        a, b, c, d = self._energy_dependence_angle_pars(E, particle)
        distro = np.array([
            (a * np.exp(b * np.abs(
                1. / n - np.cos(np.deg2rad(phi_val)))**c
            ) + d)
            for phi_val in phi
        ])

        return np.nan_to_num(distro)

    def _energy_dependence_angle_pars(
            self, E, particle: Particle):
        """ Parametrizes the energy dependence of the angular distribution
        parameters

        Parameters
        ----------
        E : float/np.array
            The energy(ies) of interest
        particle : Particle
            The particle of interest

        Returns
        -------
        a : np.array
            The first parameter values for the given energies
        b : np.array
            The second parameter values for the given energies
        c : np.array
            The third parameter values for the given energies
        d : np.array
            The fourth parameter values for the given energies
        """
        params = config[
            "hadron cascade"
        ]["angular distribution"][particle._name]
        a_pars = params["a pars"]
        b_pars = params["b pars"]
        c_pars = params["c pars"]
        d_pars = params["d pars"]
        a = a_pars[0] * (np.log(E))**a_pars[1]
        b = b_pars[0] * (np.log(E))**b_pars[1]
        c = c_pars[0] * (np.log(E))**c_pars[1]
        d = d_pars[0] * (np.log(E))**d_pars[1]
        return (
            np.array([a]).flatten(), np.array([b]).flatten(),
            np.array([c]).flatten(), np.array([d]).flatten()
        )
