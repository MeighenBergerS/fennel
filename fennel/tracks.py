# -*- coding: utf-8 -*-
# Name: tracks.py
# Authors: Stephan Meighen-Berger
# Constructs a track, defined by the emission of photons

import logging
import numpy as np
from .config import config
from .particle import Particle


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
            self.cherenkov_angle_distro = self._symmetric_angle_distro
        else:
            ValueError("Distribution type not implemented!")

    def additional_track_ratio(
            self, particle: Particle, interaction: str) -> np.array:
        """ Calculates the ratio between the additional track length
        and the original

        Parameters
        ----------
        particle : Particle
            The particle for which this calculation should be performed
        interaction : str
            Name of the interaction

        Returns
        -------
        ratio : np.array
            The resulting ratio
        """
        E = particle._energies
        params = config["track"]["additional track " + self._medium][
            interaction
        ]
        lambd = params["lambda"]
        kappa = params["kappa"]
        ratio = (
            lambd + kappa * np.log(E)
        )
        return ratio

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

    def _additional_track_ratio_v1(
            self, particle: Particle,
            lambd: np.array, kappa: np.array, n: float,
            scaling=True) -> np.array:
        """ Calculates the ratio between the additional track length
        and the original. This uses the explicit frank-tamm scaling

        Parameters
        ----------
        particle : Particle
            The particle for which this calculation should be performed
        lambd : np.array
            The fitted offset
        kappa : np.array
            The fitted inclination
        n : float
            The refractive index
        scaling : bool
            Optional parameter defining if frank-tamm scaling should be applied

        Returns
        -------
        ratio : np.array
            The resulting ratio
        """
        velocities = self._e2beta(particle._energies, particle._mass)
        if scaling:
            scaling_fac = self._frank_tamm_scaling(velocities, n)
        else:
            scaling_fac = 1
        ratio = lambd * scaling_fac * (
            lambd + kappa * np.log(particle._energies)
        )
        return ratio

    def additional_tracks(
            self, particle: Particle,
            tau: np.array, lmu: np.array):
        """ Calculates the mean and deviation of the additional tracks

        Parameters
        ----------
        particle : Particle
            The particle of interest
        tau : np.array
            The fitted parameter
        lmu : np.array
            The muons' track lengths

        Returns
        -------
        add_tracks, add_tracks_sd : np.array
            The additional track lengths and standard deviations
        """
        add_tracks = tau * lmu / particle._std_track
        add_tracks_sd = tau * np.sqrt(
            lmu / particle._std_track
        )

        return add_tracks, add_tracks_sd

    def _frank_tamm_scaling(self, beta: np.array, n: float) -> np.array:
        """ Calculates the scaling factor to get an equivalent fully
        relativistic track

        Parameters
        ----------
        beta: np.array
            The velocities of interest
        n: np.array
            The mediums refractive index

        Returns
        -------
        scaling : np.array
            The resulting scaling values
        """
        scaling = (
            self._sin2_cherenkov_angle(beta, n) /
            self._sin2_cherenkov_angle(np.ones(len(beta)), n)
        )
        return scaling

    def _sin2_cherenkov_angle(self, beta: np.array, n: float) -> np.array:
        """ Calculates the squared sine of the Cherenkov angle

        Parameters
        ----------
        beta: np.array
            The velocities of interest
        n: np.array
            The mediums refractive index

        Returns
        -------
        sin_angle : np.array
            The resulting angle values
        """
        sin_angle = 1. - (1. / (beta * n))**2.
        return sin_angle

    def _e2beta(self, E: np.array, mass: float) -> np.array:
        """ Calculates the particle velocity in units of 1/c from energy

        Parameters
        ----------
        E : np.array
            The energies of interest in GeV
        mass : float
            The mass of the particle in GeV

        Returns
        -------
        beta : np.array
            The velocities of the particle for the given energies
        """
        beta = np.sqrt(1 - ((E / mass + 1)**(-1))**2.)
        return beta

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
        a, b, c = self._energy_dependence_angle_pars(particle._energies)
        distro = np.array([
            (a * np.exp(b * np.abs(
                1. / n - np.cos(np.deg2rad(phi_val)))**c
            ))
            for phi_val in phi
        ])

        return distro

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
