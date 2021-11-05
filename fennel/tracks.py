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
        particle : Particle
            The particle for which the tracks should be generated

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

    def _e2beta(E: np.array, mass: float) -> np.array:
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
            a: float, b: float, c: float) -> np.array:
        """ Calculates the symmetric angular distribution of the Cherenkov
        emission. The error should lie below 10%

        Parameters
        ----------
        phi : np.array
            The angles of interest in degrees
        n : float
            The refractive index
        a : float
            Fitted parameter
        b : float
            Fitted parameter
        c : float
            Fitted parameter

        Returns
        -------
        distro : np.array
            The distribution of emitted photons given the angle
        """
        distro = a * np.exp(b * np.abs(
            1. / n - np.cos(np.deg2rad(phi)))**c
        )
        return distro
