# -*- coding: utf-8 -*-
# Name: photons.py
# Authors: Stephan Meighen-Berger
# Calculates the number of photons depending on the track length

import logging
import numpy as np
from .config import config
from .tracks import Track
from .em_cascades import EM_Cascade
from .hadron_cascades import Hadron_Cascade

_log = logging.getLogger(__name__)


class Photon(object):
    """Constructs the Photon object.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    """
    def __init__(
            self,
            particle, track: Track,
            em_cascade: EM_Cascade, hadron_cascade: Hadron_Cascade):
        """Constructs the photon.

        Parameters
        ----------
        particles : dic
            Library of the particles of interst, each being a Particle object
        track : Track
            The particle for which the tracks should be generated
        em_cascade : Particle
            The particle for which the tracks should be generated
        hadron_cascades : Hadron_Cascade
            The particle for which the tracks should be generated


        Returns
        -------
        None

        Raises
        ------
        ValueError
            Other distribution type for emission angles is not implemented
        """
        _log.debug('Constructing a photon object')
        self._medium = config["scenario"]["medium"]
        self._n = config["mediums"][self._medium]["refractive index"]
        self._alpha = config["advanced"]["fine structure"]
        self._charge = config["advanced"]["particle charge"]
        self._wavelengths = config["advanced"]["wavelengths"]
        self._tracklengths = config["advanced"]["track lengths"]
        self._angle_grid = config["advanced"]["angles"]
        self._zgrid = config["advanced"]["z grid"]
        self.__particles = particle
        self.__track = track
        self.__em_cascade = em_cascade
        self.__hadron_cascade = hadron_cascade
        self._rstate = config["runtime"]["random state"]
        _log.debug('Finished a photon object. Launch using sim method')

    def _track_fetcher(
            self, energy, deltaL, interaction='total',
            function=False):
        """ Fetcher function for a specific particle and energy. This is for
        tracks and currently only for muons and symmetric distros.
        If energy and deltaL are arrays with length larger than 1, their
        shapes must be the same.

        Parameters
        ----------
        energy : float/np.array
            The energy of the particle in GeV
        deltaL : float/np.array
            The step size for the current track length in cm
        interaction : str
            Optional: The interaction(s) which should produce the light
        function : bool
            returns the functional form instead of the evaluation

        Returns
        -------
        differential_counts : np.array
            dN/dlambda The differential photon counts per track length (in cm).
            The shape of the array is (len(wavelengths), len(deltaL)).
        angles : np.array
            The angular distribution in degrees
        """
        if function:
            def counts(energy, wavelengths, interaction):
                """ Fetcher function for a specific particle and energy. This is for
                tracks and their photon counts

                Parameters
                ----------
                energy : float/np.array
                    The energy of the particle
                wavelengths : np.array
                    The wavelengths of interest
                mean : bool
                    Optional: Switch to use either the mean value or a sample

                Returns
                -------
                counts : float/np.array
                    The photon counts
                """
                tmp_track_frac = (
                    self.__track.additional_track_ratio_fetcher(
                        energy, interaction
                    )
                )
                new_track = deltaL * (1. + tmp_track_frac)
                new_track = np.array([new_track]).flatten()
                return self._cherenkov_counts(wavelengths, new_track)
            angles = self.__track._symmetric_angle_distro_fetcher
        else:
            tmp_track_frac = (
                self.__track.additional_track_ratio_fetcher(
                    energy, interaction
                )
            )
            new_track = deltaL * (1. + tmp_track_frac)
            new_track = np.array([new_track]).flatten()
            counts = self._cherenkov_counts(self._wavelengths, new_track)
            # The angular distribution
            angles = self.__track._symmetric_angle_distro_fetcher(
                self._angle_grid,
                self._n,
                energy)
        return counts, angles

    def _em_cascade_fetcher(
            self, energy: float, particle: int,
            mean=True, function=False):
        """ Fetcher function for a specific particle and energy. This is for
        em cascades and currently only symmetric distros

        Parameters
        ----------
        energy : float/np.array
            The energy of the particle
        particle : int
            The particle of interest with its pdg id
        mean : bool
            Optional: Switch to use either the mean value or a sample
        function : bool
            Optional: Switches between the functional and explicit forms

        Returns
        -------
        differential_counts : np.array
            dN/dlambda The differential photon counts per track length (in cm).
            The shape of the array is (len(wavelengths), len(deltaL)).
        long_profile : np.array
            The distribution along the shower axis for cm
        angles : np.array
            The angular distribution in degrees
        """
        if function:
            def counts(energy, wavelengths, particle, mean=True):
                """ Fetcher function for a specific particle and energy. This is for
                em cascades and their photon counts

                Parameters
                ----------
                energy : float/np.array
                    The energy of the particle
                wavelengths : np.array
                    The wavelengths of interest
                particle : int
                    The particle of interest with its pdg id
                mean : bool
                    Optional: Switch to use either the mean value or a sample

                Returns
                -------
                counts : float/np.array
                    The photon counts
                """
                tmp_track, tmp_track_sd = (
                    self.__em_cascade.track_lengths_fetcher(
                        energy, self.__particles[particle]
                    )
                )
                if mean:
                    tmp_track = np.array([tmp_track]).flatten()
                    return self._cherenkov_counts(wavelengths, tmp_track)
                else:
                    tmp_track_sample = self._rstate.normal(
                        tmp_track, tmp_track_sd
                    )
                    tmp_track_sample = np.array([tmp_track_sample]).flatten()
                    return self._cherenkov_counts(
                        wavelengths, tmp_track_sample
                    )
            long_profile = (
                self.__em_cascade._log_profile_func_fetcher
            )
            angles = self.__em_cascade.cherenkov_angle_distro
        else:
            # The track length
            tmp_track, tmp_track_sd = (
                self.__em_cascade.track_lengths_fetcher(
                    energy, self.__particles[particle]
                )
            )
            # Light yields
            if mean:
                tmp_track = np.array([tmp_track]).flatten()
                counts = self._cherenkov_counts(
                    self._wavelengths, tmp_track)
            else:
                tmp_track_sample = self._rstate.normal(tmp_track, tmp_track_sd)
                tmp_track_sample = np.array([tmp_track_sample]).flatten()
                counts = self._cherenkov_counts(
                    self._wavelengths, tmp_track_sample
                )
            # Long profile
            long_profile = (
                self.__em_cascade._log_profile_func_fetcher(
                    energy, self._zgrid, self.__particles[particle]
                )
            )
            # Angle distribution
            angles = self.__em_cascade.cherenkov_angle_distro(
                self._angle_grid, self._n, self.__particles[particle]
            )
        return counts, long_profile, angles

    def _hadron_cascade_fetcher(
            self, energy: float, particle: int,
            mean=True, function=False):
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
        differential_counts : np.array
            dN/dlambda The differential photon counts per track length (in cm).
            The shape of the array is (len(wavelengths), len(deltaL)).
        long_profile : np.array
            The distribution along the shower axis for cm
        em_fraction : np.array
            The amount of em in the shower
        angles : np.array
            The angular distribution in degrees
        """
        if function:
            def counts(energy, wavelengths, particle, mean=True):
                """ Fetcher function for a specific particle and energy.
                This is for hadron cascades and their photon counts

                Parameters
                ----------
                energy : float/np.array
                    The energy of the particle
                wavelengths : np.array
                    The wavelengths of interest
                particle : int
                    The particle of interest with its pdg id
                mean : bool
                    Optional: Switch to use either the mean value or a sample

                Returns
                -------
                counts : float/np.array
                    The photon counts
                """
                tmp_track, tmp_track_sd = (
                    self.__hadron_cascade.track_lengths_fetcher(
                        energy,
                        self.__particles[particle]
                    )
                )
                # Light yields
                if mean:
                    tmp_track = np.array([tmp_track]).flatten()
                    return self._cherenkov_counts(
                        wavelengths, tmp_track
                    )
                else:
                    tmp_track_sample = self._rstate.normal(
                        tmp_track, tmp_track_sd
                    )
                    tmp_track_sample = np.array([tmp_track_sample]).flatten()
                    return self._cherenkov_counts(
                        wavelengths, tmp_track_sample
                    )

            def em_fraction(energy, particle, mean=True):
                """ Fetcher function for a specific particle and energy. This is for
                hadron cascades and their em fraction

                Parameters
                ----------
                energy : float/np.array
                    The energy of the particle
                particle : int
                    The particle of interest with its pdg id
                mean : bool
                    Optional: Switch to use either the mean value or a sample

                Returns
                -------
                fraction : float/np.array
                    The fraction
                """
                tmp_fract, tmp_fract_sd = (
                    self.__hadron_cascade.em_fraction_fetcher(
                        energy, self.__particles[particle]
                    )
                )
                if mean:
                    return tmp_fract
                else:
                    return self._rstate.normal(tmp_fract, tmp_fract_sd)
            long_profile = (
                self.__hadron_cascade._log_profile_func_fetcher
            )
            # Angle distribution
            angles = self.__hadron_cascade._symmetric_angle_distro_fetcher
        else:
            # The track length
            tmp_track, tmp_track_sd = (
                self.__hadron_cascade.track_lengths_fetcher(
                    energy,
                    self.__particles[particle]
                )
            )
            # Light yields
            if mean:
                counts = self._cherenkov_counts(
                    self._wavelengths, [tmp_track]
                )
            else:
                tmp_track_sample = self._rstate.normal(tmp_track, tmp_track_sd)
                counts = self._cherenkov_counts(
                    self._wavelengths, [tmp_track_sample]
                )
            # EM Fraction
            tmp_fract, tmp_fract_sd = (
                self.__hadron_cascade.em_fraction_fetcher(
                    energy, self.__particles[particle]
                )
            )
            if mean:
                em_fraction = tmp_fract
            else:
                em_fraction = self._rstate.normal(tmp_fract, tmp_fract_sd, 1)
            # Long profile
            long_profile = (
                self.__hadron_cascade._log_profile_func_fetcher(
                    energy, self._zgrid, self.__particles[particle]
                )
            )
            # Angle distribution
            angles = self.__hadron_cascade._symmetric_angle_distro_fetcher(
                energy, self._angle_grid, self._n, self.__particles[particle]
            )
        return counts, long_profile, em_fraction, angles

    def _cherenkov_counts(
            self,
            wavelengths: np.array, track_lengths: np.array) -> np.array:
        """ Calculates the differential number of photons for the given
        wavelengths and track-lengths assuming a constant velocity with beta=1.

        Parameters
        ----------
        wavelengths : np.array
            The wavelengths of interest
        track_lengths : np.array
            The track lengths of interest in cm

        Returns
        -------
        counts : np.array
            A 2-d array filled witht the produced photons. The first axis
            defines the wavelengths, the second the track length.
        """
        prefac = (
            2. * np.pi * self._alpha * self._charge**2. /
            (1. - 1. / self._n**2.)
        )
        # 1e-7 due to the conversion from nm to cm
        diff_counts = np.array([[
            prefac / (lambd * 1e-9)**2. * length * 1e-2
            for length in track_lengths]
            for lambd in wavelengths
        ])
        return diff_counts * 1e-9 / np.pi
