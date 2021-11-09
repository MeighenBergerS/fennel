# -*- coding: utf-8 -*-
# Name: photons.py
# Authors: Stephan Meighen-Berger
# Calculates the number of photons depending on the track length

import logging
import numpy as np
from time import time
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

    def _sim(self):
        """ Generates the light yield tables for the different particles
        and objects

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        start = time()
        _log.debug("Generating storage dic")
        results = {}
        for particle_id in config["pdg id"]:
            results[particle_id] = {}
        _log.info("Generating the track tables")
        for particle in config["simulation"]["track particles"]:
            results[particle]["track"] = {}
            # parameters
            results[particle]["track"]["wavelength grid"] = (
                self._wavelengths
            )
            results[particle]["track"]["angle grid"] = (
                self._angle_grid
            )
            results[particle]["track"]['length grid'] = (
                self._tracklengths
            )
            # Looping over possible interactions
            for interaction in (
                        config["simulation"]["track interactions"]
                    ):
                results[particle]["track"][interaction] = {}
                # Parameters
                results[particle]["track"][interaction]['e grid'] = (
                    self.__particles[particle]._energies
                )
                # The additional track length
                tmp_track_frac = (
                    self.__track.additional_track_ratio(
                        self.__particles[particle], interaction
                    )
                )
                # Adding length to the injected track length
                total_lengths = np.array([
                    length * (1 + tmp_track_frac)
                    for length in self._tracklengths
                ])
                results[particle]["track"][interaction]['length'] = (
                    total_lengths
                )
                # Calculating light yields
                light_yields = np.array([
                    self._cherenkov_counts(self._wavelengths, track_lengths)
                    for track_lengths in total_lengths
                ])
                results[particle]["track"][interaction]['light yields'] = (
                    light_yields
                )
            # The angular distribution
            angles = self.__track.cherenkov_angle_distro(
                self._angle_grid,
                self._n,
                self.__particles[particle])
            results[particle]["track"]['emission angles'] = (
                angles
            )
        _log.info("Generating the em cascade tables")
        for particle in config["simulation"]["em particles"]:
            results[particle]["em cascade"] = {}
            # The em track parameters
            results[particle]["em cascade"]['e grid'] = (
                self.__particles[particle]._energies
            )
            results[particle]["em cascade"]["wavelength grid"] = (
                self._wavelengths
            )
            results[particle]["em cascade"]["angle grid"] = (
                self._angle_grid
            )
            results[particle]["em cascade"]["z grid"] = (
                self._zgrid
            )
            # The track length
            tmp_track, tmp_track_sd = (
                self.__em_cascade.track_lengths(self.__particles[particle])
            )
            results[particle]["em cascade"]['length'] = (
                tmp_track
            )
            results[particle]["em cascade"]['length sd'] = (
                tmp_track_sd
            )
            # Light yields
            light_yields = self._cherenkov_counts(self._wavelengths, tmp_track)
            results[particle]["em cascade"]['mean light yields'] = (
                light_yields
            )
            # Long profile
            long_profile = (
                self.__em_cascade._log_profile_func(
                    self._zgrid, self.__particles[particle]
                )
            )
            results[particle]["em cascade"]['long profile'] = (
                long_profile
            )
            # Angle distribution
            angles = self.__em_cascade.cherenkov_angle_distro(
                self._angle_grid, self._n, self.__particles[particle]
            )
            results[particle]["em cascade"]['emission angles'] = (
                angles
            )
        _log.info("Generating the hadron cascade tables")
        for particle in config["simulation"]["hadron particles"]:
            results[particle]["hadron cascade"] = {}
            # The em track parameters
            results[particle]["hadron cascade"]['e grid'] = (
                self.__particles[particle]._energies
            )
            results[particle]["hadron cascade"]["wavelength grid"] = (
                self._wavelengths
            )
            results[particle]["hadron cascade"]["angle grid"] = (
                self._angle_grid
            )
            results[particle]["hadron cascade"]["z grid"] = (
                self._zgrid
            )
            # The track length
            tmp_track, tmp_track_sd = (
                self.__hadron_cascade.track_lengths(self.__particles[particle])
            )
            results[particle]["hadron cascade"]['length'] = (
                tmp_track
            )
            results[particle]["hadron cascade"]['length sd'] = (
                tmp_track_sd
            )
            # EM Fraction
            tmp_fract, tmp_fract_sd = (
                self.__hadron_cascade.em_fraction(self.__particles[particle])
            )
            results[particle]["hadron cascade"]['em fraction'] = (
                tmp_fract
            )
            results[particle]["hadron cascade"]['em fraction'] = (
                tmp_fract_sd
            )
            # Light yields
            light_yields = self._cherenkov_counts(self._wavelengths, tmp_track)
            results[particle]["hadron cascade"]['mean light yields'] = (
                light_yields
            )
            # Long profile
            long_profile = (
                self.__hadron_cascade._log_profile_func(
                    self._zgrid, self.__particles[particle]
                )
            )
            results[particle]["hadron cascade"]['long profile'] = (
                long_profile
            )
            # Angle distribution
            angles = self.__hadron_cascade.cherenkov_angle_distro(
                self._angle_grid, self._n, self.__particles[particle]
            )
            results[particle]["hadron cascade"]['emission angles'] = (
                angles
            )
        end = time()
        _log.debug("The simulation took %.f seconds" % (end - start))
        return results

    def _track_fetcher(
            self, energy: float, deltaL: float, interaction='total'):
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

        Returns
        -------
        counts : float
            The photon counts
        angles : np.array
            The angular distribution
        """
        tmp_track_frac = (
            self.__track.additional_track_ratio_fetcher(
                energy, interaction
            )
        )
        new_track = deltaL * (1. + tmp_track_frac)
        counts = self._cherenkov_counts(self._wavelengths, [new_track])
        # The angular distribution
        angles = self.__track._symmetric_angle_distro_fetcher(
            self._angle_grid,
            self._n,
            energy)
        return counts.flatten(), angles

    def _em_cascade_fetcher(
            self, energy: float, particle: int,
            mean=True):
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

        Returns
        -------
        counts : float
            The photon counts
        long_profile : np.array
            The distribution along the shower axis
        angles : np.array
            The angular distribution
        """
        # The track length
        tmp_track, tmp_track_sd = (
            self.__em_cascade.track_lengths_fetcher(
                energy, self.__particles[particle]
            )
        )
        # Light yields
        if mean:
            counts = self._cherenkov_counts(self._wavelengths, [tmp_track])
        else:
            tmp_track_sample = self._rstate.normal(tmp_track, tmp_track_sd, 1)
            counts = self._cherenkov_counts(
                self._wavelengths, [tmp_track_sample]
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
        return counts.flatten(), long_profile, angles

    def _hadron_cascade_fetcher(
            self, energy: float, particle: int,
            mean=True):
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
        # The track length
        tmp_track, tmp_track_sd = (
            self.__hadron_cascade.track_lengths_fetcher(
                energy,
                self.__particles[particle]
            )
        )
        # Light yields
        if mean:
            counts = self._cherenkov_counts(self._wavelengths, [tmp_track])
        else:
            tmp_track_sample = self._rstate.normal(tmp_track, tmp_track_sd, 1)
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
        return counts.flatten(), long_profile, em_fraction, angles

    def _cherenkov_counts(
            self,
            wavelengths: np.array, track_lengths: np.array) -> np.array:
        """ Calculates the number of photons for given wavelengths and
        track-lengths assuming a constant velocity with beta=1.

        Parameters
        ----------
        wavelengths : np.array
            The wavelengths of interest
        track_lengths : np.array
            The track lengths of interest

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
        counts = np.array([[
            prefac / (lambd * 1e-7)**2. * length for length in track_lengths]
            for lambd in wavelengths
        ])
        return counts
