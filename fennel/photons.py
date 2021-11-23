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
if config["general"]["jax"]:
    import jax.numpy as jnp
    from jax import jit
    from jax.random import normal as jnormal


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
        self._angle_grid = config["advanced"]["angles"]
        self._zgrid = config["advanced"]["z grid"]
        self.__particles = particle
        self.__track = track
        self.__em_cascade = em_cascade
        self.__hadron_cascade = hadron_cascade
        self._rstate = config["runtime"]["random state"]
        self._deltaL = config["advanced"]["track length"]
        self._track_interactions = config["simulation"]["track interactions"]
        # Building the functions
        _log.info("Building the necessary functions")
        self._track_builder()
        self._em_cascade_builder()
        # Tracks

        _log.debug('Finished a photon object.')

    def _track_fetcher(
            self, energy,
            wavelengths=config["advanced"]["wavelengths"],
            angle_grid=config["advanced"]["angles"],
            n=config["mediums"][
                config["scenario"]["medium"]]["refractive index"],
            interaction='total',
            function=False):
        """ Fetcher function for a specific energy and wavelength. This is for
        tracks and currently only for muons. Note in JAX mode the functions
        only take scalars!

        Parameters
        ----------
        energy : float
            The energy(ies) of the particle in GeV
        wavelengths : np.array
            Optional: The desired wavelengths
        angle_grid : np.array
            Optional: The desired angles
        n : float
            The refractive index of the medium.
        interaction : str
            Optional: The interaction which should produce the light
        function : bool
            Optional: returns the functional form instead of the evaluation

        Returns
        -------
        differential_counts : np.array/function
            dN/dlambda The differential photon counts per track length (in cm).
            The shape of the array is len(wavelengths).
        angles : np.array/function
            The angular distribution in degrees
        """
        if function:
            _log.debug("Fetching track functions for " + interaction)
            return (
                self._track_functions_dic[interaction]["dcounts"],
                self._track_functions_dic[interaction]["angles"]
            )
        else:
            _log.debug("Fetching track values for " + interaction)
            if config["general"]["jax"]:
                return(
                    np.array([
                        self._track_functions_dic[interaction]["dcounts"](
                            energy, wavelength
                        )
                        for wavelength in wavelengths
                    ]),
                    np.array([
                        self._track_functions_dic[interaction]["angles"](
                            angle,
                            n,
                            energy
                        )
                        for angle in angle_grid
                    ]),
                )
            else:
                return(
                    self._track_functions_dic[interaction]["dcounts"](
                        energy, wavelengths
                    ),
                    self._track_functions_dic[interaction]["angles"](
                        angle_grid,
                        n,
                        energy
                    )
                )

    def _em_cascade_fetcher(
            self, energy,
            particle: int,
            wavelengths=config["advanced"]["wavelengths"],
            angle_grid=config["advanced"]["angles"],
            n=config["mediums"][
                config["scenario"]["medium"]]["refractive index"],
            z_grid=config["advanced"]["z grid"],
            function=False):
        """ Fetcher function for a specific particle and energy. This is for
        em cascades.

       Parameters
        ----------
        energy : float
            The energy(ies) of the particle in GeV
        particle : int
            The pdg id of the particle of interest
        wavelengths : np.array
            Optional: The desired wavelengths
        angle_grid : np.array
            Optional: The desired angles in degress
        n : float
            Optional: The refractive index of the medium.
        z_grid : np.array
            Optional: The grid in cm for the long. distributions
        function : bool
            Optional: returns the functional form instead of the evaluation

        Returns
        -------
        differential_counts : function/float/np.array
            dN/dlambda The differential photon counts per track length (in cm).
            The shape of the array is (len(wavelengths), len(deltaL)).
        differential_counts_sample : float/np.array
            A sample of the differential counts distribution. Same shape as
            the differential counts
        long_profile : function/float/np.array
            The distribution along the shower axis for cm
        angles : function/float/np.array
            The angular distribution in degrees
        """
        print(particle)
        if function:
            _log.debug("Fetching em functions for pdg_id " + str(particle))
            return (
                self._em_cascade_function_dic[particle]["dcounts"],
                self._em_cascade_function_dic[particle]["dcounts sample"],
                self._em_cascade_function_dic[particle]["long distro"],
                self._em_cascade_function_dic[particle]["angle distro"]
            )
        # Fetching explicit values
        else:
            _log.debug("Fetching track values for " + str(particle))
            if config["general"]["jax"]:
                return(
                    np.array([
                        self._em_cascade_function_dic[particle]["dcounts"](
                            energy, wavelength
                        )
                        for wavelength in wavelengths
                    ]),
                    np.array([
                        self._em_cascade_function_dic[particle][
                            "dcounts sample"
                            ](energy, wavelength)
                        for wavelength in wavelengths
                    ]),
                    np.array([
                        self._em_cascade_function_dic[particle][
                            "long distro"
                            ](energy, z)
                        for z in z_grid
                    ]),
                    np.array([
                        self._em_cascade_function_dic[particle][
                            "angle distro"](
                                angle,
                                n
                            )
                        for angle in angle_grid
                    ]),
                )
            else:
                return(
                        self._em_cascade_function_dic[particle]["dcounts"](
                            energy, wavelengths
                        ),
                        self._em_cascade_function_dic[particle][
                            "dcounts sample"
                            ](energy, wavelengths),
                        self._em_cascade_function_dic[particle][
                            "long distro"
                            ](energy, z_grid),
                        self._em_cascade_function_dic[particle][
                            "angle distro"](
                                angle_grid,
                                n,
                                energy
                            )
                )

    def _track_builder(self):
        """ Builder function for the track functions.

        Parameters
        ----------
        interaction : str
            Optional: The interaction(s) which should produce the light

        Returns
        -------
        None
        """
        # Looping over the interaction types
        _log.info("Building track functions")
        self._track_functions_dic = {}
        for interaction in self._track_interactions:
            # Photon count function and angle function
            self._track_functions_dic[interaction] = {}

            # Building the counts functionn
            def track_mean(energy, interaction=interaction):
                """ Fetcher function for a specific particle
                and energy. This is for tracks

                Parameters
                ----------
                energy : float
                    The energy of the particle
                interaction : str
                    The interaction type

                Returns
                -------
                counts : float
                    The photon counts
                """
                tmp_track_frac = (
                    self.__track.additional_track_ratio(
                        energy, interaction=interaction
                    )
                )
                new_track = self._deltaL * (1. + tmp_track_frac)
                return new_track
            if config["general"]["jax"]:
                _log.debug(
                    "Constructing Jax function for " + interaction
                )

                def counts_mean(energy, wavelengths, interaction=interaction):
                    """ Calculates the differential photon counts.
                    Jax implemenation
                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: float/np.array
                        The wavelength(s) of interest
                    interaction : str
                        The interaction type

                    Returns
                    -------
                    differential_counts : np.array
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_mean(energy, interaction=interaction)
                    return self._cherenkov_counts_jax(wavelengths, new_track)
                # jitting
                counts = jit(counts_mean, static_argnames=['interactions'])
                angles = jit(self.__track.cherenkov_angle_distro)
            else:
                _log.debug(
                    "Constructing Jax function for " + interaction
                )

                def counts_mean(energy, wavelengths, interaction=interaction):
                    """Calculates the differential photon counts.

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: float/np.array
                        The wavelength(s) of interest
                    interaction : str
                        The interaction type

                    Returns
                    -------
                    differential_counts : np.array
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_mean(energy, interaction=interaction)
                    return self._cherenkov_counts(wavelengths, new_track)
                # No jitting here
                counts = counts_mean
                angles = self.__track.cherenkov_angle_distro
            self._track_functions_dic[interaction]["dcounts"] = counts
            self._track_functions_dic[interaction]["angles"] = angles

    def _em_cascade_builder(self):
        """ Builder function for a the cascade functions. This is for
        em cascades.

        Parameters
        ----------

        Returns
        -------
        None
        """
        _log.debug("Building the em cascade functions")
        self._em_cascade_function_dic = {}
        for particle_id in config["simulation"]["em particles"]:
            name = particle_id
            self._em_cascade_function_dic[particle_id] = {}

            def track_mean(energy, name=name):
                """ Fetcher function for a specific particle and energy.
                This is for em cascades and their photon counts

                Parameters
                ----------
                energy : float
                    The energy of the particle
                name : int
                    The name of the particle of interest

                Returns
                -------
                tmp_track : float
                    The track length
                """
                tmp_track, _ = (
                    self.__em_cascade.track_lengths(
                        energy, name
                    )
                )
                return tmp_track

            def track_sampler(energy, name=name):
                """ Fetcher function for a specific particle and energy.
                This samples the distribution

                Parameters
                ----------
                energy : float
                    The energy of the particle
                name : int
                    The name of the particle of interest

                Returns
                -------
                tmp_track_sample : float
                    The sampled photon counts
                """
                tmp_track, tmp_track_sd = (
                    self.__em_cascade.track_lengths(
                        energy, name
                    )
                )
                tmp_track_sample = (
                    tmp_track + tmp_track_sd * jnormal(self._rstate)
                )
                return tmp_track_sample

            def long_profile(energy, z_grid, name=name):
                """ The longitudinal profile of the em cascade

                Parameters
                ----------
                energy : float
                    The energy of the particle
                z_grid : float/np.array
                    The grid to evaluate the distribution in in cm
                name : int
                    The name of the particle of interest

                Returns
                -------
                long_distro : float/np.array
                    The resulting longitudinal distribution
                """
                return self.__em_cascade.long_profile(energy, z_grid, name)

            def angle_distro(
                    angles,
                    n=config["mediums"][self._medium]["refractive index"],
                    name=name
                    ):
                """ The angle distribution of the cherenkov photons for
                the em cascade

                Parameters
                ----------
                angles : float/np.array
                    The angles of interest
                n : float
                    Optional: The refractive index of the material
                name : int
                    Optional: The name of the particle of interest

                Returns
                -------
                angle_distro : float/np.array
                    The resulting longitudinal distribution
                """
                return self.__em_cascade.cherenkov_angle_distro(
                    angles, n, name
                )
            # Storing the functions
            if config["general"]["jax"]:
                _log.debug(
                    "Constructing Jax function for pdg_id " + str(name)
                )

                def counts_mean(energy, wavelengths, name=name):
                    """Calculates the differential photon counts.
                    Jax implemenation

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: float
                        The wavelength(s) of interest
                    name : int
                        Name of the particle

                    Returns
                    -------
                    differential_counts : float
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_mean(energy, name=name)
                    return self._cherenkov_counts_jax(wavelengths, new_track)

                def counts_sampler(energy, wavelengths, name=name):
                    """Calculates a sample of the differential photon counts.
                    Jax implemenation

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: float
                        The wavelength(s) of interest
                    name : int
                        Name of the particle

                    Returns
                    -------
                    differential_counts : float
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_sampler(energy, name=name)
                    return self._cherenkov_counts_jax(wavelengths, new_track)
                # Jit the jax functions
                counts = jit(counts_mean, static_argnames=['name'])
                counts_sample = jit(counts_sampler, static_argnames=['name'])
                long = jit(long_profile, static_argnames=['name'])
                angles = jit(angle_distro, static_argnames=['name'])
            else:
                _log.debug(
                    "Constructing numpy function for pdg_id " + str(name)
                )

                def counts_mean(energy, wavelengths, name=name):
                    """Calculates the differential photon counts.
                    Jax implemenation

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: np.array
                        The wavelength(s) of interest
                    name : int
                        Name of the particle

                    Returns
                    -------
                    differential_counts : np.array
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_mean(energy, name=name)
                    return self._cherenkov_counts(wavelengths, new_track)

                def counts_sampler(energy, wavelengths, name=name):
                    """Calculates a sample of the differential photon counts.
                    Jax implemenation

                    Parameters
                    ----------
                    energy : float
                        The energy of the particle in GeV
                    wavelengths: np.array
                        The wavelength(s) of interest
                    name : int
                        Name of the particle

                    Returns
                    -------
                    differential_counts : np.array
                        The differential photon counts
                        per track length (in cm).
                    """
                    new_track = track_sampler(energy, name=name)
                    return self._cherenkov_counts(wavelengths, new_track)
                # Don't jist the numpy functions
                counts = counts_mean
                counts_sample = counts_sampler
                long = long_profile
                angles = angle_distro
            # Storing
            self._em_cascade_function_dic[particle_id] = {
                "dcounts": counts,
                "dcounts sample": counts_sample,
                "long distro": long,
                "angle distro": angles,
            }

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

    def _cherenkov_counts_jax(
            self,
            wavelengths: float, track_lengths: float) -> float:
        """ Calculates the differential number of photons for the given
        wavelengths and track-lengths assuming a constant velocity with beta=1.

        Parameters
        ----------
        wavelengths : float
            The wavelengths of interest
        track_lengths : float
            The track lengths of interest in cm

        Returns
        -------
        counts : float
            The counts (differential)
        """
        prefac = (
            2. * jnp.pi * self._alpha * self._charge**2. /
            (1. - 1. / self._n**2.)
        )
        # 1e-7 due to the conversion from nm to cm
        diff_counts = (
            prefac / (wavelengths * 1e-9)**2. * track_lengths * 1e-2
        )
        return diff_counts * 1e-9 / jnp.pi
