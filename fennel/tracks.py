# -*- coding: utf-8 -*-
# Name: tracks.py
# Authors: Stephan Meighen-Berger
# Constructs a track, defined by the emission of photons

import logging
import numpy as np
import pickle
import pkgutil
from .config import config
# Checking if jax should be used
try:
    import jax.numpy as jnp
except ImportError:
    if config["general"]["jax"]:
        raise ImportError("Jax not found!")


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
        None
        """
        if not config["general"]["enable logging"]:
            _log.disabled = True
        _log.debug('Constructing a track object')
        self._medium = config["scenario"]["medium"]
        self._n = config["mediums"][self._medium]["refractive index"]
        if config["scenario"]["parametrization"] == "aachen":
            _log.info("Loading the aachen parametrization")
            param_file = pkgutil.get_data(
                    __name__,
                    "data/%s.pkl" % config["scenario"]["parametrization"]
            )
            self._params = pickle.loads(param_file)["track"]
        else:
            raise ValueError("Track parametrization " +
                             config["scenario"]["parametrization"] +
                             " not implemented!")
        if config["general"]["jax"]:
            _log.debug("Using JAX")
            self.additional_track_ratio = (
                self._additional_track_ratio_fetcher_jax
            )
            self.cherenkov_angle_distro = (
                self._symmetric_angle_distro_fetcher_jax
            )
        else:
            _log.debug("Using basic numpy")
            self.additional_track_ratio = self._additional_track_ratio_fetcher
            self.cherenkov_angle_distro = self._symmetric_angle_distro_fetcher

    ###########################################################################
    # Basic numpy
    def _additional_track_ratio_fetcher(
            self, E, interaction: str) -> np.array:
        """ Calculates the ratio between the additional track length
        and the original for a single energy.

        Parameters
        ----------
        E : float/np.array
            The energy of the particle in GeV
        interaction : str
            Name of the interaction

        Returns
        -------
        ratio : np.array
            The resulting ratio

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: \\lambda + \\kappa log(E)
        """
        params = self._params["additional track " + self._medium][
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

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: a e^{b (1/n - cos(\\phi))^c}
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
        params = self._params["angular distribution"]
        a_pars = params["a pars"]
        b_pars = params["b pars"]
        c_pars = params["c pars"]
        a = a_pars[0] * (np.log(E)) * a_pars[1]
        b = b_pars[0] * (np.log(E)) * b_pars[1]
        c = c_pars[0] * (np.log(E)) * c_pars[1]
        return a, b, c

    ###########################################################################
    # JAX
    def _additional_track_ratio_fetcher_jax(
            self, E: float, interaction: str) -> float:
        """ Calculates the ratio between the additional track length
        and the original for a single energy. JAX implementation

        Parameters
        ----------
        E : float
            The energy of the particle in GeV
        interaction : str
            Name of the interaction

        Returns
        -------
        ratio : float
            The resulting ratio

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: \\lambda + \\kappa log(E)
        """
        params = self._params["additional track " + self._medium][
            interaction
        ]
        lambd = params["lambda"]
        kappa = params["kappa"]
        ratio = (
            lambd + kappa * jnp.log(E)
        )
        return ratio

    def _symmetric_angle_distro_fetcher_jax(
            self,
            phi: float, n: float,
            E: float) -> float:
        # TODO: Add asymmetry function
        """ Calculates the symmetric angular distribution of the Cherenkov
        emission for a single energy. The error should lie below 10%.
        JAX implementation

        Parameters
        ----------
        phi : float
            The angles of interest in degrees
        n : float
            The refractive index
        E : float
            The energy of interest

        Returns
        -------
        distro : float
            The distribution of emitted photons given the angle. The
            result is a 2d array with the first axis for the angles and
            the second for the energies.

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: a e^{b (1/n - cos(\\phi))^c}
        """
        a, b, c = self._energy_dependence_angle_pars_jax(E)
        distro = (a * jnp.exp(b * jnp.abs(
                1. / n - jnp.cos(jnp.deg2rad(phi)))**c
            ))
        return distro

    def _energy_dependence_angle_pars_jax(
            self, E: float) -> float:
        """ Parametrizes the energy dependence of the angular distribution
        parameters. JAX implementation

        Parameters
        ----------
        E : jnp.array
            The energies of interest

        Returns
        -------
        a : float
            The first parameter value for the given energy
        b : float
            The second parameter value for the given energy
        c : float
            The third parameter value for the given energy

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: par = par_0  log(E) par_1
        """
        params = self._params["angular distribution"]
        a_pars = params["a pars"]
        b_pars = params["b pars"]
        c_pars = params["c pars"]
        a = a_pars[0] * (jnp.log(E)) * a_pars[1]
        b = b_pars[0] * (jnp.log(E)) * b_pars[1]
        c = c_pars[0] * (jnp.log(E)) * c_pars[1]
        return a, b, c
