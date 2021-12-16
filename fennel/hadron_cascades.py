# -*- coding: utf-8 -*-
# Name: hadron_cascades.py
# Authors: Stephan Meighen-Berger
# Constructs a hadron cascade, defined by the emitted photons

import logging
import numpy as np
from scipy.special import gamma as gamma_func
from scipy.interpolate import UnivariateSpline
import pickle
import pkgutil
from .config import config
from .particle import Particle
try:
    import jax.numpy as jnp
    from jax.scipy.stats import gamma as jax_gamma
except ImportError:
    if config["general"]["jax"]:
        ImportError("Jax not found!")


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
        ValueError
            File containing the muon production data was not found
        """
        if not config["general"]["enable logging"]:
            _log.disabled = True
        _log.debug('Constructing a hadron cascade object')
        self._medium = config["mediums"][config["scenario"]["medium"]]
        self._n = self._medium["refractive index"]
        self._radlength = self._medium["radiation length"]
        self._Lrad = self._radlength / self._medium["density"]
        if config["scenario"]["parametrization"] == "aachen":
            _log.info("Loading the aachen parametrization")
            param_file = pkgutil.get_data(
                    __name__,
                    "data/%s.pkl" % config["scenario"]["parametrization"]
            )
            self._params = pickle.loads(param_file)["hadron cascade"]
            muon_data = pkgutil.get_data(
                    __name__,
                    "data/%s_muon_production.pkl" % (
                        config["scenario"]["parametrization"]
                    )
            )
            if muon_data is None:
                raise ValueError("Muon production data not found!")
            self.__muon_prod_dict = pickle.loads(muon_data)
            _log.debug("Constructing spline dictionary")
            self.__muon_prod_spl_pars = {}
            for pdg_id in config["simulation"]["hadron particles"]:
                self.__muon_prod_spl_pars[pdg_id] = {
                    "alpha": UnivariateSpline(
                        self.__muon_prod_dict[pdg_id][0],
                        self.__muon_prod_dict[pdg_id][1],
                        k=1, s=0, ext=3
                    ),
                    "beta": UnivariateSpline(
                        self.__muon_prod_dict[pdg_id][0],
                        self.__muon_prod_dict[pdg_id][2],
                        k=1, s=0, ext=3
                    ),
                    "gamma": UnivariateSpline(
                        self.__muon_prod_dict[pdg_id][0],
                        self.__muon_prod_dict[pdg_id][3],
                        k=1, s=0, ext=3
                    )
                }
        else:
            raise ValueError("Hadronic parametrization " +
                             config["scenario"]["parametrization"] +
                             " not implemented!")
        if config["general"]["jax"]:
            _log.info("Running with JAX functions")
            self.cherenkov_angle_distro = self._symmetric_angle_distro_jax
            self.track_lengths = self._track_lengths_fetcher_jax
            self.em_fraction = self._em_fraction_fetcher_jax
            self.long_profile = self._log_profile_func_fetcher_jax
            self.muon_production = self._muon_production_fetcher_jax
        else:
            _log.info("Running with basic functions")
            self.cherenkov_angle_distro = self._symmetric_angle_distro
            self.track_lengths = self._track_lengths_fetcher
            self.em_fraction = self._em_fraction_fetcher
            self.long_profile = self._log_profile_func_fetcher
            self.muon_production = self._muon_production_fetcher

    ###########################################################################
    # Numpy
    def _track_lengths_fetcher(
            self, E, particle: int):
        """ Parametrization for the energy dependence of the tracks

        Parameters
        ----------
        E : float/np.array
            The energy of interest in GeV
        particle : int
            The particle of interest

        Returns
        -------
        track_length : np.array
            The track lengths for different energies
        track_length_dev : np.array
            The track lengths deviations for different energies

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: \\alpha E^{\\beta}
        """
        params = self._params["track parameters"][particle]
        alpha = params["alpha"]
        beta = params["beta"]
        alpha_dev = params["alpha dev"]
        beta_dev = params["beta dev"]
        track_length = alpha * E**beta
        track_length_dev = alpha_dev * E**beta_dev
        return track_length, track_length_dev

    def _em_fraction_fetcher(self, E, particle: int):
        """ Parametrization of the EM contribution in a hadronic shower

        Parameters
        ----------
        E : float/np.array
            The energy of interest in GeV
        particle : int
            The particle of interest

        Returns
        -------
        em_fraction : np.array
            The fraction for the given energies
        em_fraction_sd : np.array
            The standard deviation

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: 1 - (1 - f_0)\\left(\\frac{E}{E_s}\\right)^{-m}
        """
        params = self._params["em fraction"][particle]
        Es = params["Es"]
        f0 = params["f0"]
        m = params["m"]
        sigma0 = params["sigma0"]
        gamma = params["gamma"]
        em_fraction = 1. - (1. - f0)*(E / Es)**(-m)
        em_fraction_sd = sigma0 * np.log(E)**(-gamma)
        return em_fraction, em_fraction_sd

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

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: b\\times \\frac{(tb)^{a-1}e^{-tb}}{\\Gamma(a)}
        """
        t = z / self._Lrad
        b = self._b_energy_fetcher(particle)
        a = self._a_energy_fetcher(E, particle)
        a = np.array([a]).flatten()
        # gamma.pdf seems far slower than the explicit implementation
        res = np.array([
            b * (
                (t * b)**(a_val - 1.) * np.exp(-(t*b)) / gamma_func(a_val)
            ) for a_val in a
        ])
        return res

    def _a_energy_fetcher(self, E: float, particle: int) -> np.array:
        """ Parametrizes the energy dependence of the a parameter for the
        longitudinal profiles

        Parameters
        ----------
        E : float
            The energy of interest in GeV
        particle : int
            The particle of interest

        Returns
        -------
        a : np.array
            The values for the energies of interest

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: \\alpha + \\beta log_{10}(E)
        """
        params = self._params["longitudinal parameters"][
            particle]
        alpha = params["alpha"]
        beta = params["beta"]
        a = alpha + beta * np.log10(E)
        return a

    def _b_energy_fetcher(self, particle: int) -> np.array:
        """ Parametrizes the energy dependence of the b parameter for the
        longitudinal profiles. Currently assumed to be constant

        Parameters
        ----------
        E : float
            The energy of interest in GeV
        particle : int
            The particle of interest

        Returns
        -------
        b : np.array
            The values for the energies of interest

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: b = b
        """
        params = self._params["longitudinal parameters"][
            particle]
        b = params["b"]
        return b

    def _symmetric_angle_distro(
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

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: a e^{b (1/n - cos(\\phi))^c} + d
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
            self, E, particle: int):
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

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: par = par_0  log(E) par_1
        """
        params = config[
            "hadron cascade"
        ]["angular distribution"][particle]
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

    def _muon_production_fetcher(self, Eprim, Emu, particle: Particle):
        """ Parametrizes the production of muons in hadronic cascades

        Parameters
        ----------
        Eprim : float/np.array
            The energy(ies) of the primary particle
        Emu : float/np.array
            The energy(ies) of the muons
        particle : Particle
            The particle of interest

        Returns
        -------
        distro: float/np.array
            The distribution/value of the produced muons

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: -\\alpha + \\beta\\left(\\frac{E}{GeV}\\right)^{-\\gamma}
        """
        # Converting the primary energy to an array
        energy_prim = np.array([Eprim]).flatten()
        # Converting to np.array for ease of use
        energy = np.array([Emu]).flatten()
        alpha, beta, gamma = self._muon_production_pars(energy_prim, particle)
        # Removing too small values
        energy[energy <= 1.] = 0.
        # Removing all secondary energies above the primary energy(ies)
        energy_2d = np.array([
            energy
            for _ in range(len(energy_prim))
        ])
        # Removing energies above the primary
        distro = []
        for id_arr, _ in enumerate(energy_2d):
            energy_2d[id_arr][
                energy_2d[id_arr] > energy_prim[id_arr]
            ] = 0.
            # Removing too large values
            energy[
                energy > (alpha[id_arr] / beta[id_arr])**(-1. / gamma[id_arr])
            ] = 0.
            distro.append(energy_prim[id_arr] * (
                -alpha[id_arr] + beta[id_arr] * (
                    energy_2d[id_arr]**(-gamma[id_arr])
                )
            ))
        distro = np.array(distro)
        distro[distro == np.inf] = 0.
        distro[distro < 0.] = 0.
        return distro

    def _muon_production_pars(self, E, particle: Particle):
        """ Constructs the parametrization values for the energies of interest.

        Parameters
        ----------
        E : float/np.array
            The energy(ies) of interest
        particle: Particle
            The particle of interest

        Returns
        -------
        alpha : float/np.array
            The first parameter values for the given energies
        beta : float/np.array
            The second parameter values for the given energies
        gamma : float/np.array
            The third parameter values for the given energies
        """
        alpha = self.__muon_prod_spl_pars[particle._pdg_id]["alpha"](E)
        beta = self.__muon_prod_spl_pars[particle._pdg_id]["beta"](E)
        gamma = self.__muon_prod_spl_pars[particle._pdg_id]["gamma"](E)
        return alpha, beta, gamma

    ###########################################################################
    # JAX
    def _track_lengths_fetcher_jax(
            self, E: float, particle: int):
        """ Parametrization for the energy dependence of the tracks

        Parameters
        ----------
        E : float
            The energy of interest in GeV
        particle : int
            The particle of interest

        Returns
        -------
        track_length : float
            The track lengths for different energies
        track_length_dev : float
            The track lengths deviations for different energies

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: \\alpha E^{\\beta}
        """
        params = self._params["track parameters"][particle]
        alpha = params["alpha"]
        beta = params["beta"]
        alpha_dev = params["alpha dev"]
        beta_dev = params["beta dev"]
        track_length = alpha * E**beta
        track_length_dev = alpha_dev * E**beta_dev
        return track_length, track_length_dev

    def _em_fraction_fetcher_jax(self, E: float, particle: int):
        """ Parametrization of the EM contribution in a hadronic shower

        Parameters
        ----------
        E : float
            The energy of interest in GeV
        particle : int
            The particle of interest

        Returns
        -------
        em_fraction : float
            The fraction for the given energies
        em_fraction_sd : float
            The standard deviation

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: 1 - (1 - f_0)\\left(\\frac{E}{E_s}\\right)^{-m}
        """
        params = self._params["em fraction"][particle]
        Es = params["Es"]
        f0 = params["f0"]
        m = params["m"]
        sigma0 = params["sigma0"]
        gamma = params["gamma"]
        em_fraction = 1. - (1. - f0)*(E / Es)**(-m)
        em_fraction_sd = sigma0 * jnp.log(E)**(-gamma)
        return em_fraction, em_fraction_sd

    def _log_profile_func_fetcher_jax(
            self, E: float, z: float, particle: int,
            ) -> float:
        """ Parametrization of the longitudinal profile. This still needs work

        Parameters
        ----------
        E : float
            The energy of interest in GeV
        z : float
            The cascade depth in cm
        particle : int
            The particle of interest

        Returns
        -------
        res : int
            Is equal to l^(-1) * dl/dt.

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: b\\times \\frac{(tb)^{a-1}e^{-tb}}{\\Gamma(a)}
        """
        t = z / self._Lrad
        b = self._b_energy_fetcher_jax(particle)
        a = self._a_energy_fetcher_jax(E, particle)
        res = jax_gamma.pdf(t * b, a) * b
        return res

    def _a_energy_fetcher_jax(self, E: float, particle: int) -> float:
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
        a : float
            The values for the energies of interest

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: \\alpha + \\beta log_{10}(E)
        """
        params = self._params["longitudinal parameters"][
            particle]
        alpha = params["alpha"]
        beta = params["beta"]
        a = alpha + beta * jnp.log10(E)
        return a

    def _b_energy_fetcher_jax(self, particle: int) -> int:
        """ Parametrizes the energy dependence of the b parameter for the
        longitudinal profiles. Currently assumed to be constant

        Parameters
        ----------
        particle : int
            The particle of interest

        Returns
        -------
        b : int
            The values for the energies of interest

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: b = b
        """
        params = self._params["longitudinal parameters"][
            particle]
        b = params["b"]
        return b

    def _symmetric_angle_distro_jax(
            self, E: float,
            phi: float, n: float,
            particle: int) -> float:
        # TODO: Add asymmetry function
        """ Calculates the symmetric angular distribution of the Cherenkov
        emission. The error should lie below 10%

        Parameters
        ----------
        E : float
            The energy of interest in GeV
        phi : float
            The angles of interest in degrees
        n : float
            The refractive index
        particle : int
            The particle of interest

        Returns
        -------
        distro : float
            The distribution of emitted photons given the angle. The
            result is a 2d array with the first axis for the angles and
            the second for the energies.

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: a e^{b (1/n - cos(\\phi))^c} + d
        """
        a, b, c, d = self._energy_dependence_angle_pars_jax(E, particle)
        distro = (
            (a * jnp.exp(b * jnp.abs(
                1. / n - jnp.cos(jnp.deg2rad(phi)))**c
            ) + d)
        )

        return jnp.nan_to_num(distro)

    def _energy_dependence_angle_pars_jax(
            self, E: float, particle: int):
        """ Parametrizes the energy dependence of the angular distribution
        parameters

        Parameters
        ----------
        E : float
            The energy of interest
        particle : iny
            The particle of interest

        Returns
        -------
        a : float
            The first parameter value for the given energy
        b : float
            The second parameter value for the given energy
        c : float
            The third parameter value for the given energy
        d : float
            The fourth parameter value for the given energy

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: par = par_0  log(E) par_1
        """
        params = self._params["angular distribution"][particle]
        a_pars = params["a pars"]
        b_pars = params["b pars"]
        c_pars = params["c pars"]
        d_pars = params["d pars"]
        a = a_pars[0] * (jnp.log(E))**a_pars[1]
        b = b_pars[0] * (jnp.log(E))**b_pars[1]
        c = c_pars[0] * (jnp.log(E))**c_pars[1]
        d = d_pars[0] * (jnp.log(E))**d_pars[1]
        return (
            a, b,
            c, d
        )

    def _muon_production_fetcher_jax(
            self, Eprim: float, Emu: float, particle: int
            ) -> float:
        """ Parametrizes the production of muons in hadronic cascades

        Parameters
        ----------
        Eprim : float
            The energy of the primary particle
        Emu : float
            The energy of the muons
        particle : int
            The particle of interest

        Returns
        -------
        distro: float
            The distribution/value of the produced muons

        Notes
        -----
        The analytical form of the parametrization is:

        .. math:: -\\alpha + \\beta\\left(\\frac{E}{GeV}\\right)^{-\\gamma}
        """
        energy_prim = Eprim
        energy = Emu
        alpha, beta, gamma = self._muon_production_pars_jax(
            energy_prim, particle
        )
        # Removing too small values
        if Emu < 1.:
            return 0.
        # Removing all secondary energies above the primary energy(ies)
        if Emu >= energy_prim:
            return 0.
        # Removing too large values
        if Emu > (alpha / beta)**(-1. / gamma):
            return 0.
        distro = (energy_prim * (
            -alpha + beta * (
                energy**(-gamma)
            )
        ))
        # Removing numerical errors
        if distro == np.inf:
            return 0.
        if distro < 0.:
            return 0.
        return distro

    def _muon_production_pars_jax(self, E: float, particle: int):
        """ Constructs the parametrization values for the energies of interest.

        Parameters
        ----------
        E : float
            The energy(ies) of interest
        particle: int
            The particle of interest

        Returns
        -------
        alpha : float
            The first parameter values for the given energy
        beta : float
            The second parameter values for the given energy
        gamma : float
            The third parameter values for the given energy
        """
        alpha = self.__muon_prod_spl_pars[particle]["alpha"](E)
        beta = self.__muon_prod_spl_pars[particle]["beta"](E)
        gamma = self.__muon_prod_spl_pars[particle]["gamma"](E)
        return alpha, beta, gamma
