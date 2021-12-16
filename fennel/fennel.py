# -*- coding: utf-8 -*-
# Name: fennel.py
# Authors: Stephan Meighen-Berger
# Main interface to the fennel model. Calculates the light yields using the
# Aachen parametrization from
# https://www.institut3b.physik.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaaapwhjz
# which is Leif Raedel's Master thesis

# Imports
# Native modules
import logging
import sys
import numpy as np
import yaml
# -----------------------------------------
# Package modules
from .config import config
from .particle import Particle
from .tracks import Track
from .em_cascades import EM_Cascade
from .hadron_cascades import Hadron_Cascade
from .photons import Photon
from .definition_generator import Definitions_Generator
try:
    from jax.random import PRNGKey
except ImportError:
    if config["general"]["jax"]:
        raise ImportError("Jax not found!")

# unless we put this class in __init__, __name__ will be contagion.contagion
_log = logging.getLogger("fennel")


class Fennel(object):
    """
    class: Fennel
    Interace to the fennel package. This class
    stores all methods required to run the simulation
    of the particle light yields
    Parameters
    ----------
    config : dic
        Configuration dictionary for the simulation

    Returns
    -------
    None
    """
    def __init__(self, userconfig=None):
        """
        function: __init__
        Initializes the class fennel.
        Here all run parameters are set.
        Parameters
        ----------
        config : dic
            Configuration dictionary for the simulation

        Returns
        -------
        None
        """
        # Inputs
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)

        # Create RandomState
        if config["general"]["random state seed"] is None:
            _log.warning("No random state seed given, constructing new state")
            if config["general"]["jax"]:
                rstate = PRNGKey(1337)
            else:
                rstate = np.random.RandomState()
        else:
            if config["general"]["jax"]:
                rstate = PRNGKey(config["general"]["random state seed"])
            else:
                rstate = np.random.RandomState(
                    config["general"]["random state seed"]
                )
        config["runtime"] = {"random state": rstate}

        # Logger
        # Logging formatter
        fmt = "%(levelname)s: %(message)s"
        fmt_with_name = "[%(name)s] " + fmt
        formatter_with_name = logging.Formatter(fmt=fmt_with_name)
        # creating file handler with debug messages
        if config["general"]["enable logging"]:
            fh = logging.FileHandler(
                config["general"]["log file handler"], mode="w"
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter_with_name)
            _log.addHandler(fh)
        else:
            _log.disabled = True
        # console logger with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(config["general"]["debug level"])
        # add class name to ch only when debugging
        if config["general"]["debug level"] == logging.DEBUG:
            ch.setFormatter(formatter_with_name)
        else:
            formatter = logging.Formatter(fmt=fmt)
            ch.setFormatter(formatter)
        _log.addHandler(ch)
        _log.setLevel(logging.DEBUG)
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Welcome to Fennel!')
        _log.info('This package will help you model light yields')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Creating particles...')
        self._particles = {}
        for particle_id in config["pdg id"].keys():
            # Particle creation
            self._particles[particle_id] = Particle(particle_id)
        _log.info('Creation finished')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Creating a track...')
        # Track creation
        self._track = Track()
        _log.info('Creation finished')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Creating an em cascade...')
        # EM cascade creation
        self._em_cascade = EM_Cascade()
        _log.info('Creation finished')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Creating a hadron cascade...')
        # Hadron cascade creation
        self._hadron_cascade = Hadron_Cascade()
        _log.info('Creation finished')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Creating a photon...')
        # Hadron cascade creation
        self._photon = Photon(
            self._particles, self._track,
            self._em_cascade, self._hadron_cascade
        )
        # Creating the definitions storer
        self._dg = Definitions_Generator(
            self._track, self._em_cascade, self._hadron_cascade
        )
        _log.info('Creation finished')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')

    def close(self):
        """ Wraps up the program

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        # A new simulation
        if config["general"]["enable logging"]:
            _log.debug(
                "Dumping run settings into %s",
                config["general"]["config location"],
            )
            with open(config["general"]["config location"], "w") as f:
                yaml.dump(config, f)
            _log.debug("Finished dump")
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info("Have a great day and until next time!")
        _log.info("                  @*****&@         @@.                    ")
        _log.info("           @@@((@ @*******@     @%((((@@((@               ")
        _log.info("         @(((((((@(@*******@@@((@(((((((((@*              ")
        _log.info("        @((((((@(((@*******@(((@%(((((&%(((@              ")
        _log.info("         #@((@((&@((&*******@((@(((((@#(((@               ")
        _log.info("        @@*****@((#@@********@@(((((@(((@/**@@            ")
        _log.info("        @********@((@@*************@@@#@******&%          ")
        _log.info("@@  @ @&(@(***,*******,*******,*******,*******,@      .@@*")
        _log.info(" @   @@((((@*************.*********...*******@(@      ./  ")
        _log.info(" @     %(((#@**********...*,.. ........**,**(@(#*     @   ")
        _log.info("  @      @@,.*.***,@@@.......*,...@@@..*,.....@@    &@    ")
        _log.info("   @(    @....,,...@@......... ,*.@%...*......,%  @@      ")
        _log.info("       @@@.....,***...............,****.......*@,         ")
        _log.info("        .........*............ ........****..*.@          ")
        _log.info("         #*........*...&       @...............@          ")
        _log.info("         @ *.. ... ..,,..@@@@. ... ... ... ...&           ")
        _log.info("          @.,*...........*,...................@           ")
        _log.info("            @.**.............*,..............@            ")
        _log.info("             .@.**...............**.........@             ")
        _log.info("           @@@%   &@@,........ .......**#@    @@          ")
        _log.info("        @,                 (@@@@@@@@(             @       ")
        _log.info("                                                %@*       ")
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        # Closing log
        logging.shutdown()

    def auto_yields(
            self, energy,
            particle: int,
            interaction='total',
            wavelengths=config["advanced"]["wavelengths"],
            angle_grid=config["advanced"]["angles"],
            n=config["mediums"][
                config["scenario"]["medium"]]["refractive index"],
            z_grid=config["advanced"]["z grid"],
            function=False):
        """ Auto fetcher function for a given particle and energy. This will
        fetch/evaluate the functions corresponding to the given particle.
        Some of the output will be none depending on the constructed object

        Parameters
        ----------
        energy : float
            The energy(ies) of the particle in GeV
        particle : int
            The pdg id of the particle of interest
        wavelengths : np.array
            Optional: The desired wavelengths
        interaction : str
            Optional: The interaction which should produce the light.
            This is used during track construction.
        angle_grid : np.array
            Optional: The desired angles in degress
        n : float
            Optional: The refractive index of the medium.
        z_grid : np.array
            Optional: The grid in cm for the long. distributions.
            Used when modeling cascades.
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
        em_fraction_mean : float/np.array
            The fraction of em particles
        em_fraction_sample : float/np.array
            A sample of the em_fraction
        long_profile : function/float/np.array
            The distribution along the shower axis for cm
        angles : function/float/np.array
            The angular distribution in degrees
        """
        if particle in config["simulation"]["track particles"]:
            _log.debug("Fetching/evaluating track functions for " +
                       str(particle))
            dcounts, angles = self.track_yields(
                energy,
                wavelengths=wavelengths,
                angle_grid=angle_grid,
                n=n,
                interaction=interaction,
                function=function
            )
            # Unfilled variables
            dcounts_s = None
            em_frac = None
            em_frac_s = None
            long = None
        elif particle in config["simulation"]["em particles"]:
            _log.debug("Fetching/evaluating em functions for " +
                       str(particle))
            dcounts, dcounts_s, long, angles = self.em_yields(
                energy,
                particle,
                wavelengths=wavelengths,
                angle_grid=angle_grid,
                n=n,
                z_grid=z_grid,
                function=function
            )
            # Unfilled variables
            em_frac = None
            em_frac_s = None
        elif particle in config["simulation"]["hadron particles"]:
            _log.debug("Fetching/evaluating hadron functions for " +
                       str(particle))
            dcounts, dcounts_s, em_frac, em_frac_s, long, angles = (
                self.hadron_yields(
                    energy,
                    particle,
                    wavelengths=wavelengths,
                    angle_grid=angle_grid,
                    n=n,
                    z_grid=z_grid,
                    function=function
                )
            )
        else:
            raise ValueError(
                "Track/cascade object corresponding to " +
                str(particle) + " is unknown. Please contact " +
                "the authors if there is a need for this species"
            )
        return dcounts, dcounts_s, em_frac, em_frac_s, long, angles

    def track_yields(
            self,
            energy: float,
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
            returns the functional form instead of the evaluation

        Returns
        -------
        differential_counts : np.array/function
            dN/dlambda The differential photon counts per track length (in cm).
            The shape of the array is len(wavelengths).
        angles : np.array/function
            The angular distribution in degrees
        """
        return self._photon._track_fetcher(
            energy, wavelengths, angle_grid, n, interaction, function
        )

    def em_yields(
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
        return self._photon._em_cascade_fetcher(
            energy, particle, wavelengths, angle_grid, n, z_grid, function
        )

    def hadron_yields(
            self, energy,
            particle: int,
            wavelengths=config["advanced"]["wavelengths"],
            angle_grid=config["advanced"]["angles"],
            n=config["mediums"][
                config["scenario"]["medium"]]["refractive index"],
            z_grid=config["advanced"]["z grid"],
            function=False):
        """ Fetcher function for a specific particle and energy. This is for
        hadron cascades.

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
        em_fraction_mean : float/np.array
            The fraction of em particles
        em_fraction_sample : float/np.array
            A sample of the em_fraction
        long_profile : function/float/np.array
            The distribution along the shower axis for cm
        angles : function/float/np.array
            The angular distribution in degrees
        """
        return self._photon._hadron_cascade_fetcher(
            energy, particle, wavelengths, angle_grid, n, z_grid, function
        )

    def definitions(self):
        """ Write the definitions file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._dg._write()

    def pars2csv(self):
        """ Write the parameters to a csv file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._dg._pars2csv()

    def hidden_function(self):
        """ Yaha! You found me!
        """
        print("                  @*****&@         @@.                    ")
        print("           @@@((@ @*******@     @%((((@@((@               ")
        print("         @(((((((@(@*******@@@((@(((((((((@*              ")
        print("        @((((((@(((@*******@(((@%(((((&%(((@              ")
        print("         #@((@((&@((&*******@((@(((((@#(((@               ")
        print("        @@*****@((#@@********@@(((((@(((@/**@@            ")
        print("        @********@((@@*************@@@#@******&%          ")
        print("@@  @ @&(@(***,*******,*******,*******,*******,@      .@@*")
        print(" @   @@((((@*************.*********...*******@(@      ./  ")
        print(" @     %(((#@**********...*,.. ........**,**(@(#*     @   ")
        print("  @      @@,.*.***,@@@.......*,...@@@..*,.....@@    &@    ")
        print("   @(    @....,,...@@......... ,*.@%...*......,%  @@      ")
        print("       @@@.....,***...............,****.......*@,         ")
        print("        .........*............ ........****..*.@          ")
        print("         #*........*...&       @...............@          ")
        print("         @ *.. ... ..,,..@@@@. ... ... ... ...&           ")
        print("          @.,*...........*,...................@           ")
        print("            @.**.............*,..............@            ")
        print("             .@.**...............**.........@             ")
        print("           @@@%   &@@,........ .......**#@    @@          ")
        print("        @,                 (@@@@@@@@(             @       ")
        print("                                                %@*       ")
