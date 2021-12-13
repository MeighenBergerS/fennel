# -*- coding: utf-8 -*-
# Name: pyface.py
# Authors: Stephan Meighen-Berger
# Main interface to the pyface model. simplifies the running of the Geant4 code

# Imports
# Native modules
import os
from subprocess import Popen, PIPE
import yaml
# -----------------------------------------
# Package modules
from .config import config


class PyFace(object):
    """ Utility class to make dealing with the Geant4 scripts easier.

    Parameters
    ----------
    config : dic
        Configuration dictionary for the simulation
    """
    def __init__(self, userconfig=None):
        # Inputs in the form of the config
        if userconfig is not None:
            if isinstance(userconfig, dict):
                config.from_dict(userconfig)
            else:
                config.from_yaml(userconfig)
        # Setting basic parameters
        self._dump = config["general"]["dump"]
        self._w_dir = config["general"]["working_directory"]
        self._b_dir = config["general"]["build directory"]
        self._o_dir = config["general"]["output_location"]
        self._E = config["scenario"]["energy"]
        self._particle = config["scenario"]["particle"]
        self._prod_cut = config["scenario"]["production_cut"]
        self._events = config["scenario"]["events"]
        self._progress = config["scenario"]["progress_printing"]
        self._threads = config["scenario"]["threads"]
        self._recompile = config["scenario"]["recompile"]
        self._bash_commands = config["general"]["bash commands"]
        self._mac_file = config["general"]["mac file"]

    def simulation(
            self,
            energy=config["scenario"]["energy"],
            particle=config["scenario"]["particle"],
            ):
        """ Runs the Geant4 simulation. Simulation results are stored in
        the working directory (config file). Bash output is stored in
        the output directory (config file).

        Parameters
        ----------
        energy : str
            Optional: Defines the energy of the injected particle. If not
            defined the config value is used.
        particle : str
            Optional: Defines the injected particle species. If not defined
            the config value is used.

        Returns
        -------
        None
        """
        print("--------------------------------------------------------------")
        print("Generating the mac file. This tells the geant4 scripts")
        print("what the simulation should look like")
        self._generate_mac_file(
            energy=energy,
            particle=particle,
            production_cut=self._prod_cut,
            events=self._events,
            progress_printing=self._progress,
            threads=self._threads
        )
        print("--------------------------------------------------------------")
        print("Running (and compiling) the Geant4 scripts")
        self._bash_execution(
            bash_commands=self._bash_commands,
            working_directory=self._w_dir,
            output_location=self._o_dir,
            recompile=self._recompile
        )
        # Now executing (and compiling) the bash
        if self._dump:
            with open(config["general"]["config location"], "w") as f:
                yaml.dump(config, f)

    def _bash_execution(
            self,
            bash_commands: list,
            working_directory: str,
            output_location: str,
            recompile: bool):
        """ Executes the passed bash commands and writes the resulting output
        to files.

        Parameters
        ----------
        bash_commands: list
            A list of bash commands (each as a string)
            The first element should be the cmake command
            The 2nd one should be the make command.
            The 3rd command should be the geant4 execution
        working_directory: str
            Optional: The location the bash commands should be executed at
        output_location: str
            Optional: The location the output should be dumped to
        recompile: bool:
            If the code needs to be compiled again
        """
        if recompile:
            run_commands = bash_commands
        else:
            run_commands = [bash_commands[0], bash_commands[2]]
        for id_command, bash_command in enumerate(run_commands):
            # Store switch
            if self._dump:
                f_store = open(
                    output_location + 'command%d.txt' % id_command, 'w')
            with Popen(
                    bash_command.split(), stdout=PIPE, bufsize=1,
                    universal_newlines=True,
                    cwd=working_directory) as p:
                for line in p.stdout:
                    if self._dump:
                        f_store.write(line)
                    print(line, end='')

    def _generate_mac_file(
            self,
            energy: str,
            particle: str,
            production_cut: str,
            events: str,
            progress_printing: str,
            threads: str):
        """ Constructs the run mac file used by the Geant4 code. Strings need
        to contain the appropiate units if necessary (ones accepted by Geant4)

        Parameters
        ----------
        energy : str
            Optional: Energy of the injected particle
        particle : str
            Optional: Particle species
        production_cut : str
            Optional: Minimal track length for produced particles
        events : str
            Optional: Number of events to generate
        progress_printing : str
            Optional: After how many injected events to print some results
        threads : str
            Optional: Number of threads to use. Note this requires Geant4 to be
            compiled with multi-threading support!

        Returns
        -------
        None

        Raises
        ------
        ValueError: energy
            The energy input is not supported or too high
        """
        try:
            print("Removing the old mac file")
            os.remove("../py_run.mac")
        except FileNotFoundError:
            print("No file to remove")
        # Some energy checks
        if (energy[-3:] == "PeV" or energy[-3:] == "EeV"):
            raise ValueError("PeV and EeV not supported!")
        if (energy[-3:] == "TeV"):
            if float(energy[:-3]) > 10:
                raise ValueError("Energy too high!")
        f_store = open(self._mac_file, 'w')
        lines = [
            "# Macro file for example showers\n",
            "#\n",
            "# Change the default number" +
            " of workers (in multi-threading mode) \n",
            "/run/numberOfThreads %s\n" % threads,
            "#\n",
            "# Initialize kernel\n",
            "/run/initialize\n",
            "#\n",
            "# Frequency of printing\n",
            "/run/printProgress %s\n" % progress_printing,
            "#\n",
            "# set cut in run\n",
            "/run/setCut %s\n" % production_cut,
            "#\n",
            "# The events\n",
            "#\n",
            "/gun/particle %s\n" % particle,
            "/gun/energy %s\n" % energy,
            "/run/beamOn %s\n" % events,
        ]
        for line in lines:
            f_store.write(line)
