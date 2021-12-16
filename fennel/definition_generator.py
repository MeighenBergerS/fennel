# -*- coding: utf-8 -*-
# Name: definition_generator.py
# Authors: Stephan Meighen-Berger
# Constructs a file containing the definitions of the different functions
# Also offers the option to convert the parameter file to a csv file

# Imports
import logging
import inspect
from typing import Dict
import collections
import pandas as pd
import pickle
import pkgutil
# Local imports
from .config import config
from .tracks import Track
from .em_cascades import EM_Cascade
from .hadron_cascades import Hadron_Cascade

_log = logging.getLogger(__name__)


class Definitions_Generator(object):
    """Helps with the construction of a definitions file.

    Parameters
    ----------
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
    """
    def __init__(
            self,
            track: Track,
            em_cascade: EM_Cascade, hadron_cascade: Hadron_Cascade):
        """Helps with the construction of a definitions file.

        Parameters
        ----------
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
        """
        if not config["general"]["enable logging"]:
            _log.disabled = True
        self._fname = config["advanced"]["generated definitions"]
        self._lines_to_write = []
        # The tracks
        self._lines_to_write.append(
            "# --------------------------------------------------\n",
        )
        self._lines_to_write.append(
            "# Tracks\n",
        )
        self._lines_to_write.append(
            "# --------------------------------------------------\n",
        )
        for val in track.__dict__.values():
            if callable(val):
                self._lines_to_write.append(
                    inspect.getsource(val)
                )
        # The em cascades
        self._lines_to_write.append(
            "# --------------------------------------------------\n",
        )
        self._lines_to_write.append(
            "# EM Cascades\n",
        )
        self._lines_to_write.append(
            "# --------------------------------------------------\n",
        )
        for val in em_cascade.__dict__.values():
            if callable(val):
                self._lines_to_write.append(
                    inspect.getsource(val)
                )
        # The hadron cascades
        self._lines_to_write.append(
            "# --------------------------------------------------\n",
        )
        self._lines_to_write.append(
            "# Hadronic Cascades\n",
        )
        self._lines_to_write.append(
            "# --------------------------------------------------\n",
        )
        for val in hadron_cascade.__dict__.values():
            if callable(val):
                self._lines_to_write.append(
                    inspect.getsource(val)
                )

    def _write(self):
        """ Write the definitions file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        with open(self._fname, 'w') as f:
            for line in self._lines_to_write:
                f.write(line)

    def _pars2csv(self):
        """ Converts the calculation parameters to a csv file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        param_file = pkgutil.get_data(
                __name__,
                "data/%s.pkl" % config["scenario"]["parametrization"]
        )
        params = pickle.loads(param_file)
        # Flatten
        params = self._flatten(params)
        pd.DataFrame.from_dict(
            data=params, orient='index'
            ).to_csv('parameters.csv', header=False)

    def _flatten(self, d: Dict, parent_key='', sep='_'):
        """ Helper function to flatten a dictionary of dictionaries

        Parameters
        ----------
        d : Dict
            The dictionary to flatten
        parent_key : str
            Optional: Key in the parent dictionary
        sep : str
            The seperator used

        Returns
        -------
        flattened_dic : dic
            The flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + str(k) if parent_key else str(k)
            if isinstance(v, collections.abc.MutableMapping):
                items.extend(self._flatten(v, str(new_key), sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
