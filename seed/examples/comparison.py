# comparison.py
# Authors: Stephan Meighen-Berger
# Comparing fennel and pyface
# Note this requires fennel to be installed (pip install fennel-seed)

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle as pkl
from fennel import Fennel
# package
sys.path.append("../")
from pyface import PyFace, config

# Location where the images should be stored
str_pics = "../pics/"
store_name = "/mnt/c/Users/steph/Desktop/e_10_GeV"
# initializing
number_of_runs = "1000"
config["scenario"]["events"] = number_of_runs
pf = PyFace()
fe = Fennel()
energy_inj = 1e1
# Running the simulation
pf.simulation(
    energy="10 GeV",
    particle="e-"
)

# Running fennel
dcounts, dcounts_sampler, em_frac, em_frac_sample, long, angles = (
    fe.auto_yields(energy_inj, 11, function=True)
)

# Loading data
str_build_dir = config["general"]["build directory"]
str_edeposit = 'Shower_h1_Etarget.csv'
str_meantrack = 'Shower_h1_Ltarget.csv'
str_pos = 'Shower_h2_Target_XZ.csv'
str_track = 'Shower_h2_Target_TrackZ.csv'
str_file_name = str_build_dir + str_edeposit
dataE = pd.read_csv(str_file_name, skiprows=6)
str_file_name = str_build_dir + str_meantrack
dataL = pd.read_csv(str_file_name, skiprows=6)
str_file_name = str_build_dir + str_pos
dataPos = pd.read_csv(str_file_name, skiprows=9)
str_file_name = str_build_dir + str_track
dataTrack = pd.read_csv(str_file_name, skiprows=9)

# Fixing up the positional data
pos_vector = np.array(dataPos["entries"]).reshape((1000 + 2, 1000 + 2))

print('Creating shower plot')

# The particles over all runs
underflow_cut = 1
overflow_cut = 1
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(
    pos_vector[underflow_cut:-overflow_cut, underflow_cut:-overflow_cut].T,
    aspect='auto', extent=(-20, 20, -2.5, 2.5,)
)
ax.autoscale(False)
ax.set_xscale("linear")
ax.set_yscale("linear")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlabel(r"$Z\;\mathrm{[m]}$")
ax.set_ylabel(r"$X\;\mathrm{[m]}$")
fig.savefig(store_name + "Shower.png", dpi=500)
with open(store_name + "shower_data.pkl", "wb") as f:
    pkl.dump(
        pos_vector[underflow_cut:-overflow_cut, underflow_cut:-overflow_cut].T,
        f)


# Constructing the multiplicity projection
multiplicity = np.sum(
    pos_vector[underflow_cut:-overflow_cut, underflow_cut:-overflow_cut].T,
    axis=0)

# Fixing up the track data
track_vector_unweighted = np.array(dataTrack["entries"]).reshape((1000 + 2, 10000 + 2))
track_vector_weighted = np.array(dataTrack["Sw"]).reshape((1000 + 2, 10000 + 2))

# Constructing the weights track lengths projection
underflow_cut = 1
overflow_cut = 1
number_of_runs_int = int(number_of_runs)
track_vector_u_w = (
    track_vector_unweighted[
        underflow_cut:-overflow_cut,
        underflow_cut:-overflow_cut].T *
        (np.logspace(-9, 2, 10000) / 1e3 / number_of_runs_int)[:, np.newaxis]
)
track_vector_w_w = (
    track_vector_weighted[
        underflow_cut:-overflow_cut,
        underflow_cut:-overflow_cut].T *
        (np.logspace(-9, 2, 10000) / 1e3 / number_of_runs_int)[:, np.newaxis]
)

# Track sums for z:
track_sum_unweighted = np.sum(
    track_vector_u_w,
    axis=0
)
track_sum_weighted = np.sum(
    track_vector_w_w,
    axis=0
)

track_cumsum_unweighted = np.cumsum(track_sum_unweighted)
track_cumsum_weighted = np.cumsum(track_sum_weighted)

print('Creating track length plot')

z_grid = np.linspace(0., 40., 1000)

# The differential of the cummulative sum of the individuals
underflow_cut = 1
overflow_cut = 1
fig, ax = plt.subplots(figsize=(10, 10))
# ax.plot(
#     np.linspace(0., 40., 1000)[:-1],
#     np.diff(track_cumsum_unweighted) / np.diff(np.linspace(0., 40., 1000) * 1e2) / track_cumsum_unweighted[-1]
# )
ax.plot(
    np.linspace(0., 40., 1000)[:-1],
    np.diff(track_cumsum_weighted) / np.diff(np.linspace(0., 40., 1000) * 1e2) / track_cumsum_weighted[-1],
    label=r'Geant4'
)
ax.plot(
    z_grid,
    long(energy_inj, z_grid * 1e2)[0] / np.trapz(long(energy_inj, z_grid * 1e2)[0], x=z_grid*1e2),
    label=r'Fennel'
)
ax.set_xscale("linear")
ax.set_yscale("log")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0., 20)
ax.set_ylim(1e-7, 1e-2)
ax.set_xlabel(r"$Z\;\mathrm{[m]}$")
ax.set_ylabel(r"$Diff.\;Current\;Track\;Length$")
plt.legend()
with open(store_name + "comp_data.pkl", "wb") as f:
    pkl.dump(
        [[
            np.linspace(0., 40., 1000)[:-1],
            np.diff(track_cumsum_weighted) /
            np.diff(np.linspace(0., 40., 1000) * 1e2) /
            track_cumsum_weighted[-1]],
        [
            z_grid,
            long(energy_inj, z_grid * 1e2)[0] /
            np.trapz(long(energy_inj, z_grid * 1e2)[0], x=z_grid*1e2)
          ]],
        f)
fig.savefig(store_name + "Fennel_Comparison.png", dpi=500)
