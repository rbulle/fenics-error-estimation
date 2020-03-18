## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpltools import annotation

def marker(x_data, y_datas, position, gap):
    middle = np.floor(len(x_data)/2.).astype(int)
    anchor_1_1 = []
    anchor_2_1 = []

    for data in y_datas:
        anchor_1_1.append(data[middle])
        anchor_2_1.append(data[-1])
    anchor_1_1 = min(anchor_1_1)
    anchor_2_1 = min(anchor_2_1)

    anchor_1_0 = x_data[middle]
    anchor_2_0 = x_data[-1]

    anchor_1 = [anchor_1_0, anchor_1_1]
    anchor_2 = [anchor_2_0, anchor_2_1]
    marker_x = anchor_1[0]**position*anchor_2[0]**(1.-position)\
             *(anchor_2[1]/anchor_1[1])**gap
    marker_y = anchor_1[1]**position*anchor_2[1]**(1.-position)\
             *(anchor_1[0]/anchor_2[0])**gap
    return marker_x, marker_y

import matplotlib.cm as cm
dark_map = [cm.get_cmap('Dark2')(i/8.) for i in range(8)]

df = pd.read_pickle("output/results.pkl")
print(df)

height = 3.50394/1.608
width = 3.50394
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.markersize': 5})
plt.rcParams.update({'figure.figsize': [width, height]})
plt.rcParams.update({'figure.autolayout': True})
fig = plt.figure()
plt.loglog(df["num_dofs"], df["error_hu"], '^-',
           label=r"$\eta_{u}$", color=dark_map[3])
plt.loglog(df["num_dofs"], df["error_hz"], '^-',
           label=r"$\eta_{z}$", color=dark_map[4])
plt.loglog(df["num_dofs"], df["error_hw"], '^-',
           label=r"$\eta_{w}$", color=dark_map[2])
plt.loglog(df["num_dofs"], df["error"], '^--',
           label=r"$\eta_{e}$", color=dark_map[0])
plt.xlabel("Number of dofs")
plt.ylabel("$\eta$")
marker_x, marker_y = marker(df["num_dofs"].values, [df["num_dofs"].values, df["error_hw"].values, df["error"].values], 0.4, 0.1)
annotation.slope_marker((marker_x, marker_y), (-1, 1), invert=True)
marker_x, marker_y = marker(df["num_dofs"].values, [df["num_dofs"].values, df["error_hu"].values, df["error_hz"].values], 0.4, 0.15)
annotation.slope_marker((marker_x, marker_y), (-0.5, 1), invert=True)
plt.legend(loc=(1.04,0.25))
plt.savefig("output/error.pdf")
