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
dark_map = [cm.get_cmap("Dark2")(i/8.) for i in range(8)]

df = pd.read_pickle('output/results.pkl')
print(df)

height = 3.50394/1.608
width = 3.50394
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.markersize': 5})
plt.rcParams.update({'figure.figsize': [width, height]})
plt.rcParams.update({'figure.autolayout': True})
fig = plt.figure()
plt.loglog(df["num_dofs"], df['error_bw'], '^-',
           label=r"$\eta_{\mathrm{bw}}$", color=dark_map[2])
plt.loglog(df["num_dofs"], df["exact_error"], '-',
           label="Exact error", color=dark_map[0])
plt.xlabel("Number of dof")
plt.ylabel("$\eta$")

marker_x, marker_y = marker(df["num_dofs"].values, [df["error_bw"].values, df["exact_error"].values], 0.3, 0.05)
annotation.slope_marker((marker_x, marker_y), (-1, 3), invert=True)
plt.legend()
plt.savefig('output/error.pdf')
