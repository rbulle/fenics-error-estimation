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
dark_map = [cm.get_cmap("tab20b")(i/20.) for i in range(20)]

df_linear = pd.read_pickle('output/linear/results.pkl')
df_quadratic = pd.read_pickle('output/quadratic/results.pkl')
print('Linear FEM:\n')
print(df_linear)
print('Quadratic FEM:\n')
print(df_quadratic)

height = 3.50394/1.608
width = 3.50394
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.markersize': 5})
plt.rcParams.update({'figure.figsize': [width, height]})
plt.rcParams.update({'figure.autolayout': True})
plt.figure()
plt.loglog(df_linear["num_dofs"], df_linear['error_bw'], '^-',
           label=r"$\eta_{\mathrm{bw}}$", color=dark_map[1])
plt.loglog(df_linear["num_dofs"], df_linear['error_res'], '^-',
           label=r"$\eta_{\mathrm{res}}$", color=dark_map[5])
plt.loglog(df_linear["num_dofs"], df_linear["exact_error"], '--',
           label="Exact error", color=dark_map[9])
plt.xlabel("Number of dof")
plt.ylabel("$\eta$")

marker_x, marker_y = marker(df_linear["num_dofs"].values, [df_linear["error_bw"].values, df_linear["error_res"].values, df_linear["exact_error"].values], 0.3, 0.05)
annotation.slope_marker((marker_x, marker_y), (-1, 3), invert=True)

plt.loglog(df_quadratic["num_dofs"], df_quadratic['error_bw'], '^-',
           label=r"$\eta_{\mathrm{bw}}$", color=dark_map[2])
plt.loglog(df_quadratic["num_dofs"], df_quadratic['error_res'], '^-',
           label=r"$\eta_{\mathrm{res}}$", color=dark_map[6])
plt.loglog(df_quadratic["num_dofs"], df_quadratic["exact_error"], '--',
           label="Exact error", color=dark_map[10])
marker_x, marker_y = marker(df_quadratic["num_dofs"].values, [df_quadratic["error_bw"].values, df_quadratic["error_res"].values, df_quadratic["exact_error"].values], 0.5, 0.1)
annotation.slope_marker((marker_x, marker_y), (-2, 3), invert=True)

plt.legend(loc='upper left')
plt.savefig('output/error.pdf')

plt.figure()
steps = np.arange(len(df_linear['num_dofs'].values))
bw_eff = np.divide(df_linear['exact_error'].values, df_linear['error_bw'].values)
res_eff = np.divide(df_linear['exact_error'].values, df_linear['error_res'].values)
plt.plot(steps, bw_eff, '^-', label=r"$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{bw}}}$", color=dark_map[2])
plt.plot(steps, res_eff, '^-', label=r"$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{res}}}$", color=dark_map[4])

steps = np.arange(len(df_quadratic['num_dofs'].values))
bw_eff = np.divide(df_quadratic['exact_error'].values, df_quadratic['error_bw'].values)
res_eff = np.divide(df_quadratic['exact_error'].values, df_quadratic['error_res'].values)
plt.plot(steps, bw_eff, '^-', label=r"$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{bw}}}$", color=dark_map[2])
plt.plot(steps, res_eff, '^-', label=r"$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{res}}}$", color=dark_map[4])

xmin, xmax, ymin, ymax = plt.axis()
plt.hlines(1., xmin, xmax)
plt.xlabel('Refinement steps')
plt.ylabel('Efficiencies')
plt.legend()
plt.savefig('output/efficiencies.pdf')
