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

df_bw = pd.read_pickle('output/linear/bank-weiser/results.pkl')
df_bw_struc = pd.read_pickle('output_structured/output/linear/bank-weiser/results.pkl')
df_res = pd.read_pickle('output/linear/residual/results.pkl')
df_res_struc = pd.read_pickle('output_structured/output/linear/residual/results.pkl')

print('BW adaptive:\n')
print(df_bw)

x = np.log(df_bw['num_dofs'].values[-3:])
y = np.log(df_bw['error_bw'].values[-3:])
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print('BW slope =', m)

y = np.log(df_bw['exact_error'].values[-3:])
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print('BW exact error slope =', m)

print('Residual adaptive:\n')
print(df_res)

x = np.log(df_res['num_dofs'].values[-3:])
y = np.log(df_res['error_res'].values[-3:])
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print('Res slope =', m)

y = np.log(df_res['exact_error'].values[-3:])
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print('Residual exact error slope =', m)

height = 3.50394/1.608
width = 3.50394
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.markersize': 5})
plt.rcParams.update({'figure.figsize': [width, height]})
plt.rcParams.update({'figure.autolayout': True})
plt.figure()
plt.loglog(df_bw["num_dofs"], df_bw['error_bw'], '^-',
           label=r"$\eta_{\mathrm{bw}}$", color=dark_map[0])
plt.loglog(df_bw["num_dofs"], df_bw["exact_error"], '-',
           label="Exact error (bw)", color=dark_map[1])
plt.loglog(df_res["num_dofs"], df_res['error_res'], '^-',
           label=r"$\eta_{\mathrm{res}}$", color=dark_map[4])
plt.loglog(df_res["num_dofs"], df_res["exact_error"], '-',
           label="Exact error (res)", color=dark_map[5])
plt.loglog(df_bw_struc["num_dofs"], df_bw_struc['error_bw'], '^--',
           label=r"$\eta_{\mathrm{bw}}$ (struct)", color=dark_map[8])
plt.loglog(df_bw_struc["num_dofs"], df_bw_struc["exact_error"], '--',
           label="Exact error (bw, struct)", color=dark_map[9])
plt.loglog(df_res_struc["num_dofs"], df_res_struc['error_res'], '^--',
           label=r"$\eta_{\mathrm{res}}$ (struct)", color=dark_map[12])
plt.loglog(df_res_struc["num_dofs"], df_res_struc["exact_error"], '--',
           label="Exact error (res, struct)", color=dark_map[13])


plt.xlabel("Number of dof")
plt.ylabel("$\eta$")

marker_x, marker_y = marker(df_bw["num_dofs"].values, [df_bw["error_bw"].values, df_bw["error_res"].values, df_bw["exact_error"].values], 0.4, 0.05)
annotation.slope_marker((marker_x, marker_y), (-1, 3), invert=True)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig('output/error.pdf')

plt.figure()
steps = np.arange(len(df_bw['num_dofs'].values))
bw_eff = np.divide(df_bw['exact_error'].values, df_bw['error_bw'].values)
res_eff = np.divide(df_bw['exact_error'].values, df_bw['error_res'].values)
print('BW eff:', bw_eff)
print('Res eff:', res_eff)
plt.plot(steps, bw_eff, '^-', label=r"$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{bw}}}$", color=dark_map[0])
plt.plot(steps, res_eff, '^-', label=r"$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{res}}}$", color=dark_map[4])

steps = np.arange(len(df_bw_struc['num_dofs'].values))
bw_eff_struct = np.divide(df_bw_struc['exact_error'].values, df_bw_struc['error_bw'].values)
res_eff_struct = np.divide(df_bw_struc['exact_error'].values, df_bw_struc['error_res'].values)
print('BW eff struct:', bw_eff_struct)
print('Res eff struct:', res_eff_struct)
plt.plot(steps, bw_eff_struct, '^-', label=r"$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{bw}}}$ (struct)", color=dark_map[8])
plt.plot(steps, res_eff_struct, '^-', label=r"$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{res}}}$ (struct)", color=dark_map[12])

xmin, xmax, ymin, ymax = plt.axis()
plt.hlines(1., xmin, xmax)
plt.xlabel('Refinement steps')
plt.ylabel('Efficiencies')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig('output/efficiencies.pdf')
