import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gr
import pandas as pd
from mpltools import annotation

k = 3

if k == 1:
    ncol = 3
else:
    ncol = 2

if k == 3:
    ymax = 20
else:
    ymax = 10

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

df_bw = pd.read_pickle('output/results.pkl')

print('Results:\n')
print(df_bw)

if 'zz' in df_bw:
    names = ['bw', 'ver', 'zz', 'res', 'exact']
    names_ltx = [r'$\eta_{\mathrm{bw}}$', r'$\eta_{\mathrm{ver}}$', r'$\eta_{\mathrm{zz}}$', r'$\eta_{\mathrm{res}}$', r'$\eta_{\mathrm{exact}}$']
    colors = [dark_map[0], dark_map[2], dark_map[6], dark_map[1], "black"] 
else:
    names = ['bw', 'ver', 'res', 'exact']
    names_ltx = [r'$\eta_{\mathrm{bw}}$', r'$\eta_{\mathrm{ver}}$', r'$\eta_{\mathrm{res}}$', r'$\eta_{\mathrm{exact}}$']
    colors = [dark_map[0], dark_map[2], dark_map[1], "black"] 

height = 3.50394/1.608
width = 3.50394
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'lines.markersize': 5})
plt.rcParams.update({'figure.figsize': [width, height]})
plt.rcParams.update({'figure.autolayout': True})
plt.figure()

for name, color, name_ltx in zip(names, colors, names_ltx):
    x = np.log(df_bw['dofs_{}'.format(name)].values[-5:])
    y = np.log(df_bw['{}'.format(name)].values[-5:])

    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    print('{} slope ='.format(name), m)

    if name is 'exact':
        plt.loglog(df_bw["dofs_{}".format(name)], df_bw['{}'.format(name)], '^--',
               label=name_ltx, color=color)
    else:
        plt.loglog(df_bw["dofs_{}".format(name)], df_bw['{}'.format(name)], '^-',
               label=name_ltx, color=color)

    plt.xlabel("Number of dof")
    plt.ylabel("$\eta$")

try:
    marker_x, marker_y = marker(df_bw["dofs_exact"].values, [df_bw['ver'].values, df_bw['res'].values, df_bw['zz'].values, df_bw['exact'].values, df_bw["bw"].values], 0.4, 0.1)
except:
    marker_x, marker_y = marker(df_bw["dofs_exact"].values, [df_bw['ver'].values, df_bw['res'].values, df_bw['exact'].values, df_bw["bw"].values], 0.4, 0.1)

annotation.slope_marker((marker_x, marker_y), (-k, 2), invert=True)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=ncol, mode="expand", borderaxespad=0.)
plt.savefig('output/error.pdf')

plt.figure()
names = names[:-1]
if 'zz' in df_bw:
    names_ltx = [r'$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{bw}}}$',r'$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{ver}}}$',r'$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{zz}}}$',r'$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{res}}}$']
else:
    names_ltx = [r'$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{bw}}}$',r'$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{ver}}}$',r'$\frac{||\nabla(u - u_h)||}{\eta_{\mathrm{res}}}$']

colors = colors[:-1]
for name, color, name_ltx in zip(names, colors, names_ltx):
    steps = np.arange(len(df_bw['dofs_{}'.format(name)].values))
    eff = np.divide(df_bw['{}'.format(name)].values, df_bw['{}_exact_error'.format(name)].values)
    print('{} eff:'.format(name), eff[-1])

    plt.plot(steps, eff, '^-', label=name_ltx, color=color)

xmin, xmax, ymin, ymax = plt.axis()
plt.hlines(1., xmin, xmax, linestyle="--", label='1', zorder= 10)
plt.xlabel('Refinement steps')
plt.ylabel('Efficiencies')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=ncol, mode="expand", borderaxespad=0.)
plt.yticks(np.arange(0, ymax, 2.0))
plt.savefig('output/efficiencies.pdf')


plt.rcParams.update({'font.size': 5})
plt.rcParams.update({'axes.labelsize': 5})
plt.rcParams.update({'lines.markersize': 2})
plt.rcParams.update({'figure.autolayout': True})

if 'zz' in df_bw:
    names_ltx = [r'$\eta_{\mathrm{bw}}$', r'$\eta_{\mathrm{ver}}$', r'$\eta_{\mathrm{zz}}$', r'$\eta_{\mathrm{res}}$']
else:
    names_ltx = [r'$\eta_{\mathrm{bw}}$', r'$\eta_{\mathrm{ver}}$', r'$\eta_{\mathrm{res}}$']

plt.figure()
N = len(names)-1

# Define the grid size for subplots
if N<2:
    grid = [N, 1]
if N<=4:
    grid = [2,2]
elif N<7:
    grid = [3,2]
else:
    gird = [3,3]

gs = gr.GridSpec(grid[0], grid[1])

ax = [plt.subplot(gs[l, j]) for l in range(grid[0])\
                                for j in range(grid[1])]
plt.subplots_adjust(wspace=0.25, hspace=0.3)

for i, name, name_ltx in zip(np.arange(len(names)), names, names_ltx):
    if name is 'exact':
        continue
    else:
        ax[i].loglog(df_bw['dofs_{}'.format(name)], df_bw['{}'.format(name)],\
                        '^-', color=colors[i],\
                       linewidth = 1.,\
                       label=name_ltx)

        ax[i].loglog(df_bw['dofs_{}'.format(name)], df_bw['{}_exact_error'.format(name)].values,\
                        color = 'black', zorder=10,\
                        linewidth= 1., label='Error')

        ax[i].legend(loc=1, ncol=1, facecolor='white',\
                        framealpha=0.5)

        ax[i].set_xlabel('Number of dof', labelpad=1)
        ax[i].set_ylabel(r'$\eta$', labelpad=1)
        ax[i].tick_params(axis='x', which='both', labelsize=3.5)

        if i == 0:
            marker_x, marker_y = marker(df_bw["dofs_exact"].values, [df_bw['{}_exact_error'.format(name)].values], 0.15, 0.1)
        else:
            marker_x, marker_y = marker(df_bw["dofs_exact"].values, [df_bw['{}_exact_error'.format(name)].values], 0.15, 0.01)
        annotation.slope_marker((marker_x, marker_y), (np.round(-k, 1), 2), invert=True, ax=ax[i])


if k>1:
    plt.delaxes(ax[-1])

plt.savefig('output/multi_conv.pdf')
