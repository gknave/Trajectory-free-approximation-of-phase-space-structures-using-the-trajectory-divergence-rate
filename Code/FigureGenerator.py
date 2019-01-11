import numpy as np
import numpy.linalg as LA
import seaborn as sns
from scipy.interpolate import interp1d,interp2d,RectBivariateSpline
import matplotlib.pyplot as plt
import scipy.integrate as scint
from scipy.integrate import ode
from scipy import io
import scipy.optimize as opt
from matplotlib import cm
from matplotlib.collections import LineCollection
import matplotlib
import os
import manifoldid as mid
sns.set_style('ticks')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif', serif='Computer Modern')
labelfont = {'fontsize':9, 'fontname':'Computer Modern'}
tickfont = {'fontsize':8, 'fontname':'Computer Modern'}
matplotlib.rc('axes', linewidth=0.4, labelpad=1.2)
matplotlib.rc('xtick.major', size=2.5, pad=1, width=0.4)
matplotlib.rc('ytick.major', size=2.5, pad=1, width=0.4)
matplotlib.rc('figure.subplot', wspace=0.2)
matplotlib.rc('legend', frameon=False, labelspacing=0, borderpad=0.01, borderaxespad=0.7, handletextpad=0.5, handlelength=1.2)
matplotlib.rcParams['text.latex.preamble'].append(r'\usepackage{amsfonts}')

cdict = {'red':  [(0.0, 0.0000, 0.0000),
                  (0.125, 0.000, 0.000),
                  (0.5, 1.0000, 1.0000),
                  (0.875, 0.900, 0.900),
                  (1.0, 0.6000, 0.6000)],
        'green': [(0.0, 0.3270, 0.3270),
                  (0.125, 0.545, 0.545),
                  (0.5, 1.0000, 1.0000),
                  (0.875, 0.490, 0.490),
                  (1.0, 0.3270, 0.3270)],
        'blue':  [(0.0, 0.3270, 0.3270),
                  (0.125, 0.545, 0.545),
                  (0.5, 1.0000, 1.0000),
                  (0.875, 0.000, 0.000),
                  (1.0, 0.0000, 0.0000)]}
plt.register_cmap(name='CyOrDark', data=cdict)

cdictbr = {'red':  [(0.0, 0.2039, 0.2039),
                  (0.5, 1.0000, 1.0000),
                  (1.0, 0.9059, 0.9059)],
        'green': [(0.0, 0.5961, 0.5961),
                  (0.5, 1.0000, 1.0000),
                  (1.0, 0.2980, 0.2980)],
        'blue':  [(0.0, 0.8588, 0.8588),
                  (0.5, 1.0000, 1.0000),
                  (1.0, 0.2353, 0.2353)]}
plt.register_cmap(name='BlueRed', data=cdictbr)

# trajectorycolor = (0.74, 0.75, 0.75)
trajectorycolor = (0.44, 0.46, 0.46)

# phaseplotcolor = (0.2, 0.62, 0.65)
phaseplotcolor = (0.15, 0.55, 0.51)
##
## Linear saddle
##
def saddle(y):
  return np.array([y[0], -y[1]])

# Calculate Repulsion Rate
xlims = [-1.0, 1.0]; ylims = [-1.0, 1.0]

# Phase plot
plt.figure(figsize=(3.4, 2.8))
ax = plt.gca()
mid.advect_trajectories(saddle, xlims, ylims, color=trajectorycolor, linewidth=0.6, newfig=False, offset=0, N=13)
ylist = [[0.0001, 0], [-0.0001, 0], [0.1, 0.1], [0.1, -0.1], [-0.1, 0.1], [-0.1, -0.1],
  [0.2, 0.2], [0.2, -0.2], [-0.2, 0.2], [-0.2, -0.2]]
for y0 in ylist:
  t, y = mid.autonomous_odeint(saddle, y0, tf=10)
  ax.plot(y[:, 0], y[:, 1], color=trajectorycolor, linewidth=0.6)
  t, y = mid.autonomous_odeint(saddle, y0, tf=-10)
  ax.plot(y[:, 0], y[:, 1], color=trajectorycolor, linewidth=0.6)
mid.phase_plot(saddle, xlims, ylims, color=phaseplotcolor, newfig=False, paths=False)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.xlabel(r'$x$', **labelfont)
plt.ylabel(r'$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/PhasePlot_saddle.eps', transparent=True, bbox_inches='tight')


ds = 0.005
x1, x2, rho_dot = mid.repulsion_rate(saddle, xlims, ylims, ds, output=True, plot=False, newfig=False)

# Plot Repulsion Rate
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(rho_dot))
plt.figure(figsize=(3.4, 2.8))
ax = plt.gca()
mesh = ax.contourf(X1, X2, rho_dot, 501, cmap='CyOrDark', vmin=-lim, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True, ticks=[-1, -0.5, 0, 0.5, 1])
clb.ax.set_title(r'$\dot{\rho}$', fontsize=9, y=1.02)
clb.set_ticks([-1, -0.5, 0, 0.5, 1])
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.xlabel(r'$x$', **labelfont)
plt.ylabel(r'$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/RepRate_saddle.eps', transparent=True, bbox_inches='tight')
plt.savefig('../Figures/RepRate_saddle.png', dpi=500, transparent=True, bbox_inches='tight')

##
## Hopf bifurcation
##
def hopf(y, mu=-1, eps=0.25):
  xd = 1/eps*(2*mu-(y[0]**2+y[1]**2))*y[0]-y[1]
  yd = 1/eps*(2*mu-(y[0]**2+y[1]**2))*y[1]+y[0]
  return np.array([xd, yd])

# Calculate Repulsion Rate
xlims = [-4.0, 4.0]; ylims = [-4.0, 4.0]

# Phase plot
plt.figure(figsize=(3.4, 2.8))
ax = plt.gca()
mid.advect_trajectories(hopf, xlims, ylims, color=trajectorycolor, linewidth=0.6, newfig=False, offset=0, N=9)
# ylist = [[0.0001, 0.0001], [-0.0001, 0.0001], [0.0001, -0.0001], [-0.0001, -0.0001]]
# for y0 in ylist:
#   t, y = mid.autonomous_odeint(hopf, y0, tf=10)
#   ax.plot(y[:, 0], y[:, 1], color=trajectorycolor, linewidth=0.6)
mid.phase_plot(hopf, xlims, ylims, color=phaseplotcolor, newfig=False, paths=False)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-4.0, -2.0, 0.0, 2.0, 4.0], **tickfont)
plt.yticks([-4.0, -2.0, 0.0, 2.0, 4.0], **tickfont)
plt.xlabel(r'$x$', **labelfont)
plt.ylabel(r'$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/PhasePlot_hop1.eps', transparent=True, bbox_inches='tight')

def hopf(y, mu=1, eps=0.25):
  xd = 1/eps*(2*mu-(y[0]**2+y[1]**2))*y[0]-y[1]
  yd = 1/eps*(2*mu-(y[0]**2+y[1]**2))*y[1]+y[0]
  return np.array([xd, yd])

# Calculate Repulsion Rate
xlims = [-4.0, 4.0]; ylims = [-4.0, 4.0]

# Phase plot
plt.figure(figsize=(3.4, 2.8))
ax = plt.gca()
mid.advect_trajectories(hopf, xlims, ylims, color=trajectorycolor, linewidth=0.6, newfig=False, offset=0, N=9)
r = 0.0001; thlist = np.linspace(0, 2*np.pi, 8)
ylist = [[r*np.cos(th), r*np.sin(th)] for th in thlist]
# ylist = [[0.0001, 0.0001], [-0.0001, 0.0001], [0.0001, -0.0001], [-0.0001, -0.0001]]
for y0 in ylist:
  t, y = mid.autonomous_odeint(hopf, y0, tf=10)
  ax.plot(y[:, 0], y[:, 1], color=trajectorycolor, linewidth=0.6)
mid.phase_plot(hopf, xlims, ylims, color=phaseplotcolor, newfig=False, paths=False)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-4.0, -2.0, 0.0, 2.0, 4.0], **tickfont)
plt.yticks([-4.0, -2.0, 0.0, 2.0, 4.0], **tickfont)
plt.xlabel(r'$x$', **labelfont)
plt.ylabel(r'$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/PhasePlot_hop2.eps', transparent=True, bbox_inches='tight')



##
## Rotating hoop, repulsion rate and comparison with s2
##
def rotHoop(y, eps=0.02, gamma=2.3): #gamma=1.0 reproduces Strogatz figure
  return np.array([y[1], 1/eps*(np.sin(y[0])*(gamma*np.cos(y[0])-1)-y[1])])

xlims = [-3.14, 3.14]; ylims = [-2.4, 2.4] # xlims=[-6.28, 6.28] reproduces Strogatz figure

# Phase plot
plt.figure(figsize=(3.2, 2.1))
ax = plt.gca()
mid.advect_trajectories(rotHoop, xlims, ylims, color=trajectorycolor, linewidth=0.6, newfig=False, offset=0.1, N=12)
ylist = [[0.0001, 0], [-0.0001, 0], [-3.14, 0], [3.14, 0]]
for y0 in ylist:
  t, y = mid.autonomous_odeint(rotHoop, y0, tf=10)
  ax.plot(y[:, 0], y[:, 1], color=trajectorycolor, linewidth=0.6)
mid.phase_plot(rotHoop, xlims, ylims, color=phaseplotcolor, newfig=False, paths=False)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-3, -2, -1, 0, 1, 2, 3], **tickfont)
plt.yticks([-2, -1, 0, 1, 2], **tickfont)
plt.xlabel(r'$\phi$', **labelfont)
plt.ylabel(r'$\dot{\phi}$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/PhasePlot_rotating-hoop.eps', transparent=True, bbox_inches='tight')



# Calculate Repulsion Rate
x1, x2, rho_dot = mid.repulsion_rate(rotHoop, xlims, ylims, 0.01, output=True, plot=False, newfig=False)

# Plot Repulsion Rate
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(rho_dot))
plt.figure(figsize=(3.2, 2.1))
ax = plt.gca()
mesh = ax.contourf(X1, X2, rho_dot, 501, cmap='CyOrDark', vmin=-lim, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True, ticks=[-100, -50, 0, 50])
clb.ax.set_title(r'$\dot{\rho}$', fontsize=9, y=1.02)
clb.set_ticks([-100.0, -50.0, 0.0, 50.0])
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-3, -2, -1, 0, 1, 2, 3], **tickfont)
plt.yticks([-2, -1, 0, 1, 2], **tickfont)
plt.xlabel(r'$\phi$', **labelfont)
plt.ylabel(r'$\dot{\phi}$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/RepRate_rotating-hoop.eps', transparent=True, bbox_inches='tight')
plt.savefig('../Figures/RepRate_rotating-hoop.png', dpi=600, transparent=True, bbox_inches='tight')


x1, x2, nu_dot = mid.repulsion_ratio_rate(rotHoop, xlims, ylims, 0.01, output=True, plot=False, newfig=False)

# Plot Repulsion Rate
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(nu_dot))
plt.figure(figsize=(3.2, 2.1))
ax = plt.gca()
mesh = ax.contourf(X1, X2, nu_dot, 501, cmap='BlueRed', vmin=-lim, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\dot{\nu}$', fontsize=9, y=1.02)
clb.set_ticks([-200.0, -100.0, 0.0, 100.0, 200.0])
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-3, -2, -1, 0, 1, 2, 3], **tickfont)
plt.yticks([-2, -1, 0, 1, 2], **tickfont)
plt.xlabel(r'$\phi$', **labelfont)
plt.ylabel(r'$\dot{\phi}$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/RepRatio_rotating-hoop.png', dpi=300, transparent=True, bbox_inches='tight')
plt.savefig('../Figures/RepRatio_rotating-hoop.eps', transparent=True, bbox_inches='tight')

masked = np.ma.masked_where(np.logical_not(np.logical_or(np.logical_and(rho_dot<0, nu_dot<0), np.logical_and(rho_dot>0, nu_dot>0))), rho_dot)

X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(rho_dot))
plt.figure(figsize=(3.2, 2.1))
ax = plt.gca()
mesh = ax.contourf(X1, X2, masked, 501, cmap='CyOrDark', vmin=-lim, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\dot{\rho}$', fontsize=9, y=1.02)
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-3, -2, -1, 0, 1, 2, 3], **tickfont)
plt.yticks([-2, -1, 0, 1, 2], **tickfont)
plt.xlabel(r'$\phi$', **labelfont)
plt.ylabel(r'$\dot{\phi}$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/RepRate_masked_rotating-hoop.png', dpi=600, transparent=True, bbox_inches='tight')
plt.savefig('../Figures/RepRate_masked_rotating-hoop.eps', transparent=True, bbox_inches='tight')

# Calculate s1
x1, x2, s1 = mid.s1(rotHoop, xlims, ylims, 0.01, output=True, plot=False, newfig=False)

# Plot s1
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(s1))
plt.figure(figsize=(3.2, 2.1))
ax = plt.gca()
mesh = ax.contourf(X1, X2, s1, 501, cmap='bone', vmin=0, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$s_1$', fontsize=9, y=1.02)
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-3, -2, -1, 0, 1, 2, 3], **tickfont)
plt.yticks([-2, -1, 0, 1, 2], **tickfont)
plt.xlabel(r'$\phi$', **labelfont)
plt.ylabel(r'$\dot{\phi}$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/s1_rotating-hoop.eps', transparent=True, bbox_inches='tight')

x1, x2, s2 = mid.s2(rotHoop, xlims, ylims, 0.01, output=True, plot=False, newfig=False)

# Plot s2
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(s2))
plt.figure(figsize=(3.2, 2.1))
ax = plt.gca()
mesh = ax.contourf(X1, X2, s2, 501, cmap='bone', vmin=0, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$s_2$', fontsize=9, y=1.02)
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-3, -2, -1, 0, 1, 2, 3], **tickfont)
plt.yticks([-2, -1, 0, 1, 2], **tickfont)
plt.xlabel(r'$\phi$', **labelfont)
plt.ylabel(r'$\dot{\phi}$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/s2_rotating-hoop.eps', transparent=True, bbox_inches='tight')


# dictin = io.loadmat('ftle-reprate')
# x1 = dictin['x1']
# x2 = dictin['x2']
# rhoT = dictin['rhoT']
# sigmaT = dictin['sigmaT']
# Calculate rhoT
x1, x2, rhoT = mid.repulsion_factor(rotHoop, xlims, ylims, 0.01, -1.2, output=True, plot=False, newfig=False)

# Plot rhoT
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(rhoT))
plt.figure(figsize=(3.2, 2.1))
ax = plt.gca()
mesh = ax.contourf(X1, X2, rhoT, 501, cmap='inferno_r', vmin=0, vmax=lim/500)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\rho_T$', fontsize=9, y=1.02)
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-3, -2, -1, 0, 1, 2, 3], **tickfont)
plt.yticks([-2, -1, 0, 1, 2], **tickfont)
plt.xlabel(r'$\phi$', **labelfont)
plt.ylabel(r'$\dot{\phi}$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/rhoT_rotating-hoop.png', dpi=300, transparent=True, bbox_inches='tight')
plt.savefig('../Figures/rhoT_rotating-hoop.eps', transparent=True, bbox_inches='tight')

# Calculate ftle field
x1, x2, sigmaT = mid.ftle_field(rotHoop, xlims, ylims, 0.01, -1.2, output=True, plot=False, newfig=False)

# Plot ftle
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(sigmaT))
plt.figure(figsize=(3.2, 2.1))
ax = plt.gca()
mesh = ax.contourf(X1, X2, sigmaT, 501, cmap='gray', vmin=0, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\sigma_T$', fontsize=9, y=1.02)
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-3, -2, -1, 0, 1, 2, 3], **tickfont)
plt.yticks([-2, -1, 0, 1, 2], **tickfont)
plt.xlabel(r'$\phi$', **labelfont)
plt.ylabel(r'$\dot{\phi}$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/ftle_rotating-hoop.png', dpi=300, transparent=True, bbox_inches='tight')
plt.savefig('../Figures/ftle_rotating-hoop.eps', transparent=True, bbox_inches='tight')

##
## Haller example
##

def ex11(y):
  return np.array([np.tanh(y[0]**2/4)+y[0],y[0]+2*y[1]])

dictin = io.loadmat('ex11-metrics')
x1 = dictin['x1']
x2 = dictin['x2']
rho_dot = dictin['rho_dot']
s2 = dictin['s2']
rhoT = dictin['rhoT']
sigmaT = dictin['sigmaT']

xlims = [-2, 2]; ylims = [-2, 2] # xlims=[-6.28, 6.28] reproduces Strogatz figure

def ex11_r(y):
  return -ex11(y)

# Phase plot
plt.figure(figsize=(3.0, 2.6))
ax = plt.gca()
mid.advect_trajectories(ex11_r, xlims, ylims, color=trajectorycolor, linewidth=0.6, newfig=False, offset=0.1, N=9)
mid.phase_plot(ex11, xlims, ylims, color=phaseplotcolor, newfig=False, paths=False)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-2, -1, 0, 1, 2], **tickfont)
plt.yticks([-2, -1, 0, 1, 2], **tickfont)
plt.xlabel(r'$x$', **labelfont)
plt.ylabel(r'$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/PhasePlot_haller-example.eps', transparent=True, bbox_inches='tight')

# Calculate Repulsion Rate
# x1, x2, rho_dot = mid.divergence_rate(ex11, xlims, ylims, 0.01, output=True, plot=False, newfig=False)

# Plot Repulsion Rate
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(rho_dot))
plt.figure(figsize=(3.0, 2.6))
ax = plt.gca()
mesh = ax.pcolormesh(X1, X2, rho_dot, cmap='CyOrDark', vmin=-lim, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\dot{\rho}$', fontsize=9, y=1.02)
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-2, -1, 0, 1, 2], **tickfont)
plt.yticks([-2, -1, 0, 1, 2], **tickfont)
plt.xlabel(r'$x$', **labelfont)
plt.ylabel(r'$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/RepRate_haller-example.png', dpi=300, transparent=True, bbox_inches='tight')
# plt.savefig('../Figures/RepRate_haller-example.eps', transparent=True, bbox_inches='tight')

# Calculate s1
x1, x2, s1 = mid.s1(ex11, xlims, ylims, 0.01, output=True, plot=False, newfig=False)

# Plot s1
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(s1))
plt.figure(figsize=(3.0, 2.6))
ax = plt.gca()
mesh = ax.pcolormesh(X1, X2, s1, cmap='bone', vmin=0, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$s_1$', fontsize=9, y=1.02)
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-2, -1, 0, 1, 2], **tickfont)
plt.yticks([-2, -1, 0, 1, 2], **tickfont)
plt.xlabel(r'$x$', **labelfont)
plt.ylabel(r'$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/s1_haller-example.png', dpi=300, transparent=True, bbox_inches='tight')
plt.savefig('../Figures/s1_haller-example.eps', transparent=True, bbox_inches='tight')

# Calculate rhoT
x1, x2, rhoT = mid.repulsion_rate(ex11, xlims, ylims, 0.01, 1.2, output=True, plot=False, newfig=False)

# Plot rhoT
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(rhoT))
plt.figure(figsize=(3.0, 2.6))
ax = plt.gca()
mesh = ax.pcolormesh(X1, X2, rhoT, cmap='inferno_r', vmin=0, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\rho_T$', fontsize=9, y=1.02)
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-2, -1, 0, 1, 2], **tickfont)
plt.yticks([-2, -1, 0, 1, 2], **tickfont)
plt.xlabel(r'$x$', **labelfont)
plt.ylabel(r'$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/rhoT_haller-example.png', dpi=300, transparent=True, bbox_inches='tight')
# plt.savefig('../Figures/rhoT_haller-example.eps', transparent=True, bbox_inches='tight')

# Calculate ftle field
x1, x2, sigmaT = mid.ftle_field(ex11, xlims, ylims, 0.01, 1.2, output=True, plot=False, newfig=False)

# Plot ftle
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(sigmaT))
plt.figure(figsize=(3.0, 2.6))
ax = plt.gca()
mesh = ax.pcolormesh(X1, X2, sigmaT, cmap='gray', vmin=0, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\sigma_T$', fontsize=9, y=1.02)
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-2, -1, 0, 1, 2], **tickfont)
plt.yticks([-2, -1, 0, 1, 2], **tickfont)
plt.xlabel(r'$x$', **labelfont)
plt.ylabel(r'$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/ftle_haller-example.png', dpi=300, transparent=True, bbox_inches='tight')
# plt.savefig('../Figures/ftle_haller-example.eps', transparent=True, bbox_inches='tight')

dictout = {'x1':x1, 'x2':x2, 'rho_dot':rho_dot, 's1':s1, 'rhoT':rhoT, 'sigmaT':sigmaT}
io.savemat('ex11-metrics.mat', dictout)



##
## Glider model
##


def Cplate(alpha):
  return 1.4-np.cos(2*alpha), 1.2*np.sin(2*alpha)

def glider(v, theta=-np.pi/36, C=Cplate):
  #v = [vx, vz]
  psi = -np.arctan2(v[1], v[0])
  Yp = [(v[0]**2+v[1]**2)*(C(psi + theta)[1]*np.sin(psi)-C(psi + theta)[0]*np.cos(psi)), (v[0]**2+v[1]**2)*(C(psi + theta)[1]*np.cos(psi)+C(psi + theta)[0]*np.sin(psi))-1]
  return np.array(Yp)

dictin = io.loadmat('glider-metrics')
x1 = dictin['x1']
x2 = dictin['x2']
rho_dot = dictin['rho_dot']
nu_dot = dictin['nu_dot']
masked = dictin['masked']
s1 = dictin['s1']
s2 = dictin['s2']
rhoT = dictin['rhoT']
sigmaT = dictin['sigmaT']

xlims = [-1.1, 1.1]; ylims = [-1.1, 0] # xlims=[-6.28, 6.28] reproduces Strogatz figure


theta = -5*np.pi/180
C = Cplate

def bruteRootFind(f, a, b, args=(), steps=51):
  sols = []
  for P in np.linspace(a, b, steps):
    X, infodict, ier, mesg = opt.fsolve(f, P, args=args, full_output=True)
    if ier == 1 and (not np.any(np.abs(sols-X)<10**(-5))) and X<b and X>a:
      sols.append(X)
  return sols
def h(gamma, theta=0, C=Cplate):
  C0 = C(gamma+theta)
  return 1/np.tan(gamma)-C0[1]/C0[0]
sols = bruteRootFind(h, 10e-2, np.pi-10e-2, args=(theta, C))
sols = [float(s) for s in sols]
V = [(C(sol+theta)[1]*np.cos(sol)+C(sol+theta)[0]*np.sin(sol))**(-0.5) for sol in sols]
Vxz = [(v*np.cos(sol), -v*np.sin(sol)) for (v, sol) in zip(V, sols)]
Jac = ndt.Jacobian(glider)

vals, vecs = LA.eig(Jac(Vxz[0], theta, C))
wspts = [vxz+vecs[0]*0.001, vxz-vecs[0]*0.001]

wscolor = sns.color_palette()[1]
sscolor = sns.color_palette()[3]

# Phase plot
plt.figure(figsize=(3.4, 1.6))
ax = plt.gca()
mid.advect_trajectories(glider, xlims, ylims, color=trajectorycolor, linewidth=0.6, newfig=False, offset=0.1, N=11)
mid.phase_plot(glider, xlims, ylims, color=phaseplotcolor, newfig=False, paths=False)
mid.peelingOff(glider, [-1.5, 1.5], [-2, 2], theta, C, color=sscolor, linewidth=1.0)
for y0 in wspts:
  t, y = mid.autonomous_odeint(glider, y0, tf=-5)
  ax.plot(y[:, 0], y[:, 1], color=wscolor, linewidth=1.0)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-1, -0.5, 0, 0.5, 1], **tickfont)
plt.yticks([-1, -0.5, 0], **tickfont)
plt.xlabel(r'$v_x$', **labelfont)
plt.ylabel(r'$v_z$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/PhasePlot_glider-example.png', dpi=600, transparent=True, bbox_inches='tight')
plt.savefig('../Figures/PhasePlot_glider-example.eps', transparent=True, bbox_inches='tight')

# Calculate Repulsion Rate
# x1, x2, rho_dot = mid.repulsion_rate(glider, xlims, ylims, 0.01, output=True, plot=False, newfig=False)

# Plot Repulsion Rate
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(rho_dot))
plt.figure(figsize=(3.4, 1.6))
ax = plt.gca()
mesh = ax.contourf(X1, X2, rho_dot, 501, cmap='CyOrDark', vmin=-lim, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\dot{\rho}$', fontsize=9, y=1.02)
clb.set_ticks([-3.0, -2.0, -1.0, 0.0])
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-1, -0.5, 0, 0.5, 1], **tickfont)
plt.yticks([-1, -0.5, 0], **tickfont)
plt.xlabel(r'$v_x$', **labelfont)
plt.ylabel(r'$v_z$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/RepRate_glider-example.png', dpi=600, transparent=True, bbox_inches='tight')
plt.savefig('../Figures/RepRate_glider-example.eps', transparent=True, bbox_inches='tight')

# x1, x2, nu_dot = mid.repulsion_ratio_rate(glider, xlims, ylims, 0.01, output=True, plot=False, newfig=False)

X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(nu_dot))
plt.figure(figsize=(3.4, 1.6))
ax = plt.gca()
mesh = ax.contourf(X1, X2, nu_dot, 501, cmap='BlueRed', vmin=-lim, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\dot{\nu}$', fontsize=9, y=1.02)
clb.set_ticks([-5.0, -2.5, 0.0, 2.5, 5.0])
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-1, -0.5, 0, 0.5, 1], **tickfont)
plt.yticks([-1, -0.5, 0], **tickfont)
plt.xlabel(r'$v_x$', **labelfont)
plt.ylabel(r'$v_z$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/RepRatio_glider-example.png', dpi=600, transparent=True, bbox_inches='tight')
plt.savefig('../Figures/RepRatio_glider-example.eps', transparent=True, bbox_inches='tight')

# masked = np.ma.masked_where(np.logical_not(np.logical_or(np.logical_and(rho_dot<0, nu_dot<0), np.logical_and(rho_dot>0, nu_dot>0))), rho_dot)

X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(rho_dot))
plt.figure(figsize=(3.4, 1.6))
ax = plt.gca()
mesh = ax.contourf(X1, X2, masked, 501, cmap='CyOrDark', vmin=-lim, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\dot{\rho}$', fontsize=9, y=1.02)
clb.set_ticks([-3.0, -2.0, -1.0, 0.0])
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-1, -0.5, 0, 0.5, 1], **tickfont)
plt.yticks([-1, -0.5, 0], **tickfont)
plt.xlabel(r'$v_x$', **labelfont)
plt.ylabel(r'$v_z$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/RepRate_masked_glider-example.png', dpi=600, transparent=True, bbox_inches='tight')
plt.savefig('../Figures/RepRate_masked_glider-example.eps', transparent=True, bbox_inches='tight')

# Calculate s1
# x1, x2, s1 = mid.s1(glider, xlims, ylims, 0.01, output=True, plot=False, newfig=False)

# Plot s1
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(s1))
plt.figure(figsize=(3.4, 1.6))
ax = plt.gca()
mesh = ax.contourf(X1, X2, s1, 501, cmap='bone', vmin=-lim, vmax=0)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$s_1$', fontsize=9, y=1.02)
clb.set_ticks([-0.9, -0.6, -0.3, 0.0])
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-1, -0.5, 0, 0.5, 1], **tickfont)
plt.yticks([-1, -0.5, 0], **tickfont)
plt.xlabel(r'$v_x$', **labelfont)
plt.ylabel(r'$v_z$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/s1_glider-example.png', dpi=600, transparent=True, bbox_inches='tight')
plt.savefig('../Figures/s1_glider-example.eps', transparent=True, bbox_inches='tight')


# Calculate s2
# x1, x2, s2 = mid.s2(glider, xlims, ylims, 0.01, output=True, plot=False, newfig=False)

# Plot s2
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(s2))
plt.figure(figsize=(3.4, 1.6))
ax = plt.gca()
mesh = ax.contourf(X1, X2, s2, 501, cmap='bone', vmin=-lim, vmax=0)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$s_2$', fontsize=9, y=1.02)
clb.set_ticks([-6.0, -4.0, -2.0, 0.0])
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-1, -0.5, 0, 0.5, 1], **tickfont)
plt.yticks([-1, -0.5, 0], **tickfont)
plt.xlabel(r'$v_x$', **labelfont)
plt.ylabel(r'$v_z$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/s2_glider-example.png', dpi=600, transparent=True, bbox_inches='tight')
plt.savefig('../Figures/s2_glider-example.eps', transparent=True, bbox_inches='tight')

# Calculate rhoT
# x1, x2, rhoT = mid.repulsion_factor(glider, xlims, ylims, 0.01, -0.33, output=True, plot=False, newfig=False)

# Plot rhoT
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(rhoT))
plt.figure(figsize=(3.4, 1.6))
ax = plt.gca()
mesh = ax.contourf(X1, X2, rhoT, 501, cmap='inferno_r', vmin=1.0, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\rho_T$', fontsize=9, y=1.02)
clb.set_ticks([1.0, 2.0, 3.0])
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-1, -0.5, 0, 0.5, 1], **tickfont)
plt.yticks([-1, -0.5, 0], **tickfont)
plt.xlabel(r'$v_x$', **labelfont)
plt.ylabel(r'$v_z$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/rhoT_glider-example.png', dpi=600, transparent=True, bbox_inches='tight')
plt.savefig('../Figures/rhoT_glider-example.eps', transparent=True, bbox_inches='tight')

# Calculate ftle field
x1, x2, sigmaT = mid.ftle_field(glider, xlims, ylims, 0.01, -0.33, output=True, plot=False, newfig=False)

# Plot ftle
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(sigmaT))
plt.figure(figsize=(3.4, 1.6))
ax = plt.gca()
mesh = ax.contourf(X1, X2, sigmaT, 501, cmap='gray_r', vmin=0, vmax=lim)
clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\sigma_T$', fontsize=9, y=1.02)
clb.set_ticks([0.0, 5500, 11000, 16500, 22000])
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-1, -0.5, 0, 0.5, 1], **tickfont)
plt.yticks([-1, -0.5, 0], **tickfont)
plt.xlabel(r'$v_x$', **labelfont)
plt.ylabel(r'$v_z$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/ftle_glider-example.png', dpi=600, transparent=True, bbox_inches='tight')
plt.savefig('../Figures/ftle_glider-example.eps', transparent=True, bbox_inches='tight')

dictout = {'x1':x1, 'x2':x2, 'rho_dot':rho_dot, 'nu_dot':nu_dot, 'masked':masked, 's1':s1, 's2':s2, 'rhoT':rhoT, 'sigmaT':sigmaT}
io.savemat('glider-metrics.mat', dictout)




##
## Verhulst model, repulsion rate and repulsion ratio rate
##
def verhulst(y, eps=0.01):
  return [1, 1/eps*(y[0]*y[1]-y[1]**2)]

# Calculate Repulsion Rate
xlims = [-1.0, 1.0]; ylims = [-1.0, 1.0]
x1, x2, rho_dot = mid.divergence_rate(verhulst, xlims, ylims, 0.004, output=True, plot=False, newfig=False)

# Plot Repulsion Rate
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(rho_dot))
plt.figure(figsize=(3.4, 2.8))

ax = plt.gca()
mesh = ax.pcolormesh(X1, X2, rho_dot, cmap='CyOrDark', vmin=-lim, vmax=lim)
clb = plt.colorbar(mesh, ax=ax, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\dot{\rho}$', fontsize=9, y=1.02)
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.xlabel(r'$x$', **labelfont)
plt.ylabel(r'$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/RepRate_verhulst.eps', transparent=True, bbox_inches='tight')

# Calculate Repulsion Ratio Rate
x1, x2, nu_dot = mid.repulsion_ratio_rate(verhulst, xlims, ylims, 0.004, output=True, plot=False, newfig=False)

# Plot Repulsion Ratio Rate
X1, X2 = np.meshgrid(x1, x2)
lim = np.max(np.abs(nu_dot))
plt.figure(figsize=(3.4, 2.8))
ax = plt.gca()
mesh = ax.pcolormesh(X1, X2, nu_dot, cmap='BlueRed', vmin=-lim, vmax=lim)
clb = plt.colorbar(mesh, ax=ax, cax=None, use_gridspec=True)
clb.ax.set_title(r'$\dot{\nu}$', fontsize=9, y=1.02)
clb.set_ticks([-300, -150, 0, 150, 300], **tickfont)
# text = clb.ax.yaxis.label
# font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
# text.set_font_properties(font)
plt.xlim(xlims); plt.ylim(ylims)
plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.xlabel(r'$x$', **labelfont)
plt.ylabel(r'$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/RepRatio_verhulst.eps', transparent=True, bbox_inches='tight')

##
## Comparison of Rho Dot over a limit cycle trajectory
##
def vanderPol(y, eps=0.01, a=0.575):
  return np.array([1/eps*(y[1]-y[0]**3+y[0]), a-y[0]])

def resampleTrajectory(Xin, Yin, n=10):
  ds = np.sqrt((Xin[:-1]-Xin[1:])**2+(Yin[:-1]-Yin[1:])**2)
  ds = np.concatenate((np.array([0,]), ds))
  Xin = np.concatenate((np.array([Xin[0],]), Xin[ds>10e-8]))
  Yin = np.concatenate((np.array([Yin[0],]), Yin[ds>10e-8]))
  ds = np.concatenate((np.array([0,]), ds[ds>10e-8]))
  s = np.cumsum(ds)
  splX = interp1d(s, Xin, kind='cubic')
  splY = interp1d(s, Yin, kind='cubic')
  sNew = np.linspace(0, s[-2], n)
  return sNew, splX(sNew), splY(sNew)

# Phase plot
plt.figure(figsize=(3.2, 2.1))
ax = plt.gca()
mid.advect_trajectories(vanderPol, [-1.25, 1.25], [-1, 1], color=trajectorycolor, linewidth=0.6, newfig=False, offset=0, N=7)
xpoints = np.linspace(-0.5, 0.5, 5)
ylist = [[x+0.01, x**3-x] for x in xpoints]+[[x-0.01, x**3-x] for x in xpoints]
for y0 in ylist:
  t, y = mid.autonomous_odeint(vanderPol, y0, tf=10)
  ax.plot(y[:, 0], y[:, 1], color=trajectorycolor, linewidth=0.6)
  t, y = mid.autonomous_odeint(vanderPol, y0, tf=-10)
  ax.plot(y[:, 0], y[:, 1], color=trajectorycolor, linewidth=0.6)
mid.phase_plot(vanderPol, [-1.25, 1.25], [-1, 1], color=phaseplotcolor, newfig=False, paths=False)
plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.xlim([-1.25, 1.25])
plt.ylim([-1, 1])
plt.xlabel('$x$', **labelfont)
plt.ylabel('$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/PhasePlot_vanderPol.eps', transparent=True, bbox_inches='tight')


limit = 1.6
x1, x2, rhodot = mid.repulsion_rate(vanderPol, [-limit, limit], [-limit, limit], 0.002, output=True, plot=False, newfig=False)
rhofunction = interp2d(x1, x2, rhodot, kind='cubic')
Tvdp, Yvdp = mid.autonomous_odeint(vanderPol, [1.5, 1.5], tf=30)
sn, xn, yn = resampleTrajectory(Yvdp[:, 0], Yvdp[:, 1], n=3001)
rhoDotn = [rhofunction(xstep, ystep) for xstep, ystep in zip(xn, yn)]

# Show trajectory over Rho Dot field
vmin, vmax = -20, 20
# plt.figure(figsize=(3, 3))
# ax = plt.gca()
# X1, X2 = np.meshgrid(x1, x2)
# mesh = ax.pcolormesh(X1, X2, rhodot, cmap='CyOrDark', vmin=vmin, vmax=vmax)
# clb = plt.colorbar(mesh, cax=None, use_gridspec=True)
# clb.ax.set_title(r'$\dot{\rho}$', fontsize=9, y=1.02)
# text = clb.ax.yaxis.label
# font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
# text.set_font_properties(font)
# plt.savefig('../Figures/RepRate_vanderPol.eps', transparent=True, bbox_inches='tight')

# Plot repulsion vs arc length
plt.figure(figsize=(3, 2))
plt.plot(sn[:-80], np.zeros(sn[:-80].shape), linewidth=0.4, color=(0.2, 0.2, 0.2))
plt.plot(sn[:-80], rhoDotn[80:], linewidth=1.0, zorder=2)
plt.xlabel('Arc Length', **labelfont)
plt.ylabel(r'$\dot{\rho}$', **labelfont)
plt.xticks([0, 10, 20, 30, 40], **tickfont)
plt.yticks([-300, -150, 0], **tickfont)
plt.xlim([0, 45])
plt.ylim([-325, 25])
plt.savefig('../Figures/repulsion-values.eps', transparent=True, bbox_inches='tight')

# Rho Dot field for van der Pol
plt.figure(figsize=(3.2, 2.1))
# x1, x2, rhodot = mid.divergence_rate(vanderPol, [-1.25, 1.25], [-1, 1], 0.05, newfig=False, plot=False, output=True)
X1, X2 = np.meshgrid(x1, x2)
ax = plt.gca()
# vmin = -10; vmax=10
vmin, vmax = -20, 20
mesh = ax.pcolormesh(X1, X2, rhodot, cmap='CyOrDark', vmin=vmin, vmax=vmax)
clb = plt.colorbar(mesh, ax=ax)
# clb.set_ticks([-20, -10, -5, 0, 5, 10])
clb.ax.set_title(r'$\dot{\rho}$', fontsize=9, y=1.02)
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.xlim([-1.25, 1.25])
plt.ylim([-1, 1])
plt.xlabel('$x$', **labelfont)
plt.ylabel('$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
plt.savefig('../Figures/RepRate_vanderPol.eps', transparent=True, bbox_inches='tight')

# Rho Dot of a limit cycle
plt.figure(figsize=(3.2, 2.1))
ax = plt.gca()
# vmax=10; vmin=-10
CyOrDark = matplotlib.colors.LinearSegmentedColormap('CyOrDark', cdict)
ax.plot(xn[80:], yn[80:], color=(0.0, 0.0, 0.0), linewidth=1.2, zorder=1)
points = np.array([xn[80:], yn[80:]]).T.reshape([-1, 1, 2])
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=CyOrDark, norm=plt.Normalize(0, 1))
lc.set_array(((np.array(rhoDotn[80:])-vmin)/(vmax-vmin)).reshape(sn[80:].shape))
lc.set_linewidths(0.7)
lc.set_zorder(2)
ax.add_collection(lc)
# ax.scatter(xn, yn, color=CyOrDark(((np.array(rhoDotn)-vmin)/(vmax-vmin)).reshape(sn.shape)), edgecolors=(0.0, 0.0, 0.0), linewidths=0.1, s=7)
plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], **tickfont)
plt.xlim([-1.25, 1.25])
plt.ylim([-1, 1])
plt.xlabel('$x$', **labelfont)
plt.ylabel('$y$', **labelfont)
ax.set_aspect('equal', adjustable='box', anchor='C')
clb = plt.colorbar(mesh, orientation='vertical', ax=ax)
text = clb.ax.yaxis.label
font = matplotlib.font_manager.FontProperties(family='Computer Modern', size=8)
text.set_font_properties(font)
clb.ax.set_title(r'$\dot{\rho}$', fontsize=10, y=1.02)
plt.savefig('../Figures/limit-cycle_RepRate-v2.eps', transparent=True, bbox_inches='tight')


####
#### Extra junk for now
####

# Lighter Cyan Orange colormap
cdict = {'red':  [(0.0, 0.0000, 0.0000),
                  (0.5, 1.0000, 1.0000),
                  (1.0, 1.0000, 1.0000)],
        'green': [(0.0, 0.5450, 0.5450),
                  (0.5, 1.0000, 1.0000),
                  (1.0, 0.5450, 0.5450)],
        'blue':  [(0.0, 0.5450, 0.5450),
                  (0.5, 1.0000, 1.0000),
                  (1.0, 0.0000, 0.0000)]}
plt.register_cmap(name='CyanOrange', data=cdict)



##
## Calculation of Mean Rho Dot
##
def rotHoop(y, eps=0.02, gamma=2.3): #gamma=1.0 reproduces Strogatz figure
  return np.array([y[1], 1/eps*(np.sin(y[0])*(gamma*np.cos(y[0])-1)-y[1])])

limit = 11
# x0 = -3.0 # y0 = 0.432
# x0 = -0.1 # -0.120
def event(t, y):
  return np.abs(y[1])-limit
event.terminal = True
def tempFunc(t, y):
  return rotHoop(y)

points = 1001
x1, x2, rhodot = mid.repulsion_rate(rotHoop, [-5, 5], [-11.5, 11.5], 0.005, output=True, plot=False, newfig=False)
X1, X2 = np.meshgrid(x1, x2)
rhofunction = RectBivariateSpline(x2, x1, rhodot)
out = np.zeros(points)
t_span1 = [0, -10]
x0=-3.0
for k, y0 in enumerate(np.linspace(0.422, 0.442, points)):
  sol = scint.solve_ivp(tempFunc, t_span1, [x0, y0], method='BDF', events=event, max_step=0.01)
  yback = sol.y.transpose()
  # tback = sol.t
  # tback, yback = scipy.integrate.solve_ivp(rotHoop, time, [x0, y0], method='BDF', events=event)
  # tback, yback = mid.autonomous_odeint(rotHoop, [x0, y0], tf=-1)
  tfor, yfor = mid.autonomous_odeint(rotHoop, [x0, y0], tf=30)
  y = np.concatenate((yback[-1:0:-1, :], yfor))
  # Xin = y[np.abs(y[:, 1]) < limit][:, 0]
  # Yin = y[np.abs(y[:, 1]) < limit][:, 1]
  # print(k)
  sn, xn, yn = resampleTrajectory(y[:, 0], y[:, 1], n=2001)
  out[k] = np.sum(np.array([rhofunction(ystep, xstep)*ds for xstep, ystep, ds in zip(xn[:-1], yn[:-1], sn[1:]-sn[:-1])]))/sn[-1]

# Mean Rho Dot versus y position, rotating hoop
plt.figure(figsize=(3,2))
ax = plt.gca()
ax.plot(np.linspace(0.422,0.442, points), out)
plt.xlabel('$y_0$', **labelfont)
plt.ylabel(r'$(\Sigma\dot{\rho}ds)/s$', **labelfont)
# plt.xticks([-3, -1.5, 0, 1.5, 3], **tickfont)
# plt.yticks([-11, -9, -7, -5, -3], **tickfont)
# plt.xlim((-3, 3))
# plt.ylim((-11, -3))
plt.savefig('../Figures/rho-dot-field-zoom.eps', transparent=True, bbox_inches='tight')

# Mean Rho Dot versus y position zoomed in, rotating hoop
plt.figure(figsize=(3,2))
plt.plot(np.linspace(-3, 3, points), out)
plt.xlabel('$y_0$', **labelfont)
plt.ylabel(r'$(\Sigma\dot{\rho}ds)/s$', **labelfont)
plt.xlim([0.3, 0.6])
plt.ylim([-10.5, -9])
# plt.xticks([-1, -0.75, -0.5, -0.25, 0, 0.25], fontsize=8)
# plt.yticks([-1.75, -1.5, -1.25], fontsize=8)
plt.savefig('../Figures/rho-dot-field-zoom2.eps', transparent=True, bbox_inches='tight')

# Minimum Rho Dot location on a vector field, rotating hoop
plt.figure(figsize=(3, 2))
mid.advect_trajectories(rotHoop, [-3.14, 3.14], [-3, 3], newfig=False, offset=0.1, N=10, linewidth=0.6, color=trajectorycolor)
mid.phase_plot(rotHoop, [-3.14, 3.14], [-3, 3], newfig=False, paths=False, color=phaseplotcolor)
ax = plt.gca()
ax.scatter([-3, -0.1, 0.1, 3], [0.432, -0.12, 0.12, -0.432], color=sns.color_palette()[3], s=2, zorder=4)
for Y0 in [[-3, 0.432], [-0.1, -0.12], [0.1, 0.12], [3, -0.432]]:
  Ttraj, Ytraj = mid.autonomous_odeint(rotHoop, Y0, tf=5)
  ax.plot(Ytraj[:, 0], Ytraj[:, 1], linewidth=0.5, color=sns.color_palette()[3])
plt.xlim([-3.14, 3.14])
plt.xticks([-3, -2, -1, 0, 1, 2, 3], fontsize=8)
plt.ylim([-3, 3])
plt.yticks([-3, -2, -1, 0, 1, 2, 3], fontsize=8)
plt.xlabel(r'$\phi$', fontsize=10)
plt.ylabel(r'$\dot{\phi}$', fontsize=10)
plt.savefig('../Figures/min-rho-dot.eps', transparent=True, bbox_inches='tight')
