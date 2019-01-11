#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:21:47 2017

@author: gknave
"""

import numpy as np
import numpy.linalg as LA
from skimage import measure
from mayavi import mlab
from PIL import Image
import scipy.integrate as scint
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt

def autonomous_odeint(func, y0, t0=0, dt=0.01, tf=200, ret_success=False, stiff=True):
  dt = np.abs(dt)*np.sign(tf)
  def odefun(t, v):
    return func(v)
  if stiff:
    r = scint.ode(odefun).set_integrator('vode', atol=10**(-8), rtol=10**-8, nsteps=50000, method='bdf')
  else:
    r = scint.ode(odefun).set_integrator('lsoda', atol=10**(-8), rtol=10**(-8))
  r.set_initial_value(y0, t0)
  Y = []; T = []; Y.append(y0); T.append(t0)
  while r.successful() and r.t/tf < 1:
    r.integrate(r.t+dt)
    Y.append(np.float_(r.y))
    T.append(r.t)
  if ret_success:
    return T, Y, r.successful()
  return np.array(T), np.array(Y)

# def rk4Normal(func, y0, ds):
#   k1 = func(y0)
#   k2 = func(y0+k1*ds/2)
#   k3 = func(y0+k2*ds/2)
#   k4 = func(y0+k3*ds)
#   y1 = y0 + ds/6*(k1+2*k2+2*k3+k4)
#   v1 = func(y1)
#   vp = (v1-k1)/ds
#   return vp/np.sqrt(np.dot(vp, vp))

# def frenet(func, y0, ds):
#   v = func(y0)
#   t = v/np.sqrt(np.dot(v, v))
#   n = rk4Normal(func, y0, ds)
#   b = np.cross(t, n)/
#   return t, n, b

# def testfun(y, eps=0.001):
#   return np.array([-y[0]-y[2], -y[1]-y[2], 1/eps*(y[0]**2-y[1]**2-y[2])])

def repulsionRate3(func, xlims, ylims, zlims, ds, *fargs):
  x1 = np.arange(xlims[0], xlims[1]+ds, ds)
  x2 = np.arange(ylims[0], ylims[1]+ds, ds)
  x3 = np.arange(zlims[0], zlims[1]+ds, ds)
  X1, X2, X3 = np.meshgrid(x1, x2, x3)
  U, V, W = zip(*[tuple(func(Y, *fargs)) for Y in zip(X1.ravel(), X2.ravel(), X3.ravel())])
  U = np.reshape(np.array(U), np.shape(X1))
  V = np.reshape(np.array(V), np.shape(X1))
  W = np.reshape(np.array(W), np.shape(X1))
  # [DUz, DUy, DUx] = np.gradient(U[:, :, :], ds, edge_order=2)
  # [DVz, DVy, DVx] = np.gradient(V[:, :, :], ds, edge_order=2)
  # [DWz, DWy, DWx] = np.gradient(W[:, :, :], ds, edge_order=2)
  [DUy, DUx, DUz] = np.gradient(U[:, :, :], ds, edge_order=2)
  [DVy, DVx, DVz] = np.gradient(V[:, :, :], ds, edge_order=2)
  [DWy, DWx, DWz] = np.gradient(W[:, :, :], ds, edge_order=2)
  rho_dot = np.zeros(np.shape(U))
  J = np.array([[0, 1], [-1, 0]])
  output = []
  output2 = []
  for (u, v, w, dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz) in zip(U.ravel(), V.ravel(), W.ravel(), DUx.ravel(), DUy.ravel(), DUz.ravel(), DVx.ravel(), DVy.ravel(), DVz.ravel(), DWx.ravel(), DWy.ravel(), DWz.ravel()):
    Utemp = np.array([u, v, w])
    t = Utemp/np.sqrt(np.dot(Utemp, Utemp))
    Grad = np.array([[dux, duy, duz], [dvx, dvy, dvz], [dwx, dwy, dwz]])
    S = 0.5*(Grad + np.transpose(Grad))
    Uhat = np.array([[0, -Utemp[2], Utemp[1]], [Utemp[2], 0, Utemp[0]], [-Utemp[1], Utemp[0], 0]])
    tprime = np.dot(Uhat, np.dot(Grad, Utemp))/np.dot(Utemp, Utemp)**(1.5)
    n1 = tprime/np.sqrt(np.dot(tprime, tprime))
    n1 = n1-np.dot(n1, t)*t
    n2 = np.cross(n1, t)
    N = np.vstack([n1, n2]).transpose()
    RhoDot = np.dot(np.transpose(N), np.dot(S, N))
    vals = LA.eigvals(RhoDot)
    output.append(vals[np.argmax(vals)])
    output2.append(vals[np.argmin(vals)])
  out1 = np.reshape(np.array(output), np.shape(X1))
  out2 = np.reshape(np.array(output2), np.shape(X1))
  return x1, x2, x3, out1, out2

def repulsionFactor3(func, xlims, ylims, zlims, ds, T):
  x1 = np.arange(xlims[0], xlims[1]+ds, ds)
  x2 = np.arange(ylims[0], ylims[1]+ds, ds)
  x3 = np.arange(zlims[0], zlims[1]+ds, ds)
  X1, X2, X3 = np.meshgrid(x1, x2, x3)
  U, V, W = zip(*[tuple(func(Y)) for Y in zip(X1.ravel(), X2.ravel(), X3.ravel())])
  U = np.reshape(np.array(U), np.shape(X1))
  V = np.reshape(np.array(V), np.shape(X1))
  W = np.reshape(np.array(W), np.shape(X1))
  yOut = np.zeros(np.concatenate(((3,), np.shape(U))))
  for m in range(len(x1)):
    for n in range(len(x2)):
      for l in range(len(x3)):
        y0 = np.array([X1[n, m, l], X2[n, m, l], X3[n, m, l]])
        U[n, m, l], V[n, m, l], W[n, m, l] = func(y0)
        time, Y = autonomous_odeint(func, y0, tf=T)
        yOut[:, n, m, l] = Y[-1, :]
  [DUy, DUx, DUz] = np.gradient(yOut[0, :, :, :], ds, edge_order=2)
  [DVy, DVx, DVz] = np.gradient(yOut[1, :, :, :], ds, edge_order=2)
  [DWy, DWx, DWz] = np.gradient(yOut[2, :, :, :], ds, edge_order=2)
  J = np.array([[0, 1], [-1, 0]])
  output = []
  output2 = []
  for (u, v, w, dux, duy, duz, dvx, dvy, dvz, dwx, dwy, dwz) in zip(U.ravel(), V.ravel(), W.ravel(), DUx.ravel(), DUy.ravel(), DUz.ravel(), DVx.ravel(), DVy.ravel(), DVz.ravel(), DWx.ravel(), DWy.ravel(), DWz.ravel()):
    Utemp = np.array([u, v, w])
    t = Utemp/np.sqrt(np.dot(Utemp, Utemp))
    Grad = np.array([[dux, duy, duz], [dvx, dvy, dvz], [dwx, dwy, dwz]])
    C = np.dot(np.transpose(Grad), Grad)
    Uhat = np.array([[0, -Utemp[2], Utemp[1]], [Utemp[2], 0, Utemp[0]], [-Utemp[1], Utemp[0], 0]])
    tprime = np.dot(Uhat, np.dot(Grad, Utemp))/np.dot(Utemp, Utemp)**(1.5)
    n1 = tprime/np.sqrt(np.dot(tprime, tprime))
    n1 = n1-np.dot(n1, t)*t
    n2 = np.cross(n1, t)
    vals = [1/np.sqrt(np.dot(n1, np.dot(C, n1))), 1/np.sqrt(np.dot(n2, np.dot(C, n2)))]
    output.append(vals[np.argmax(vals)])
    output2.append(vals[np.argmax(vals)])
  out1 = np.reshape(np.array(output), np.shape(X1))
  out2 = np.reshape(np.array(output2), np.shape(X1))
  return x1, x2, x3, out1, out2


# Hopf bifurcation
def hopf(Y, eps=0.25):
  r2 = Y[0]**2+Y[1]**2
  xd = 1/eps*(2*Y[2]-r2)*Y[0]-Y[1]
  yd = 1/eps*(2*Y[2]-r2)*Y[1]+Y[0]
  mud = 0
  return np.array([xd, yd, mud])

ds = 0.06
x1, x2, x3, out1, out2 = repulsionRate3(hopf, [-4, 4], [-4, 4], [-4, 4], ds)
out3 = []
for outs in zip(out1.ravel(), out2.ravel()):
  out3.append(np.real(outs[np.argmax(np.abs(outs))]))
out3 = np.array(out3).reshape(out1.shape)
mlab.figure(size=(1600,2400))
grid = mlab.pipeline.scalar_field(out3)
grid.spacing=[ds, ds, ds]
lim = np.max(np.abs(out3))/3
mlab.pipeline.image_plane_widget(grid, plane_orientation='x_axes', slice_index=len(x1)//2, colormap='coolwarm', vmin=-lim, vmax=lim)
mlab.pipeline.image_plane_widget(grid, plane_orientation='y_axes', slice_index=len(x2)//2, colormap='coolwarm', vmin=-lim, vmax=lim)
# mlab.pipeline.image_plane_widget(grid, plane_orientation='z_axes', slice_index=len(x3)//2, colormap='coolwarm', vmin=-lim, vmax=lim)
mlab.outline(color=(0, 0, 0))
rp = np.linspace(0.0001, np.sqrt(8), 400)
thp = np.linspace(0, 2*np.pi, 400)
Rp, Thp = np.meshgrid(rp, thp)
Xp, Yp = Rp*np.cos(Thp), Rp*np.sin(Thp)
Zp = 0.5*(Xp**2+Yp**2)
mlab.mesh(Xp+5, Yp+5, Zp+5, color=(0.2, 0.2, 0.8))
t = np.linspace(0, 4, 1001)
mlab.plot3d(np.zeros(t.shape)+5, np.zeros(t.shape)+5, -t+5, color=(0.2, 0.2, 0.8), tube_radius=0.12, tube_sides=16)
mlab.plot3d(np.zeros(t.shape)+5, np.zeros(t.shape)+5, t+5, color=(0.8, 0.2, 0.2), tube_radius=0.12, tube_sides=16)
zrange = [-3, -1, 1, 3]
for z in zrange:
  for r in np.linspace(0.0001, 4.0, 2):
    for th in np.linspace(0, 2*np.pi, 9):
      x = r*np.cos(th); y = r*np.sin(th)
      temp, ytemp = autonomous_odeint(hopf, [x, y, z], tf=3.5)
      mlab.plot3d(ytemp[:, 0]+5, ytemp[:, 1]+5, ytemp[:, 2]+5, color=(0.0, 0.0, 0.0))
imgmap_RGBA = mlab.screenshot(mode='rgba', antialiased=True)
img_RGBA = Image.fromarray(np.array(imgmap_RGBA*255, dtype=np.uint8))
img_RGBA.save('../Figures/Hopf-surf-iso-trajs-loose.png')
mlab.show()

# Haller (2001)
def predPrey(Y, eps=0.2, m1=0.2, m2=0.5, r1=5, r2=4, a=3, d=1, b=1):
  u1d = m2-(m1+m2)*Y[0]+eps*Y[0]*(1-Y[0])*(r1-r2-a*Y[2])
  nd = eps*Y[1]*(r1*Y[0]+r2*(1-Y[0])-a*Y[0]*Y[2])
  pd = eps*Y[2]*(b*Y[0]*Y[1]-d)
  return np.array([u1d, nd, pd])

ds = 0.4
x1, x2, x3, out1, out2 = repulsionRate3(predPrey, [-1, 2.5], [0, 12], [0, 4], ds)
out3 = []
for outs in zip(out1.ravel(), out2.ravel()):
  out3.append(outs[np.argmax(np.abs(outs))])
out3 = np.array(out3).reshape(out1.shape)
grid = mlab.pipeline.scalar_field(out3)
grid.spacing=[ds, ds, ds]
lim = np.max(np.abs(out3))/5
mlab.pipeline.image_plane_widget(grid, plane_orientation='x_axes', slice_index=len(x1)//2, colormap='coolwarm', vmin=-lim, vmax=lim)
mlab.pipeline.image_plane_widget(grid, plane_orientation='y_axes', slice_index=len(x2)//2, colormap='coolwarm', vmin=-lim, vmax=lim)
mlab.pipeline.image_plane_widget(grid, plane_orientation='z_axes', slice_index=len(x3)//2, colormap='coolwarm', vmin=-lim, vmax=lim)
mlab.outline(color=(0, 0, 0))


# Koper model
def Koper(Y, eps=0.01, k=-10.0, lam=-7.0):
  xd = (Y[1]-Y[0]**3+3*Y[0])
  yd = eps*(k*Y[0]-2*(Y[1]+lam)+Y[2])
  zd = eps*(lam+Y[1]-Y[2])
  return np.array([xd, yd, zd])

def KoperODE(Y, t):
  return Koper(Y)

Ykop = scint.odeint(KoperODE, np.array([0.5, 0.5, -8.0]), np.linspace(0, 1000, 100000))

ds = 0.04
x1, x2, x3, out1, out2 = repulsionRate3(Koper, [-3, 3], [-3, 3], [-10, -6], ds)
out3 = []
for outs in zip(out1.ravel(), out2.ravel()):
  out3.append(outs[np.argmax(np.abs(outs))])
out3 = np.array(out3).reshape(out1.shape)
grid = mlab.pipeline.scalar_field(out3)
grid.spacing=[ds, ds, ds]
lim = np.max(np.abs(out3))/5
mlab.pipeline.image_plane_widget(grid, plane_orientation='x_axes', slice_index=len(x1)//2, colormap='coolwarm', vmin=-lim, vmax=lim)
mlab.pipeline.image_plane_widget(grid, plane_orientation='y_axes', slice_index=len(x2)//2, colormap='coolwarm', vmin=-lim, vmax=lim)
mlab.pipeline.image_plane_widget(grid, plane_orientation='z_axes', slice_index=len(x3)//2, colormap='coolwarm', vmin=-lim, vmax=lim)
mlab.outline(color=(0, 0, 0))
mlab.view(azimuth=120, elevation=135, roll=105, distance=18, focalpoint='auto')
imgmap_RGBA = mlab.screenshot(mode='rgba', antialiased=True)
img_RGBA = Image.fromarray(np.array(imgmap_RGBA*255, dtype=np.uint8))
img_RGBA.save('../Figures/Koper-iso.png')
mlab.show()

# mlab.plot3d(Ykop[13000:, 0]+4, Ykop[13000:, 1]+4, Ykop[13000:, 2]+11, color=(0.0, 0.0, 0.0))
# mlab.show()
### Note regarding the offsets in mlab.plot3d:
# Mayavi allows you to update spacing when using a scalar_field, but still begins the region
# at the value [1, 1, 1]. Therefore, I've added to the integrated results
# until they reach the point where the corner of the scalar field [-3, -3, -10]
# becomes [1, 1, 1]

def resampleTrajectory3d(Xin, Yin, Zin, n=10):
  ds = np.sqrt((Xin[:-1]-Xin[1:])**2+(Yin[:-1]-Yin[1:])**2+(Zin[:-1]-Zin[1:])**2)
  ds = np.concatenate((np.array([0,]), ds))
  Xin = np.concatenate((np.array([Xin[0],]), Xin[ds>10e-8]))
  Yin = np.concatenate((np.array([Yin[0],]), Yin[ds>10e-8]))
  Zin = np.concatenate((np.array([Zin[0],]), Zin[ds>10e-8]))
  ds = np.concatenate((np.array([0,]), ds[ds>10e-8]))
  s = np.cumsum(ds)
  splX = interp1d(s, Xin, kind='cubic')
  splY = interp1d(s, Yin, kind='cubic')
  splZ = interp1d(s, Zin, kind='cubic')
  sNew = np.linspace(0, s[-2], n)
  return sNew, splX(sNew), splY(sNew), splZ(sNew)

rhofunction = RegularGridInterpolator((x1, x2, x3), out3, method='linear')
sn, xn, yn, zn = resampleTrajectory3d(Ykop[:, 0], Ykop[:, 1], Ykop[:, 2], n=501)
rhoDotn = [rhofunction(np.array([[xstep, ystep, zstep]])) for xstep, ystep,zstep in zip(xn, yn, zn)]


plt.figure(figsize=(3, 2))
plt.plot(sn, rhoDotn, linewidth=0.8)
# plt.xlabel('Arc Length', **labelfont)
# plt.ylabel(r'$\dot{\rho}$', **labelfont)
# plt.xticks([0, 10, 20, 30, 40], **tickfont)
# plt.yticks([-300, -150, 0], **tickfont)
# plt.xlim([0, 45])
# plt.ylim([-325, 25])
plt.savefig('../Figures/repulsion-values-3d.eps', transparent=True, bbox_inches='tight')

plt.figure(figsize=(3, 2))
plt.plot(, linewidth=0.8)
plt.xlabel('Arc Length', **labelfont)
plt.ylabel(r'$\dot{\rho}$', **labelfont)
plt.xticks([0, 10, 20, 30, 40], **tickfont)
plt.yticks([-300, -150, 0], **tickfont)
plt.xlim([0, 45])
plt.ylim([-325, 25])
# plt.savefig('../Figures/repulsion-values.eps', transparent=True, bbox_inches='tight')

# Rho Dot field for van der Pol
plt.figure(figsize=(3.2, 2.1))
# x1, x2, rhodot = mid.repulsion_rate(vanderPol, [-1.25, 1.25], [-1, 1], 0.05, newfig=False, plot=False, output=True)
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


#Lorenz system
def lorenz(Y, sig=10, rho=28, beta=8/3):
  xd = sig*(Y[1]-Y[0])
  yd = rho*Y[0]-Y[1]-Y[0]*Y[2]
  zd = Y[0]*Y[1]-beta*Y[2]
  return np.array([xd, yd, zd])

def lorenzODE(Y, t):
  return lorenz(Y)

Ylor = scint.odeint(lorenzODE, np.array([0.1, 0.1, 0.1]), np.linspace(0, 50, 10000))

ds = 0.1
x1, x2, x3, out3, out4 = repulsionRate3(lorenz, [-25, 25], [-25, 25], [0.1, 50], ds)
grid = mlab.pipeline.scalar_field(out2)
grid.spacing=[ds, ds, ds]
contours = mlab.pipeline.contour_surface(grid, contours=[-15,0,15], transparent=True)
mlab.plot3d(Ylor[:, 0]+25, Ylor[:, 1]+25, Ylor[:, 2], color=(0.0, 0.0, 0.0))

def abcFlow(Y, A=1, B=np.sqrt(2.0/3), C=np.sqrt(1.0/3)):
  xd = A*np.sin(Y[2])+C*np.cos(Y[1])
  yd = B*np.sin(Y[0])+A*np.cos(Y[2])
  zd = C*np.sin(Y[1])+B*np.cos(Y[0])
  return np.array([xd, yd, zd])

def abcODE(Y, t):
  return abcFlow(Y)

Yabc = scint.odeint(abcODE, np.array([0.1, 0.1, 0.1]), np.linspace(0, 50, 10000))

ds = 0.2
x1, x2, x3, out3 = repulsionRate3(abcFlow, [0, 2*np.pi], [0, 2*np.pi], [0, 2*np.pi], ds)
grid = mlab.pipeline.scalar_field(out3)
grid.spacing=[ds, ds, ds]
contours = mlab.pipeline.contour_surface(grid, contours=[-0.7,0,0.7], transparent=True)
mlab.plot3d(Yabc[:, 0]%(2*np.pi), Yabc[:, 1]%(2*np.pi), Yabc[:, 2]%(2*np.pi), color=(0.0, 0.0, 0.0))
mlab.show()

# Added the + 25 into plot3d because contour3d wasn't working with X1, X2, X3 inputs.
# contour3d defaults to the range [0:len(out2):spacing], with a spacing of ds,
# it will map to a size of (50, 50, 50), so we move the origin of the trajectory
# to the center of X1, X2, X3 by adding 25 to X1 and X2

  # kappa[n, m] = np.sqrt(np.dot(tprime, tprime))
  # lim = np.max(np.abs(rho_dot))
  # ax = plt.gca()
  # mesh = ax.pcolormesh(X1, X2, rho_dot, cmap=cmap, vmin=vmin, vmax=vmax)
  # clb = plt.colorbar(mesh)
  # clb.ax.set_title('$\\dot{\\rho}$', fontsize=28, y=1.02)
  # plt.xlim(xlims); plt.ylim(ylims)
  # return x1, x2, x3, out


# def curvatureField(func, xlims, ylims, ds, plot=True, cmap='PRGn', newfig=True):
#   if plot and newfig:
#     goodfigure(xlims, ylims)
#   x1 = np.arange(xlims[0], xlims[1]+ds, ds)
#   x2 = np.arange(ylims[0], ylims[1]+ds, ds)
#   X1, X2 = np.meshgrid(x1, x2)
#   U = np.zeros(np.shape(X1))
#   V = np.zeros(np.shape(U))
#   for m in range(len(x1)):
#     for n in range(len(x2)):
#       y0 = np.array([X1[n, m], X2[n, m]])
#       U[n, m], V[n, m] = func(y0, *fargs)
#   [DUy, DUx] = np.gradient(U[:, :], ds, edge_order=2)
#   [DVy, DVx] = np.gradient(V[:, :], ds, edge_order=2)
#   kappa = np.zeros(np.shape(U))
#   for m in range(len(x1)):
#     for n in range(len(x2)):
#       Utemp = np.array([U[n, m], V[n, m]])
#       Uhat = np.array([V[n, m], -U[n, m]])
#       Grad = np.array([[DUx[n, m], DUy[n, m]], [DVx[n, m], DVy[n, m]]])
#       kappa[n, m] = np.dot(Uhat, np.dot(Grad, Utemp))/np.dot(Utemp, Utemp)**(1.5)
#   if plot:
#     lim = np.max(np.abs(kappa))
#     ax = plt.gca()
#     mesh = ax.pcolormesh(X1, X2, kappa, cmap=cmap, vmin=-3, vmax=3)
#     clb = plt.colorbar(mesh)
#     clb.ax.set_title('$\\kappa$', fontsize=28, y=1.02)
#     plt.xlim(xlims); plt.ylim(ylims)
#   return x1, x2, kappa