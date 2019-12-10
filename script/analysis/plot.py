################################################################################
#                                                                              #
#  UTILITIES FOR PLOTTING                                                      #
#                                                                              #
################################################################################

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from scipy.integrate import trapz

from analysis_fns import *

# Get xz slice of 3D data
def flatten_xz(array, patch_pole=False, average=False):
  if array.ndim == 2:
    N1 = array.shape[0]
    N2 = array.shape[1]
    flat = np.zeros([2*N1,N2])
    for i in range(N1):
      flat[i,:] = array[N1 - 1 - i,:]
      flat[i+N1,:] = array[i,:]
    return flat
  
  N1 = array.shape[0]; N2 = array.shape[1]; N3 = array.shape[2]
  flat = np.zeros([2*N1,N2])
  if average:
    for i in range(N1):
      # Produce identical hemispheres to get the right size output
      flat[i,:] = np.mean(array[N1 - 1 - i,:,:], axis=-1)
      flat[i+N1,:] = np.mean(array[i,:,:], axis=-1)
  else:
    for i in range(N1):
      flat[i,:] = array[N1 - 1 - i,:,N3//2]
      flat[i+N1,:] = array[i,:,0]

  # Theta is technically [small,pi/2-small]
  # This patches the X coord so the plot looks nice
  if patch_pole:
      flat[:,0] = 0
      flat[:,-1] = 0

  return flat

# Get xy slice of 3D data
def flatten_xy(array, average=False, loop=True):
  if array.ndim == 2:
    return array
  
  if average:
    slice = np.mean(array, axis=1)
  else:
    slice = array[:,array.shape[1]//2,:]
  
  if loop:
    return loop_phi(slice)
  else:
    return slice

def loop_phi(array):
  return np.vstack((array.transpose(),array.transpose()[0])).transpose()

# Plotting fns: pass dump file and var as either string (key) or ndarray
# Note integrate option overrides average
# Also note label convention:
# * "known labels" are assigned true or false,
# * "unknown labels" are assigned None or a string
# TODO pass through kwargs instead of all this duplication
def plot_xz(ax, geom, var, cmap='jet', vmin=None, vmax=None, window=[-40,40,-40,40],
            cbar=True, cbar_ticks=None, label=None, xlabel=True, ylabel=True, xticks=True, yticks=True,
            arrayspace=False, average=False, integrate=False, bh=True, half_cut=False, shading='gouraud'):

  if integrate:
    var *= geom['n3']
    average = True

  if (arrayspace):
    x1_norm = (geom['X1'] - geom['startx1']) / (geom['n1'] * geom['dx1'])
    x2_norm = (geom['X2'] - geom['startx2']) / (geom['n2'] * geom['dx2'])
    x = flatten_xz(x1_norm)[geom['n1']:,:]
    z = flatten_xz(x2_norm)[geom['n1']:,:]
    if geom['n3'] > 1:
      var = flatten_xz(var, average=average)[geom['n1']:,:]
    else:
      var = var[:,:,0]
  else:
    if half_cut:
      x = flatten_xz(geom['x'], patch_pole=True)[geom['n1']:,:]
      z = flatten_xz(geom['z'])[geom['n1']:,:]
      var = flatten_xz(var, average=average)[geom['n1']:,:]
      window[0] = 0
    else:
      x = flatten_xz(geom['x'], patch_pole=True)
      z = flatten_xz(geom['z'])
      var = flatten_xz(var, average=average)

  #print 'xshape is ', x.shape, ', zshape is ', z.shape, ', varshape is ', var.shape
  mesh = ax.pcolormesh(x, z, var, cmap=cmap, vmin=vmin, vmax=vmax,
      shading=shading)

  if arrayspace:
    if xlabel: ax.set_xlabel("X1 (arbitrary)")
    if ylabel: ax.set_ylabel("X2 (arbitrary)")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
  else:
    if xlabel: ax.set_xlabel(r"$x \frac{c^2}{G M}$")
    if ylabel: ax.set_ylabel(r"$z \frac{c^2}{G M}$")
    if window:
      ax.set_xlim(window[:2]); ax.set_ylim(window[2:])

    if bh:
      # BH silhouette
      circle1=plt.Circle((0,0), geom['r_eh'], color='k');
      ax.add_artist(circle1)

  if not half_cut:
    ax.set_aspect('equal')

  if cbar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mesh, cax=cax, ticks=cbar_ticks)

  if not xticks:
    plt.gca().set_xticks([])
    plt.xticks([])
    ax.set_xticks([])
  if not yticks:
    plt.gca().set_yticks([])
    plt.yticks([])
    ax.set_yticks([])
  if not xticks and not yticks:
    # Remove the whole frame for good measure
    #fig.patch.set_visible(False)
    ax.axis('off')


  if label is not None:
    ax.set_title(label)

def plot_xy(ax, geom, var, cmap='jet', vmin=None, vmax=None, window=[-40,40,-40,40],
            cbar=True, label=None, xlabel=True, ylabel=True, xticks=True, yticks=True,
            ticks=None, arrayspace=False, average=False, integrate=False, bh=True, shading='gouraud'):

  if integrate:
    var *= geom['n2']
    average = True
  
  if arrayspace:
    # Flatten_xy adds a rank. TODO is this the way to handle it?
    x1_norm = (geom['X1'] - geom['startx1']) / (geom['n1'] * geom['dx1'])
    x3_norm = (geom['X3'] - geom['startx3']) / (geom['n3'] * geom['dx3'])
    x = flatten_xy(x1_norm, loop=False)
    y = flatten_xy(x3_norm, loop=False)
    var = flatten_xy(var, average=average, loop=False)
  else:
    x = flatten_xy(geom['x'])
    y = flatten_xy(geom['y'])
    var = flatten_xy(var, average=average)

  #print 'xshape is ', x.shape, ', yshape is ', y.shape, ', varshape is ', var.shape
  mesh = ax.pcolormesh(x, y, var, cmap=cmap, vmin=vmin, vmax=vmax,
      shading=shading)

  if arrayspace:
    if xlabel: ax.set_xlabel("X1 (arbitrary)")
    if ylabel: ax.set_ylabel("X3 (arbitrary)")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
  else:
    if xlabel: ax.set_xlabel(r"$x \frac{c^2}{G M}$")
    if ylabel: ax.set_ylabel(r"$y \frac{c^2}{G M}$")
    if window:
      ax.set_xlim(window[:2]); ax.set_ylim(window[2:])

    if bh:
      # BH silhouette
      circle1=plt.Circle((0,0), geom['r_eh'], color='k');
      ax.add_artist(circle1)

  ax.set_aspect('equal')

  if cbar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mesh, cax=cax, ticks=ticks)

  if not xticks:
    plt.gca().set_xticks([])   
    plt.xticks([])
    ax.set_xticks([])
  if not yticks:  
    plt.gca().set_yticks([])
    plt.yticks([])
    ax.set_yticks([])
  if not xticks and not yticks:
    # Remove the whole frame for good measure
    #fig.patch.set_visible(False)
    ax.axis('off')

  if label:
    ax.set_title(label)

# TODO this is currently just for profiles already in 2D
def plot_thphi(ax, geom, var, r_i, cmap='jet', vmin=None, vmax=None, window=None,
            cbar=True, label=None, xlabel=True, ylabel=True, arrayspace=False,
            ticks=None, project=False, shading='gouraud'):

  if arrayspace:
    # X3-X2 makes way more sense than X2-X3 since the disk is horizontal
    x = (geom['X3'][r_i] - geom['startx3']) / (geom['n3'] * geom['dx3'])
    y = (geom['X2'][r_i] - geom['startx2']) / (geom['n2'] * geom['dx2'])
  else:
    radius = geom['r'][r_i,0,0]
    max_th = geom['n2']//2
    if project:
      x = loop_phi((geom['th']*np.cos(geom['phi']))[r_i,:max_th,:])
      y = loop_phi((geom['th']*np.sin(geom['phi']))[r_i,:max_th,:])
    else:
      x = loop_phi(geom['x'][r_i,:max_th,:])
      y = loop_phi(geom['y'][r_i,:max_th,:])
    var = loop_phi(var[:max_th,:])

  if window is None:
    if arrayspace:
      ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    elif project:
      window = [-1.6, 1.6, -1.6, 1.6]
    else:
      window = [-radius, radius, -radius, radius]
  else:
    ax.set_xlim(window[:2]); ax.set_ylim(window[2:])

  #print 'xshape is ', x.shape, ', yshape is ', y.shape, ', varshape is ', var.shape
  mesh = ax.pcolormesh(x, y, var, cmap=cmap, vmin=vmin, vmax=vmax,
      shading=shading)

  if arrayspace:
    if xlabel: ax.set_xlabel("X3 (arbitrary)")
    if ylabel: ax.set_ylabel("X2 (arbitrary)")
  else:
    if xlabel: ax.set_xlabel(r"$x \frac{c^2}{G M}$")
    if ylabel: ax.set_ylabel(r"$y \frac{c^2}{G M}$")

  ax.set_aspect('equal')

  if cbar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mesh, cax=cax, ticks=ticks)

  if label:
    ax.set_title(label)

def overlay_contours(ax, geom, var, levels, color='k'):
  x = flatten_xz(geom['x'])
  z = flatten_xz(geom['z'])
  var = flatten_xz(var, average=True)
  return ax.contour(x, z, var, levels=levels, colors=color)

def overlay_field(ax, geom, dump, **kwargs):
  overlay_flowlines(ax, geom, dump['B1'], dump['B2'], **kwargs)

def overlay_flowlines(ax, geom, varx1, varx2, nlines=50, arrayspace=False, reverse=False):
  N1 = geom['n1']; N2 = geom['n2']
  if arrayspace:
    x1_norm = (geom['X1'] - geom['startx1']) / (geom['n1'] * geom['dx1'])
    x2_norm = (geom['X2'] - geom['startx2']) / (geom['n2'] * geom['dx2'])
    x = flatten_xz(x1_norm)[geom['n1']:,:]
    z = flatten_xz(x2_norm)[geom['n1']:,:]
  else:
    x = flatten_xz(geom['x'])
    z = flatten_xz(geom['z'])

  varx1 = varx1.mean(axis=-1)
  varx2 = varx2.mean(axis=-1)
  AJ_phi = np.zeros([2*N1, N2])
  gdet = geom['gdet']
  for j in range(N2):
    for i in range(N1):
      if not reverse:
        AJ_phi[N1-1-i,j] = AJ_phi[i+N1,j] = (
          trapz(gdet[:i,j]*varx2[:i,j], dx=geom['dx1']) -
          trapz(gdet[i,:j]*varx1[i,:j], dx=geom['dx2']))
      else:
        AJ_phi[N1-1-i,j] = AJ_phi[i+N1,j] = (
          trapz(gdet[:i,j]*varx2[:i,j], dx=geom['dx1']) +
          trapz(gdet[i,j:]*varx1[i,j:], dx=geom['dx2']))
  AJ_phi -= AJ_phi.min()
  levels = np.linspace(0,AJ_phi.max(),nlines*2)

  if arrayspace:
    ax.contour(x, z, AJ_phi[N1:,:], levels=levels, colors='k')
  else:
    ax.contour(x, z, AJ_phi, levels=levels, colors='k')

def overlay_quiver(ax, geom, dump, JE1, JE2, cadence=64, norm=1):
  JE1 *= geom['gdet']
  JE2 *= geom['gdet']
  max_J = np.max(np.sqrt(JE1**2 + JE2**2))
  x1_norm = (geom['X1'] - geom['startx1']) / (geom['n1'] * geom['dx1'])
  x2_norm = (geom['X2'] - geom['startx2']) / (geom['n2'] * geom['dx2'])
  x = flatten_xz(x1_norm)[geom['n1']:,:]
  z = flatten_xz(x2_norm)[geom['n1']:,:]
  
  s1 = geom['n1']//cadence; s2 = geom['n2']//cadence
  
  ax.quiver(x[::s1,::s2], z[::s1,::s2], JE1[::s1,::s2], JE2[::s1,::s2],
      units='xy', angles='xy', scale_units='xy', scale=cadence*max_J/norm)

# Plot two slices together without duplicating everything in the caller
def plot_slices(ax1, ax2, geom, dump, var, field_overlay=True, nlines=10, **kwargs):

  if 'arrspace' in list(kwargs.keys()):
    arrspace = kwargs['arrspace']
  else:
    arrspace = False
  
  plot_xz(ax1, geom, var, **kwargs)
  if field_overlay:
    overlay_field(ax1, geom, dump, nlines=nlines, arrayspace=arrspace)

  plot_xy(ax2, geom, var, **kwargs)

# TODO Consistent idea of plane/average in x2,x3
def radial_plot(ax, geom, var, n2=0, n3=0, average=False,
                logr=False, logy=False, rlim=None, ylim=None, arrayspace=False,
                ylabel=None, style='k-'):

  r = geom['r'][:, geom['n2']//2, 0]
  if var.ndim == 1:
    data = var
  elif var.ndim == 2:
    data = var[:,n2]
  elif var.ndim == 3:
    if average:
      data = np.mean(var[:,n2,:], axis=-1)
    else:
      data = var[:,n2,n3]

  if arrayspace:
    ax.plot(list(range(geom['n1'])), data, style)
  else:
    ax.plot(r,data, style)

  if logr: ax.set_xscale('log')
  if logy: ax.set_yscale('log')

  if rlim: ax.set_xlim(rlim)
  if ylim: ax.set_ylim(ylim)
  
  ax.set_xlabel(r"$r \frac{c^2}{G M}$")
  if ylabel is not None: ax.set_ylabel(ylabel)

def diag_plot(ax, diag, varname, t=0, ylabel=None, ylim=None, logy=False, xlabel=True, style='k-'):
  var = diag[varname]

  ax.plot(diag['t'], var, style)

  ax.set_xlim([diag['t'][0], diag['t'][-1]])

  # Trace current t on finished plot
  if t != 0:
    ax.axvline(t, color='r')

  if ylim is not None: ax.set_ylim(ylim)
  if logy: ax.set_yscale('log')

  if xlabel:
    ax.set_xlabel(r"$t \frac{c^3}{G M}$")
  if ylabel is not None:
    ax.set_ylabel(ylabel)
  else:
    ax.set_ylabel(varname)

def hist_2d(ax, var_x, var_y, xlbl, ylbl, title=None, logcolor=False, bins=40, cbar=True, cmap='jet', ticks=None):
  # Courtesy of George Wong
  var_x_flat = var_x.flatten()
  var_y_flat = var_y.flatten()
  nidx = np.isfinite(var_x_flat) & np.isfinite(var_y_flat)
  hist = np.histogram2d(var_x_flat[nidx], var_y_flat[nidx], bins=bins)
  X,Y = np.meshgrid(hist[1], hist[2])
  
  if logcolor:
    hist[0][hist[0] == 0] = np.min(hist[0][np.nonzero(hist[0])])
    mesh = ax.pcolormesh(X, Y, np.log10(hist[0]), cmap=cmap)
  else:
    mesh = ax.pcolormesh(X, Y, hist[0], cmap=cmap)

  # Add the patented Ben Ryan colorbar
  if cbar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mesh, cax=cax, ticks=ticks)

  if title is not None: ax.set_title(title)
  ax.set_xlabel(xlbl)
  ax.set_ylabel(ylbl)
