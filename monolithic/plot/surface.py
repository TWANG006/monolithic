"""Provide the convenient functions for surface-related plots."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..math import pv, rmse


def show_surface_map(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    ax=None,
    coord_unit: float = 1e3,
    coord_str: str = 'mm',
    error_unit: float = 1e9,
    error_str: str = 'nm',
    vmin: float = None,
    vmax: float = None,
    colormap: str = 'viridis',
    title: str = 'Surface',
):
    """Display the surface (height or slope) map.

    Args:
        X (numpy.ndarray): x coordinates.
        Y (numpy.ndarray): y coordinates.
        Z (numpy.ndarray): surface errors.
        ax (matplotlib.axes): the axes where the surface will be drawn. Create a new figure if it is None.
        coord_unit (float): the unit scale factor to be multiplied to X and Y.
        coord_str (str): the coordinate unit to be displayed.
        error_unit (float): the unit to be multiplied to the errors.
        error_str (str): the error unit to be displayed.
        vmin (float): minimum of the colorbar.
        vmax (flaot): maximum of the colorbar.
        colormap (str): colormap for the surface error map.
        title (str): title for the plot.
    """
    # scale to the expected units
    Xmm = X * coord_unit
    Ymm = Y * coord_unit
    Znm = Z * error_unit

    # calculate the pv and rms
    rms_z = rmse(Znm)
    pv_z = pv(Znm)

    # create an ax if not provided
    if ax is None:
        _, ax = plt.subplots()

    # plot the surface error map
    c = ax.pcolormesh(Xmm, Ymm, Znm, cmap=colormap, shading='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title + f': PV = {pv_z:.2f} {error_str}, RMS = {rms_z:.2f} {error_str}', wrap=True)
    ax.set_xlabel(f'x [{coord_str}]')
    ax.set_ylabel(f'y [{coord_str}]')
    ax.set_aspect('equal')
    ax.set_rasterized(True)

    # plot colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(c, ax=ax, cax=cax)
    cbar.set_label(f'[{error_str}]')
