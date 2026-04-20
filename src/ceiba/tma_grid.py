"""
tma_grid.py
-----------
Utilities for assigning TMA core grid positions to HALO cell-level exports.

Given a DataFrame of cells with XCenter/YCenter coordinates, these functions
identify the row/column boundaries between cores and assign each cell a
Core_ID. Handles axis-aligned TMAs as well as rotated/misaligned grids via
PCA-based reorientation.

Typical usage
-------------
from ceiba.tma_grid import assign_grid_ids_unified, draw_safe_horizontal_lines, draw_safe_vertical_lines

df = assign_grid_ids_unified(
    df,
    horizontal_func=draw_safe_horizontal_lines,
    vertical_func=draw_safe_vertical_lines,
    clearance=100,
    min_spacing=3000,
    plot=True
)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def draw_safe_vertical_lines(
    df,
    y_col='YCenter',
    x_col='XCenter',
    clearance=100,
    min_spacing=3000,
    bins=500,
    smooth_sigma=3,
    method='clearance',
    plot=False
):
    """
    Find vertical divider lines (column boundaries) between TMA cores.

    Two methods are available:
    - 'clearance': scans candidate positions and keeps those with no cells
      within `clearance` pixels. Robust for well-separated cores.
    - 'density': finds valleys in a smoothed 1D histogram of cell positions.
      Better for cores with very narrow gaps.

    Parameters
    ----------
    df : pd.DataFrame
    y_col : str
        Column containing the vertical coordinate (default 'YCenter').
    x_col : str
        Column containing the horizontal coordinate (default 'XCenter').
    clearance : int
        Minimum distance from any cell for a candidate line (clearance mode).
    min_spacing : int
        Minimum distance between consecutive divider lines.
    bins : int
        Number of histogram bins (density mode only).
    smooth_sigma : float
        Gaussian smoothing sigma for density profile (density mode only).
    method : str
        'clearance' or 'density'.
    plot : bool
        If True, plot the dividers overlaid on cell positions.

    Returns
    -------
    safe_lines : list of float
        Y positions of vertical dividers.
    """
    y_vals = df[y_col].values
    safe_lines = []

    if method == 'clearance':
        y_min, y_max = y_vals.min(), y_vals.max()
        candidate_lines = np.arange(y_min, y_max, 50)
        for y_line in candidate_lines:
            if np.all(np.abs(y_vals - y_line) > clearance):
                if len(safe_lines) == 0 or (y_line - safe_lines[-1]) >= min_spacing:
                    safe_lines.append(y_line)

    elif method == 'density':
        hist, edges = np.histogram(y_vals, bins=bins)
        smoothed = gaussian_filter1d(hist, sigma=smooth_sigma)
        inverted = -smoothed
        peaks, _ = find_peaks(inverted, distance=min_spacing / (edges[1] - edges[0]))
        safe_lines = [edges[p] for p in peaks]

    else:
        raise ValueError("method must be 'clearance' or 'density'")

    if plot:
        fig, ax = plt.subplots(figsize=(10, 14))
        sns.scatterplot(data=df, x=y_col, y=x_col, s=2, alpha=0.4, ax=ax)
        for y_line in safe_lines:
            ax.axvline(x=y_line, color='red', linestyle='--')
        ax.invert_yaxis()
        ax.set_title(f"Vertical lines ({method} mode)")
        plt.tight_layout()
        plt.show()

    return safe_lines


def draw_safe_horizontal_lines(
    df,
    x_col='XCenter',
    y_col='YCenter',
    clearance=100,
    min_spacing=3000,
    plot=False
):
    """
    Find horizontal divider lines (row boundaries) between TMA cores.

    Uses the clearance method: scans candidate positions along the x-axis
    and keeps those with no cells within `clearance` pixels.

    Parameters
    ----------
    df : pd.DataFrame
    x_col : str
        Column containing the horizontal coordinate (default 'XCenter').
    y_col : str
        Column containing the vertical coordinate (default 'YCenter').
    clearance : int
        Minimum distance from any cell for a candidate line.
    min_spacing : int
        Minimum distance between consecutive divider lines.
    plot : bool
        If True, plot the dividers overlaid on cell positions.

    Returns
    -------
    safe_lines : list of float
        X positions of horizontal dividers.
    """
    x_vals = df[x_col].values
    x_min, x_max = x_vals.min(), x_vals.max()
    candidate_lines = np.arange(x_min, x_max, 50)
    safe_lines = []

    for x_line in candidate_lines:
        if np.all(np.abs(x_vals - x_line) > clearance):
            if len(safe_lines) == 0 or (x_line - safe_lines[-1]) >= min_spacing:
                safe_lines.append(x_line)

    if plot:
        plt.figure(figsize=(10, 14))
        sns.scatterplot(data=df, x=y_col, y=x_col, s=2, alpha=0.4)
        for x_line in safe_lines:
            plt.axhline(y=x_line, color='blue', linestyle='--', linewidth=1)
        plt.gca().invert_yaxis()
        plt.title(f"Horizontal lines >= {min_spacing} apart, clearance={clearance}")
        plt.show()

    return safe_lines


def assign_grid_ids_unified(
    df,
    x_col='XCenter',
    y_col='YCenter',
    clearance=100,
    min_spacing=3000,
    horizontal_func=None,
    vertical_func=None,
    rotate_horizontal=False,
    rotate_vertical=False,
    flip_x=False,
    flip_y=False,
    eps=300,
    min_samples=10,
    plot=False
):
    """
    Assign row, col, and Core_ID to each cell using a grid-based approach.

    For axis-aligned TMAs, boundary detection runs directly on XCenter/YCenter.
    For rotated/misaligned TMAs, PCA is used to reorient coordinates before
    boundary detection. flip_x and flip_y allow manual inversion of PCA axes
    if the orientation is inverted relative to the TMA map.

    Parameters
    ----------
    df : pd.DataFrame
        Cell-level DataFrame with coordinate columns.
    x_col : str
        Horizontal coordinate column (default 'XCenter').
    y_col : str
        Vertical coordinate column (default 'YCenter').
    clearance : int
        Passed to boundary detection functions.
    min_spacing : int
        Minimum spacing between grid lines.
    horizontal_func : callable
        Function to detect row boundaries (e.g. draw_safe_horizontal_lines).
    vertical_func : callable
        Function to detect column boundaries (e.g. draw_safe_vertical_lines).
    rotate_horizontal : bool
        If True, apply PCA rotation before detecting row boundaries.
    rotate_vertical : bool
        If True, apply PCA rotation before detecting column boundaries.
    flip_x : bool
        Invert PCA x-axis after rotation.
    flip_y : bool
        Invert PCA y-axis after rotation.
    eps : int
        Unused parameter reserved for future DBSCAN-based assignment.
    min_samples : int
        Unused parameter reserved for future DBSCAN-based assignment.
    plot : bool
        If True, plot grid dividers overlaid on cell positions.

    Returns
    -------
    df : pd.DataFrame
        Input DataFrame with 'row', 'col', 'Core_ID', and optionally
        'x_rot'/'y_rot' columns added.
    row_boundaries : list of float
    col_boundaries : list of float
    """
    df = df.copy()
    row_boundaries, col_boundaries = None, None

    if rotate_horizontal or rotate_vertical:
        coords = df[[x_col, y_col]].values
        pca = PCA(n_components=2)
        coords_rot = pca.fit_transform(coords)
        x_rot = coords_rot[:, 0]
        y_rot = coords_rot[:, 1]
        if flip_x:
            x_rot = -x_rot
        if flip_y:
            y_rot = -y_rot
        df['x_rot'] = x_rot
        df['y_rot'] = y_rot

    if rotate_horizontal:
        row_boundaries = horizontal_func(
            df, x_col='x_rot', y_col='y_rot',
            clearance=clearance, min_spacing=min_spacing
        )
        row_edges = [-np.inf] + row_boundaries + [np.inf]
        df['row'] = pd.cut(df['x_rot'], bins=row_edges, labels=False)
    else:
        row_boundaries = horizontal_func(
            df, x_col=x_col, y_col=y_col,
            clearance=clearance, min_spacing=min_spacing
        )
        row_edges = [-np.inf] + row_boundaries + [np.inf]
        df['row'] = pd.cut(df[x_col], bins=row_edges, labels=False)

    if rotate_vertical:
        col_boundaries = vertical_func(
            df, x_col='x_rot', y_col='y_rot',
            clearance=clearance, min_spacing=min_spacing
        )
        col_edges = [-np.inf] + col_boundaries + [np.inf]
        df['col'] = pd.cut(df['y_rot'], bins=col_edges, labels=False)
    else:
        col_boundaries = vertical_func(
            df, x_col=x_col, y_col=y_col,
            clearance=clearance, min_spacing=min_spacing
        )
        col_edges = [-np.inf] + col_boundaries + [np.inf]
        df['col'] = pd.cut(df[y_col], bins=col_edges, labels=False)

    df['Core_ID'] = 'Row_' + df['row'].astype(str) + '_Col_' + df['col'].astype(str)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 10))
        plot_x = 'x_rot' if (rotate_horizontal or rotate_vertical) else x_col
        plot_y = 'y_rot' if (rotate_horizontal or rotate_vertical) else y_col
        sns.scatterplot(data=df, x=plot_y, y=plot_x, s=2, alpha=0.4, ax=ax)
        if row_boundaries is not None:
            for x in row_boundaries:
                ax.axhline(y=x, color='blue', linestyle='--')
        if col_boundaries is not None:
            for y in col_boundaries:
                ax.axvline(x=y, color='red', linestyle='--')
        ax.invert_yaxis()
        ax.set_title(
            "Grid dividers (rotated view)"
            if (rotate_horizontal or rotate_vertical)
            else "Grid dividers"
        )
        plt.tight_layout()
        plt.show()

    return df, row_boundaries, col_boundaries