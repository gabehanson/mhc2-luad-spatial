"""
halo_utils.py
-------------
Utilities for loading and preprocessing HALO cell-level exports.

Covers coordinate computation, channel renaming, and patient-level
aggregation of phenotype fractions from TMA data.

Typical usage
-------------
from ceiba.halo_utils import compute_cell_centers, rename_channels, aggregate_to_patient

df = compute_cell_centers(df)
df = rename_channels(df, {'Cy5 635': 'S100P', 'Fitc 474': 'PanCK', 'Trtc 554': 'MHCII'})
patient_df = aggregate_to_patient(df, cell_types, region_col='Region', exclude_regions=['N'])
"""

import numpy as np
import pandas as pd


def compute_cell_centers(df, xmin='XMin', xmax='XMax', ymin='YMin', ymax='YMax'):
    """
    Compute cell centroid coordinates from bounding box columns.

    HALO exports bounding boxes per cell. This adds XCenter and YCenter
    columns as the midpoint of each bounding box.

    Parameters
    ----------
    df : pd.DataFrame
    xmin, xmax, ymin, ymax : str
        Names of the bounding box columns.

    Returns
    -------
    df : pd.DataFrame
        Input DataFrame with 'XCenter' and 'YCenter' columns added.
    """
    df = df.copy()
    df['XCenter'] = (df[xmax] + df[xmin]) / 2
    df['YCenter'] = (df[ymax] + df[ymin]) / 2
    return df


def rename_channels(df, channel_map):
    """
    Rename fluorescent channel identifiers in column names.

    HALO exports use raw fluorophore names (e.g. 'Cy5 635'). This replaces
    them with biological marker names across all column names in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    channel_map : dict
        Mapping of fluorophore string to marker name.
        e.g. {'Cy5 635': 'S100P', 'Fitc 474': 'PanCK', 'Trtc 554': 'MHCII'}

    Returns
    -------
    df : pd.DataFrame
        DataFrame with renamed columns.

    Example
    -------
    df = rename_channels(df, {
        'Cy5 635':  'S100P',
        'Fitc 474': 'PanCK',
        'Trtc 554': 'MHCII',
    })
    """
    df = df.copy()
    for old, new in channel_map.items():
        df.columns = df.columns.str.replace(old, new, regex=False)
    return df


def aggregate_to_patient(
    df,
    cell_type_cols,
    patient_col='PatientID',
    classification_col='patient classification',
    region_col='Region',
    total_col='Total Cells',
    exclude_regions=None,
):
    """
    Aggregate HALO cell phenotype counts to patient-level weighted fractions.

    Sums raw cell counts and total cells across all cores per patient before
    dividing, so that larger cores contribute proportionally more than small
    ones. NAT or other non-tumor cores can be excluded before aggregation.

    Parameters
    ----------
    df : pd.DataFrame
        Cell- or core-level DataFrame with phenotype count columns.
    cell_type_cols : list of str
        Column names for each phenotype count to aggregate
        (e.g. ['PanCK+S100P+MHCII+ Cells', ...]).
    patient_col : str
        Column identifying the patient.
    classification_col : str
        Column containing patient-level group labels.
    region_col : str
        Column identifying the tissue region (e.g. 'CT', 'PT', 'N').
    total_col : str
        Column containing total cell count per row.
    exclude_regions : list of str or None
        Region labels to exclude before aggregation (e.g. ['N'] to drop NAT).

    Returns
    -------
    patient_df : pd.DataFrame
        One row per patient with columns:
        - PatientID, patient classification
        - Total_Cells (sum across included cores)
        - One column per cell type (raw count sum)
        - One '{cell_type}_fraction' column per cell type
    """
    df = df.copy()

    if exclude_regions:
        df = df[~df[region_col].isin(exclude_regions)]

    agg_dict = {total_col: (total_col, 'sum')}
    for ct in cell_type_cols:
        agg_dict[ct] = (ct, 'sum')

    patient_df = (
        df.groupby([patient_col, classification_col])
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={total_col: 'Total_Cells'})
    )

    for ct in cell_type_cols:
        patient_df[f'{ct}_fraction'] = patient_df[ct] / patient_df['Total_Cells']

    return patient_df


def filter_low_cellularity(patient_df, min_cells, total_col='Total_Cells'):
    """
    Remove patients with fewer than min_cells total cells after aggregation.

    Low-cellularity patients produce unstable fraction estimates — a single
    positive cell in a core with 5 total cells gives a 20% fraction that
    is not comparable to a patient with 500 cells. This filter removes those
    patients before statistical comparisons.

    Parameters
    ----------
    patient_df : pd.DataFrame
        Output of aggregate_to_patient.
    min_cells : int
        Minimum number of total cells required to retain a patient.
    total_col : str
        Column containing total cell count.

    Returns
    -------
    filtered_df : pd.DataFrame
    n_removed : int
        Number of patients removed.
    """
    n_before = len(patient_df)
    filtered_df = patient_df[patient_df[total_col] >= min_cells].copy()
    n_removed = n_before - len(filtered_df)
    return filtered_df, n_removed