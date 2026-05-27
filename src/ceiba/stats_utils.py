"""
stats_utils.py
--------------
Statistical testing utilities for multiplexed IF and spatial transcriptomics
analysis. Provides reusable functions for non-parametric group comparisons
with multiple testing correction.

Typical usage
-------------
from ceiba.stats_utils import run_stats

stats_df, pval_map = run_stats(normalized, cell_cols)
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


def run_stats(normalized, cell_cols,
              pos_label='class II high',
              neg_label='class II low',
              classification_col='patient classification'):
    """
    Run Mann-Whitney U tests with Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    normalized : pd.DataFrame
        Normalized counts with a patient classification column.
    cell_cols : list of str
        Cell type columns to test.
    pos_label : str
        Label for the positive/high group in classification_col.
    neg_label : str
        Label for the negative/low group in classification_col.
    classification_col : str
        Column identifying patient group.

    Returns
    -------
    stats_df : pd.DataFrame
        Results with raw and FDR-adjusted p-values, medians, and
        direction of effect.
    pval_map : dict
        FDR-adjusted p-values keyed by cell type column name.
    """
    rows = []
    for col in cell_cols:
        high = normalized.loc[normalized[classification_col] == pos_label, col].dropna()
        low  = normalized.loc[normalized[classification_col] == neg_label, col].dropna()
        if len(high) > 0 and len(low) > 0:
            _, p = mannwhitneyu(high, low, alternative='two-sided')
            rows.append({
                'Cell Type':    col,
                'P-value':      p,
                'Median (high)': high.median(),
                'Median (low)':  low.median(),
                'Higher group': pos_label if high.median() > low.median() else neg_label,
            })

    stats_df = pd.DataFrame(rows)
    _, fdr, _, _ = multipletests(stats_df['P-value'].values, method='fdr_bh')
    stats_df['FDR-adjusted P-value'] = fdr
    stats_df['Significant (FDR < 0.05)'] = fdr < 0.05
    stats_df = stats_df.sort_values('FDR-adjusted P-value')
    pval_map = stats_df.set_index('Cell Type')['FDR-adjusted P-value'].to_dict()
    return stats_df, pval_map



def run_mixed_effects(normalized, cell_cols,
                      pos_label='class II high',
                      neg_label='class II low',
                      classification_col='patient classification',
                      patient_col='PatientID'):
    """
    Linear mixed effects model for ROI-level density data.
    Log10-transformed density is modeled as a function of MHC II
    classification with patient as a random intercept, accounting
    for non-independence of ROIs within patients.

    Parameters
    ----------
    normalized : pd.DataFrame
        ROI-level normalized data with classification and patient columns.
    cell_cols : list of str
        Cell type columns to test.
    pos_label : str
        Label for the positive/high group in classification_col.
    neg_label : str
        Label for the negative/low group in classification_col.
    classification_col : str
        Column identifying patient group.
    patient_col : str
        Column identifying patient for random intercept.

    Returns
    -------
    results_df : pd.DataFrame
        Fixed effect estimates, standard errors, z-scores, and
        FDR-adjusted p-values for the classification term.
    """
    rows = []
    for col in cell_cols:
        df = normalized[[patient_col, classification_col, col]].dropna()
        df = df[df[col] > 0].copy()
        df['log_density'] = np.log10(df[col])
        df['group'] = (df[classification_col] == pos_label).astype(int)

        try:
            model = smf.mixedlm(
                'log_density ~ group',
                data=df,
                groups=df[patient_col]
            ).fit(reml=True)

            rows.append({
                'Cell Type':   col,
                'Coefficient': model.fe_params['group'],
                'Std Error':   model.bse_fe['group'],
                'Z-score':     model.tvalues['group'],
                'P-value':     model.pvalues['group'],
            })
        except Exception as e:
            print(f"Model failed for {col}: {e}")

    results_df = pd.DataFrame(rows)
    _, fdr, _, _ = multipletests(results_df['P-value'].values, method='fdr_bh')
    results_df['FDR-adjusted P-value'] = fdr
    results_df['Significant (FDR < 0.05)'] = fdr < 0.05
    results_df = results_df.sort_values('FDR-adjusted P-value')
    return results_df