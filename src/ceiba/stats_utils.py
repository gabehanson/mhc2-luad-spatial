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
import scipy.sparse as sp
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning


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

def ciita_expr_by_s100p_strata_per_sample(
    adata,
    origin_values=('normal_adjacent', 'tumor_primary'),
    detection_thresh=0.0,
    s100p_thresh=0.0,
    agg='mean_detected_only',
):
    """
    Compute mean CIITA expression in S100P+ vs S100P- epithelial cells
    per sample, stratified by origin.

    Designed to test whether S100P status predicts CIITA expression level
    within the same sample. Only CIITA-expressing cells are included when
    agg='mean_detected_only', which avoids dropout inflation and tests
    expression magnitude rather than detection frequency.

    Parameters
    ----------
    adata : AnnData
        Subset to paired LUAD epithelial cells.
    origin_values : tuple of str
        Origins to include (default: normal_adjacent and tumor_primary).
    detection_thresh : float
        CIITA expression threshold for defining CIITA+ cells (default 0.0).
    s100p_thresh : float
        S100P expression threshold for defining S100P+ cells (default 0.0).
    agg : str
        'mean_all' — mean across all cells.
        'mean_detected_only' — mean among CIITA+ cells only (recommended).

    Returns
    -------
    wide : pd.DataFrame
        One row per origin × donor × sample with columns:
        CIITA_mean_S100Pneg, CIITA_mean_S100Ppos, delta_pos_minus_neg.
    """
    epi = adata[adata.obs['ann_coarse'] == 'Epithelial cell'].copy()
    epi.obs['origin'] = epi.obs['origin'].astype(str).str.lower().str.strip()
    epi = epi[epi.obs['origin'].isin(origin_values)].copy()

    ciita_id = epi.var.loc[epi.var['feature_name'] == 'CIITA'].index[0]
    s100p_id = epi.var.loc[epi.var['feature_name'] == 'S100P'].index[0]

    Xc = epi[:, ciita_id].X
    Xs = epi[:, s100p_id].X
    ciita = Xc.toarray().ravel() if sp.issparse(Xc) else np.asarray(Xc).ravel()
    s100p = Xs.toarray().ravel() if sp.issparse(Xs) else np.asarray(Xs).ravel()

    df = pd.DataFrame({
        'donor_id': epi.obs['donor_id'].astype(str).values,
        'sample':   epi.obs['sample'].astype(str).values,
        'origin':   epi.obs['origin'].values,
        'CIITA':    ciita,
        'S100P':    s100p,
    })

    df['S100P_pos'] = df['S100P'] > s100p_thresh
    df['CIITA_pos'] = df['CIITA'] > detection_thresh

    if agg == 'mean_all':
        df['CIITA_val'] = df['CIITA']
        df_use = df
    elif agg == 'mean_detected_only':
        df_use = df[df['CIITA_pos']].copy()
        df_use['CIITA_val'] = df_use['CIITA']
    else:
        raise ValueError("agg must be 'mean_all' or 'mean_detected_only'")

    g = (
        df_use
        .groupby(['origin', 'donor_id', 'sample', 'S100P_pos'], observed=True)['CIITA_val']
        .mean()
        .reset_index()
    )

    wide = g.pivot_table(
        index=['origin', 'donor_id', 'sample'],
        columns='S100P_pos',
        values='CIITA_val',
        aggfunc='mean',
    ).reset_index()

    wide = wide.rename(columns={
        False: 'CIITA_mean_S100Pneg',
        True:  'CIITA_mean_S100Ppos',
    })
    wide['delta_pos_minus_neg'] = (
        wide['CIITA_mean_S100Ppos'] - wide['CIITA_mean_S100Pneg']
    )
    return wide


def ciita_expr_cell_level_tests(
    adata,
    detection_thresh=0.0,
    s100p_thresh=0.0,
    restrict_ciita_pos=True,
):
    """
    Test whether CIITA expression magnitude differs between S100P+ and S100P-
    epithelial cells at the single-cell level, separately per origin.

    Complements the within-sample delta analysis by testing expression magnitude
    rather than detection frequency. Among cells that express CIITA, are S100P+
    cells expressing less of it?

    Parameters
    ----------
    adata : AnnData
        Subset to paired LUAD epithelial cells.
    detection_thresh : float
        CIITA expression threshold for defining CIITA+ cells (default 0.0).
    s100p_thresh : float
        S100P expression threshold for defining S100P+ cells (default 0.0).
    restrict_ciita_pos : bool
        If True, restrict to CIITA+ cells before testing (recommended).

    Returns
    -------
    pd.DataFrame
        One row per origin with columns: n_S100Ppos, n_S100Pneg,
        median_CIITA_S100Ppos, median_CIITA_S100Pneg, p (Mann-Whitney U).
    """
    epi = adata[adata.obs['ann_coarse'] == 'Epithelial cell'].copy()
    epi.obs['origin'] = epi.obs['origin'].astype(str).str.lower().str.strip()

    ciita_id = epi.var.loc[epi.var['feature_name'] == 'CIITA'].index[0]
    s100p_id = epi.var.loc[epi.var['feature_name'] == 'S100P'].index[0]

    Xc = epi[:, ciita_id].X
    Xs = epi[:, s100p_id].X
    ciita = Xc.toarray().ravel() if sp.issparse(Xc) else np.asarray(Xc).ravel()
    s100p = Xs.toarray().ravel() if sp.issparse(Xs) else np.asarray(Xs).ravel()

    df = pd.DataFrame({
        'origin': epi.obs['origin'].values,
        'CIITA':  ciita,
        'S100P':  s100p,
    })
    df['S100P_pos'] = df['S100P'] > s100p_thresh
    df['CIITA_pos'] = df['CIITA'] > detection_thresh

    if restrict_ciita_pos:
        df = df[df['CIITA_pos']].copy()

    out = []
    for origin in ['normal_adjacent', 'tumor_primary']:
        sub = df[df['origin'] == origin]
        a = sub.loc[ sub['S100P_pos'], 'CIITA']
        b = sub.loc[~sub['S100P_pos'], 'CIITA']
        if len(a) == 0 or len(b) == 0:
            out.append({
                'origin': origin, 'n_S100Ppos': len(a),
                'n_S100Pneg': len(b), 'p': np.nan,
            })
            continue
        _, p = mannwhitneyu(a, b, alternative='two-sided')
        out.append({
            'origin':                origin,
            'n_S100Ppos':            len(a),
            'n_S100Pneg':            len(b),
            'median_CIITA_S100Ppos': float(np.median(a)),
            'median_CIITA_S100Pneg': float(np.median(b)),
            'p':                     p,
        })
    return pd.DataFrame(out)