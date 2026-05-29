"""
plot_utils.py
-------------
Plotting utilities for multiplexed IF and TMA analysis figures.

Provides reusable figure components for comparing patient-level phenotype
fractions between groups. Designed for log-scale data with moderate n
(20-60 patients per group).

Typical usage
-------------
from ceiba.plot_utils import draw_boxstrip_panel, make_comparison_figure, sig_label

fig, axes = make_comparison_figure(
    plot_df,
    cell_types=['PanCK+S100P+MHCII- Cells', 'PanCK-S100P+MHCII- Cells'],
    neg_color='#462255',
    pos_color='#FF8811',
    outpath='figures/main_figure.pdf'
)
"""

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, wilcoxon


def sig_label(p):
    """
    Convert a p-value to a significance star string.

    Parameters
    ----------
    p : float

    Returns
    -------
    str : '***', '**', '*', or 'ns'
    """
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def draw_boxstrip_panel(
    ax,
    neg,
    pos,
    pval,
    neg_color='#462255',
    pos_color='#FF8811',
    neg_label='MHC II\nneg',
    pos_label='MHC II\npos',
    title=None,
    ylabel=None,
    jitter_width=0.12,
    point_size=18,
    point_alpha=0.8,
    box_width=0.35,
    title_fontsize=10,
    label_fontsize=10,
    tick_fontsize=9,
    pval_fontsize=9,
):
    """
    Draw a single unfilled boxplot + stripplot panel on a log y-axis.

    Boxes are drawn unfilled with colored borders. Individual patient points
    are overlaid with horizontal jitter. P-value annotation is placed at the
    top of the axes using axes-relative coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to draw into.
    neg : array-like
        Values for the negative/reference group.
    pos : array-like
        Values for the positive/comparison group.
    pval : float
        Pre-computed p-value to annotate (raw or FDR-adjusted).
    neg_color : str
        Hex color for the negative group.
    pos_color : str
        Hex color for the positive group.
    neg_label : str
        X-axis tick label for the negative group.
    pos_label : str
        X-axis tick label for the positive group.
    title : str or None
        Axes title. Pass None to suppress.
    ylabel : str or None
        Y-axis label. Pass None to suppress (e.g. for non-leftmost panels).
    jitter_width : float
        Half-width of uniform horizontal jitter applied to strip points.
    point_size : float
        Marker size for strip points.
    point_alpha : float
        Alpha transparency for strip points.
    box_width : float
        Width of boxplot boxes.
    title_fontsize : int
        Font size for the panel title.
    label_fontsize : int
        Font size for the y-axis label.
    tick_fontsize : int
        Font size for x- and y-axis tick labels.
    pval_fontsize : int
        Font size for the p-value annotation text.
    """
    bp = ax.boxplot(
        [neg, pos],
        positions=[0, 1],
        widths=box_width,
        patch_artist=True,
        medianprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
        showfliers=False,
    )

    for patch, color in zip(bp['boxes'], [neg_color, pos_color]):
        patch.set_facecolor('none')
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)

    for median, color in zip(bp['medians'], [neg_color, pos_color]):
        median.set_color(color)
        median.set_linewidth(2)

    for whisker, color in zip(bp['whiskers'], [neg_color, neg_color, pos_color, pos_color]):
        whisker.set_color(color)

    for cap, color in zip(bp['caps'], [neg_color, neg_color, pos_color, pos_color]):
        cap.set_color(color)

    for data, x, color in zip([neg, pos], [0, 1], [neg_color, pos_color]):
        jitter = np.random.uniform(-jitter_width, jitter_width, size=len(data))
        ax.scatter(x + jitter, data, color=color, s=point_size,
                   alpha=point_alpha, zorder=3, linewidths=0)

    ax.set_yscale('log')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([neg_label, pos_label], fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.text(
        0.5, 0.97,
        f'p = {pval:.2e} ({sig_label(pval)})',
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=pval_fontsize,
    )
    ax.spines[['top', 'right']].set_visible(False)

    if title is not None:
        ax.set_title(title, pad=6, fontsize=title_fontsize, wrap=True)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)


def make_comparison_figure(
    plot_df,
    cell_types,
    neg_label='class II negative',
    pos_label='class II positive',
    classification_col='patient classification',
    neg_color='#462255',
    pos_color='#FF8811',
    ncols=2,
    panel_size=(3.5, 5),
    outpath=None,
    dpi=300,
    pval_map=None,
    title_fontsize=10,
    label_fontsize=10,
    tick_fontsize=9,
    pval_fontsize=9,
    point_alpha=0.8,
    ylabel='Fraction of total cells',
):
    """
    Build a multi-panel comparison figure for a list of cell phenotypes.

    One panel per cell type. Y-axis limits are shared across all panels.
    The leftmost panel in each row gets a y-axis label; others are suppressed.
    If pval_map is provided, those values are used for significance annotation
    instead of running internal Mann-Whitney U tests — pass FDR-adjusted
    p-values here when multiple comparisons are being made.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Patient-level DataFrame with '{cell_type}_fraction' columns and a
        classification column.
    cell_types : list of str
        Cell phenotype names (without '_fraction' suffix).
    neg_label : str
        Value in classification_col for the negative group.
    pos_label : str
        Value in classification_col for the positive group.
    classification_col : str
        Column identifying patient group.
    neg_color : str
        Hex color for the negative group.
    pos_color : str
        Hex color for the positive group.
    ncols : int
        Number of columns in the figure grid.
    panel_size : tuple of (float, float)
        Width and height of each panel in inches.
    outpath : str or Path or None
        If provided, save figure to this path as PDF.
    dpi : int
        DPI for raster outputs.
    pval_map : dict or None
        Pre-computed p-values keyed by cell type name. If provided, these
        are used for significance annotation instead of running internal
        Mann-Whitney U tests. Intended for FDR-adjusted p-values.
    title_fontsize : int
        Font size for panel titles.
    label_fontsize : int
        Font size for axis labels.
    tick_fontsize : int
        Font size for tick labels.
    pval_fontsize : int
        Font size for p-value annotation text.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    nrows = int(np.ceil(len(cell_types) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows)
    )
    axes = np.array(axes).flatten()

    all_vals = []
    for ct in cell_types:
        col = f'{ct}_fraction'
        vals = plot_df.loc[
            plot_df[classification_col].isin([neg_label, pos_label]), col
        ].dropna().values
        all_vals.append(vals)

    all_vals_flat = np.concatenate(all_vals)
    positive_vals = all_vals_flat[all_vals_flat > 0]
    ymin = positive_vals.min() * 0.5
    ymax = all_vals_flat.max() * 2

    for i, (ax, ct) in enumerate(zip(axes, cell_types)):
        col = f'{ct}_fraction'
        neg = plot_df.loc[plot_df[classification_col] == neg_label, col].dropna().values
        pos = plot_df.loc[plot_df[classification_col] == pos_label, col].dropna().values

        if pval_map is not None:
            pval = pval_map.get(ct, np.nan)
        else:
            _, pval = mannwhitneyu(neg, pos, alternative='two-sided')

        ylabel_label = ylabel if (i % ncols == 0) else None
        title = ct.replace(' Cells', '')

        draw_boxstrip_panel(
            ax, neg, pos, pval,
            neg_color=neg_color,
            pos_color=pos_color,
            title=title,
            ylabel=ylabel,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            pval_fontsize=pval_fontsize,
            point_alpha=point_alpha,
        )
        ax.set_ylim(ymin, ymax)

    for ax in axes[len(cell_types):]:
        ax.set_visible(False)

    plt.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, bbox_inches='tight', dpi=dpi)

    return fig, axes

def draw_ridgeline(fig, pos, plot_data, value_col, group_col, color_map,
                   xlabel, title, log=False, overlap_factor=1.6):
    """
    Draw a ridgeline (joy plot) density plot into a given figure position.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    pos : matplotlib.transforms.Bbox
        Position from ax.get_position() — defines where ridgeline is drawn.
    plot_data : pd.DataFrame
        Must contain value_col and group_col.
    value_col : str
        Column to plot on x-axis.
    group_col : str
        Column defining groups (one ridge per group).
    color_map : dict
        Mapping of group value to color.
    xlabel : str
        X-axis label (shown on bottom ridge only).
    title : str
        Title placed above the ridgeline block.
    log : bool
        If True, format x-axis ticks as powers of 10.
    overlap_factor : float
        Controls how much ridges overlap vertically.
    """
    groups = list(color_map.keys())
    n = len(groups)
    x_min = plot_data[value_col].min()
    x_max = plot_data[value_col].max()
    row_height = pos.height / n

    ax_placeholder = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height])
    ax_placeholder.set_visible(False)

    for i, group in enumerate(groups):
        bottom = pos.y0 + (n - 1 - i) * row_height * 0.72
        sub_ax = fig.add_axes([pos.x0, bottom, pos.width, row_height * overlap_factor])
        color = color_map[group]
        vals = plot_data.loc[plot_data[group_col] == group, value_col]

        sns.kdeplot(vals, ax=sub_ax, fill=True, alpha=0.8,
                    color=color, bw_adjust=0.5, linewidth=1.5)
        sns.kdeplot(vals, ax=sub_ax, color='w', lw=2, bw_adjust=0.5)

        sub_ax.set_xlim(x_min, x_max)
        sub_ax.set_facecolor((0, 0, 0, 0))
        sub_ax.set_yticks([])
        sub_ax.set_ylabel('')
        sub_ax.set_xlabel('')
        sub_ax.axhline(0, color=color, linewidth=1.5)
        sub_ax.text(0.01, 0.3, str(group), color=color, fontweight='bold',
                    fontsize=9, transform=sub_ax.transAxes, va='center')
        sub_ax.spines[['left', 'top', 'right']].set_visible(False)

        if i < n - 1:
            sub_ax.set_xticks([])
            sub_ax.spines['bottom'].set_visible(False)
        else:
            if log:
                ticks = [-2, -1, 0, 1, 2]
                sub_ax.set_xticks(ticks)
                sub_ax.set_xticklabels([f'$10^{{{t}}}$' for t in ticks], fontsize=8)
            sub_ax.set_xlabel(xlabel, fontsize=9, labelpad=8)

    fig.text(pos.x0 + pos.width / 2, pos.y0 + pos.height + 0.025,
             title, ha='center', va='bottom', fontsize=11, fontweight='500')


def get_groups(ct, source_df, neg_label='class II negative', pos_label='class II positive'):
    col = f'{ct}_fraction'
    neg = source_df.loc[source_df['patient classification'] == neg_label, col].dropna().values
    pos = source_df.loc[source_df['patient classification'] == pos_label, col].dropna().values
    return neg, pos


# single cell RNA sequencing analysis

def plot_scrna_group_comparison(
    adata,
    genes,
    group_col,
    order,
    palette,
    xtick_labels,
    gene_label_col='feature_name',
    figsize_per_gene=(5, 5),
    nrows=1,
    point_size=5,
    save_path=None,
    dpi=300,
):
    """
    Compare per-sample mean expression of a gene set between two or more groups.

    Aggregates single-cell expression to sample-level means, then plots a
    violin + strip panel per gene with Mann-Whitney U significance annotation.
    Designed for comparing MHC II expression across disease contexts
    (e.g. primary vs metastasis, early vs late stage).

    Parameters
    ----------
    adata : AnnData
        Pre-filtered object containing only the cells to include.
        Group labels must already be present in adata.obs[group_col].
    genes : list of str
        Ensembl IDs to plot (matched to adata.var_names).
    group_col : str
        obs column defining comparison groups (e.g. 'origin', 'stage_group').
    order : list of str
        Group values in display order.
    palette : dict
        Mapping of group value to color.
    xtick_labels : list of str
        Display labels for x-axis ticks, corresponding to order.
    gene_label_col : str
        var column used to label each panel (default 'feature_name').
    figsize_per_gene : tuple
        (width, height) per panel.
    nrows : int
        Number of rows in the figure grid.
    point_size : int
        Marker size for strip points.
    save_path : Path or str or None
        If provided, save figure to this path.
    dpi : int
        Resolution for raster outputs.
    """
    from scipy.stats import mannwhitneyu

    # aggregate to sample-level mean expression
    expr_df = adata.to_df()[genes].copy()
    expr_df['sample']    = adata.obs['sample'].values
    expr_df[group_col]   = adata.obs[group_col].values
    sample_avg           = expr_df.groupby('sample', observed=True)[genes].mean()

    # map each sample to its group (mode, in case of mixed obs)
    sample_group = (
        adata.obs
        .groupby('sample', observed=True)[group_col]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        .dropna()
    )
    sample_avg[group_col] = sample_group
    sample_avg            = sample_avg.dropna(subset=[group_col])

    ncols     = int(np.ceil(len(genes) / nrows))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_gene[0] * ncols, figsize_per_gene[1] * nrows),
    )
    axes = np.atleast_1d(axes).flatten()

    for ax, gene in zip(axes, genes):
        plot_df = sample_avg[[gene, group_col]].dropna()

        if not set(order).issubset(plot_df[group_col].unique()):
            ax.set_title('insufficient data')
            ax.axis('off')
            continue

        sns.violinplot(
            data=plot_df, x=group_col, y=gene, hue=group_col,
            order=order, palette=palette,
            inner=None, fill=False, linewidth=1.2, cut=0,
            ax=ax, legend=False,
        )
        sns.stripplot(
            data=plot_df, x=group_col, y=gene, hue=group_col,
            order=order, palette=palette,
            edgecolor='k', linewidth=1, size=point_size,
            ax=ax, legend=False,
        )

        # resolve display label from var
        feature_label = (
            adata.var.loc[gene, gene_label_col]
            if gene in adata.var.index and gene_label_col in adata.var.columns
            else gene
        )
        ax.set_title(feature_label)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(xtick_labels)
        ax.spines[['top', 'right']].set_visible(False)

        # Mann-Whitney U — two-sided, sample level
        g1 = plot_df.loc[plot_df[group_col] == order[0], gene]
        g2 = plot_df.loc[plot_df[group_col] == order[1], gene]
        _, pval = mannwhitneyu(g1, g2, alternative='two-sided')
        star = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
        ax.text(0.5, 0.85, star, ha='center', va='bottom',
                fontsize=32, transform=ax.transAxes)

    for ax in axes[len(genes):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()


def plot_forest(results_df, fig_path=None, title=None, row_height=0.7,region_colors=None,legend_anchor=(1.02, 0)):
    """
    Forest plot of mixed effects model coefficients with 95% CI.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_mixed_effects with Coefficient, Std Error,
        FDR-adjusted P-value columns.
    fig_path : Path or None
        Output PDF path.
    title : str or None
        Figure title.
    region_colors : dict or None
        Mapping of region name to color. If None, uses default colors.
    """
        
    df = results_df.copy()

    # clean up labels
    df['label'] = (
        df['Cell Type']
        .str.replace(r'^(tumor|stroma|alveoli): ', '', regex=True)
        .str.replace(' Cells', '', regex=False)
    )
    df['region'] = df['Cell Type'].str.extract(r'^(tumor|stroma|alveoli)')
    df['ci95'] = 1.96 * df['Std Error']
    df['significant'] = df['FDR-adjusted P-value'] < 0.05

    # sort by region then coefficient
    region_order = {'tumor': 0, 'stroma': 1, 'alveoli': 2}
    df['region_order'] = df['region'].map(region_order)
    df = df.sort_values(['region_order', 'Coefficient'], ascending=[True, False])

    if region_colors is None:
        region_colors = {
            'tumor':   '#9DD9D2FF',
            'stroma':  '#046E8FFF',
            'alveoli': '#D44D5CFF',
        }

    fig, ax = plt.subplots(figsize=(7, len(df) * row_height + 1))

    for i, (_, row) in enumerate(df.iterrows()):
        color = region_colors.get(row['region'], 'gray')
        marker = 'o' if row['significant'] else 'o'
        fill = color if row['significant'] else 'none'

        ax.errorbar(
            row['Coefficient'], i,
            xerr=row['ci95'],
            fmt='none',
            color=color,
            capsize=3, lw=1.2,
        )
        ax.scatter(
            row['Coefficient'], i,
            color=color,
            facecolors=fill,
            edgecolors=color,
            s=60, zorder=3,
            linewidths=1.2,
        )

    ax.axvline(0, color='gray', linestyle='--', lw=1, alpha=0.7)
    
    # direction labels
    xlim = ax.get_xlim()
    ax.text(1.0, -0.12, 'Higher in MHC II high →',
            ha='right', va='top', fontsize=9, color='gray', style='italic',
            transform=ax.transAxes)
    ax.text(0.0, -0.12, '← Higher in MHC II low',
            ha='left', va='top', fontsize=9, color='gray', style='italic',
            transform=ax.transAxes)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['label'], fontsize=9)
    ax.set_xlabel('Coefficient (log10 cells/mm², class II high vs low)', fontsize=10)
    ax.invert_yaxis()

    if title:
        ax.set_title(title, fontsize=11)

    # region legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=region_colors[r], markersize=8, label=r)
        for r in ['tumor', 'stroma', 'alveoli']
    ]
    legend_elements += [
        Line2D([0], [0], marker='o', color='gray',
               markerfacecolor='gray', markersize=8, label='FDR < 0.05'),
        Line2D([0], [0], marker='o', color='gray',
               markerfacecolor='none', markersize=8, label='ns'),
    ]
    ax.legend(handles=legend_elements,
              loc='lower right',
              bbox_to_anchor=legend_anchor,
              frameon=False, fontsize=9)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')

    return fig

# -------------------------------------------------------------------------
# Helper functions — paired expression plots for scRNA-seq data
# plot_genes_paired_luad and plot_genes_paired_luad_percent_detected
# are defined here pending migration to ceiba.plot_utils
# plot_scrna_group_comparison is defined here as it will be used
# for figures 2c, 2d, and supplemental comparisons
# -------------------------------------------------------------------------

def plot_genes_paired_luad(
    adata,
    genes,
    palette=('tab:grey', 'tab:red'),
    celltype='Epithelial cell',
    figsize_per_gene=(6, 5),
    nrows=1,
    test_mode='nonparametric',
    return_stats=False,
    title='',
    save_path=None,
):
    """
    Plot paired tumor vs normal adjacent expression for a list of genes.

    Supports any cell type in ann_coarse. For epithelial cells, tumor cells
    are restricted to ann_fine == 'Cancer cells' to exclude non-malignant
    epithelial cells in the tumor_primary compartment. For all other cell
    types, all cells of that type in tumor_primary are included.

    Only donors with both origins represented are included (paired design).
    Statistical test is selected based on test_mode — nonparametric (Wilcoxon)
    is recommended for consistency across genes.

    Parameters
    ----------
    adata : AnnData
        Atlas object. Subsetting to celltype and LUAD is performed internally.
    genes : list of str
        Gene symbols to plot (matched via var['feature_name']).
    palette : tuple
        Colors for (normal, tumor).
    celltype : str
        ann_coarse label to subset to (e.g. 'Epithelial cell',
        'Macrophage/Monocyte').
    figsize_per_gene : tuple
        (width, height) per panel.
    nrows : int
        Number of rows in the figure grid.
    test_mode : str
        'nonparametric' — Wilcoxon signed-rank (default, recommended).
        'parametric'    — paired t-test.
        'auto'          — Shapiro-Wilk normality test per gene, then select.
    return_stats : bool
        If True, return a DataFrame of per-gene statistics.
    title : str
        Optional figure-level title.
    save_path : Path or str
        If provided, save figure to this path.
    """
    from scipy.stats import shapiro, wilcoxon, ttest_rel

    # subset to LUAD and selected cell type
    sub = adata[
        (adata.obs['ann_coarse'] == celltype) &
        (adata.obs['disease'].astype(str).str.lower().str.replace('_', ' ')
         == 'lung adenocarcinoma')
    ].copy()
    sub.obs['origin']   = sub.obs['origin'].astype(str).str.strip().str.lower()
    sub.obs['ann_fine'] = sub.obs['ann_fine'].astype(str).str.strip().str.lower()

    normal_cells = sub[sub.obs['origin'] == 'normal_adjacent'].copy()

    # for epithelial cells restrict tumor subset to malignant cells only
    # for all other cell types include all cells of that type in tumor_primary
    if 'epithelial' in celltype.lower():
        tumor_cells = sub[
            (sub.obs['origin'] == 'tumor_primary') &
            (sub.obs['ann_fine'] == 'cancer cells')
        ].copy()
    else:
        tumor_cells = sub[sub.obs['origin'] == 'tumor_primary'].copy()

    sub = ad.concat([normal_cells, tumor_cells], axis=0, merge='same')

    ncols     = int(np.ceil(len(genes) / nrows))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(figsize_per_gene[0] * ncols, figsize_per_gene[1] * nrows),
        sharey=False,
    )
    axes = np.ravel(axes)

    stats_list = []

    for ax, gene in zip(axes, genes):
        try:
            gene_id = sub.var.loc[sub.var['feature_name'] == gene].index[0]
        except IndexError:
            print(f'{gene} not found in var["feature_name"] — skipping')
            ax.axis('off')
            continue

        x = sub[:, gene_id].X
        x = x.toarray().ravel() if hasattr(x, 'toarray') else np.asarray(x).ravel()

        df = (
            pd.DataFrame({
                'donor_id': sub.obs['donor_id'].astype(str).values,
                'origin':   sub.obs['origin'].values,
                gene:       x,
            })
            .groupby(['donor_id', 'origin'], observed=True)[gene]
            .mean()
            .reset_index()
        )

        donor_counts  = df.groupby('donor_id')['origin'].nunique()
        paired_donors = donor_counts[donor_counts == 2].index
        df = df[df['donor_id'].isin(paired_donors)]
        if df.empty:
            print(f'No paired donors for {gene} — skipping')
            ax.axis('off')
            continue

        order = ['normal_adjacent', 'tumor_primary']

        sns.violinplot(
            data=df, x='origin', y=gene, hue='origin',
            order=order, palette=palette,
            inner=None, linewidth=1.2, cut=0, fill=False,
            ax=ax, legend=False,
        )
        sns.stripplot(
            data=df, x='origin', y=gene, hue='origin',
            order=order, palette=palette,
            dodge=False, size=6, alpha=0.7,
            ax=ax, legend=False,
        )

        normal_vals, tumor_vals = [], []
        for did, vals in df.groupby('donor_id'):
            norm_val  = vals.loc[vals['origin'] == 'normal_adjacent', gene].values[0]
            tumor_val = vals.loc[vals['origin'] == 'tumor_primary',   gene].values[0]
            ax.plot([0, 1], [norm_val, tumor_val], color='gray', alpha=0.4, linewidth=0.8)
            normal_vals.append(norm_val)
            tumor_vals.append(tumor_val)

        diff   = np.array(tumor_vals) - np.array(normal_vals)
        p_norm = np.nan

        if test_mode == 'auto':
            if len(diff) >= 3:
                _, p_norm = shapiro(diff)
            if np.isnan(p_norm) or p_norm <= 0.05:
                test_name, (stat, p) = 'Wilcoxon', wilcoxon(tumor_vals, normal_vals)
            else:
                test_name, (stat, p) = 'Paired t-test', ttest_rel(tumor_vals, normal_vals)
        elif test_mode == 'parametric':
            test_name, (stat, p) = 'Paired t-test', ttest_rel(tumor_vals, normal_vals)
        elif test_mode == 'nonparametric':
            test_name, (stat, p) = 'Wilcoxon', wilcoxon(tumor_vals, normal_vals)
        else:
            raise ValueError("test_mode must be 'auto', 'parametric', or 'nonparametric'")

        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ymax  = df[gene].max()
        yoff  = (df[gene].max() - df[gene].min()) * 0.15
        ax.text(0.5, ymax + yoff, star, ha='center', va='bottom',
                fontsize=28, fontweight='bold')
        ax.set_ylim(top=ymax + yoff * 2)
        ax.set_title(gene)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Normal\nAdjacent', 'Primary\nTumor'])
        ax.set_xlabel('')
        ax.set_ylabel('Mean expression' if ax == axes[0] else '')
        ax.spines[['top', 'right']].set_visible(False)

        stats_list.append({
            'Gene':         gene,
            'n_pairs':      len(diff),
            'Normality_p':  p_norm,
            'Test':         test_name,
            'Stat':         stat,
            'p_value':      p,
        })

    for ax in axes[len(genes):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=24, fontweight='bold', y=1.03)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()

    if return_stats:
        return pd.DataFrame(stats_list)


def plot_genes_paired_luad_percent_detected(
    adata,
    genes,
    palette=('tab:grey', 'tab:red'),
    celltype='Epithelial cell',
    detection_thresh=0.0,
    figsize_per_gene=(6, 5),
    return_stats=False,
    save_path=None,
):
    """
    Plot the percent of cells per sample with detected expression (> detection_thresh)
    for each gene, comparing normal adjacent vs primary tumor in paired LUAD donors.

    Complements plot_genes_paired_luad — percent detection is robust to zero-inflation
    and captures the binary presence/absence signal independently of mean expression level.

    Parameters
    ----------
    adata : AnnData
        Full atlas object. Subsetting is performed internally.
    genes : list of str
        Gene symbols to plot (matched via var['feature_name']).
    palette : tuple
        Colors for (normal, tumor).
    celltype : str
        ann_coarse label to subset to.
    detection_thresh : float
        Cells with expression > this value are counted as detected (default 0.0).
    figsize_per_gene : tuple
        (width, height) per panel.
    return_stats : bool
        If True, return a DataFrame of per-gene statistics.
    save_path : Path or str
        If provided, save figure to this path.
    """
    from scipy.stats import wilcoxon

    n_genes = len(genes)
    fig, axes = plt.subplots(
        nrows=1, ncols=n_genes,
        figsize=(figsize_per_gene[0] * n_genes, figsize_per_gene[1]),
        sharey=False,
    )
    axes = np.atleast_1d(axes).flatten()

    stats_list = []

    for ax, gene in zip(axes, genes):
        epi = adata[adata.obs['ann_coarse'] == celltype].copy()
        epi = epi[
            epi.obs['disease'].astype(str).str.lower().str.replace('_', ' ')
            == 'lung adenocarcinoma'
        ].copy()
        epi.obs['origin'] = epi.obs['origin'].astype(str).str.strip().str.lower()

        normal_epi = epi[epi.obs['origin'] == 'normal_adjacent'].copy()
        tumor_epi  = epi[
            (epi.obs['origin'] == 'tumor_primary') &
            (epi.obs['ann_fine'].astype(str).str.lower() == 'cancer cells')
        ].copy()
        epi = ad.concat([normal_epi, tumor_epi], axis=0, merge='same')

        try:
            gene_id = epi.var.loc[epi.var['feature_name'] == gene].index[0]
        except IndexError:
            print(f'{gene} not found in var["feature_name"] — skipping')
            ax.axis('off')
            continue

        x = epi[:, gene_id].X
        x = x.toarray().ravel() if hasattr(x, 'toarray') else np.asarray(x).ravel()

        # fraction of cells per sample with detected expression
        df = (
            pd.DataFrame({
                'donor_id': epi.obs['donor_id'].astype(str).values,
                'sample':   epi.obs['sample'].astype(str).values,
                'origin':   epi.obs['origin'].values,
                'detected': (x > detection_thresh).astype(int),
            })
            .groupby(['donor_id', 'sample', 'origin'], observed=True)['detected']
            .mean()
            .mul(100.0)
            .reset_index()
            .rename(columns={'detected': 'pct_detected'})
        )

        donor_counts  = df.groupby('donor_id')['origin'].nunique()
        paired_donors = donor_counts[donor_counts == 2].index
        df = df[df['donor_id'].isin(paired_donors)]
        if df.empty:
            ax.axis('off')
            continue

        order = ['normal_adjacent', 'tumor_primary']

        sns.violinplot(
            data=df, x='origin', y='pct_detected', hue='origin',
            order=order, palette=palette,
            inner=None, linewidth=1.2, cut=0, fill=False,
            ax=ax, legend=False,
        )
        sns.stripplot(
            data=df, x='origin', y='pct_detected', hue='origin',
            order=order, palette=palette,
            dodge=False, size=8, alpha=0.7,
            ax=ax, legend=False,
        )

        tumor_vals, normal_vals = [], []
        for did, vals in df.groupby('donor_id'):
            if set(order).issubset(set(vals['origin'].values)):
                norm_val  = vals.loc[vals['origin'] == 'normal_adjacent', 'pct_detected'].values[0]
                tumor_val = vals.loc[vals['origin'] == 'tumor_primary',   'pct_detected'].values[0]
                ax.plot([0, 1], [norm_val, tumor_val], color='gray', alpha=0.4, linewidth=0.8)
                normal_vals.append(norm_val)
                tumor_vals.append(tumor_val)

        if tumor_vals and normal_vals:
            stat, p = wilcoxon(tumor_vals, normal_vals, alternative='two-sided')
            star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        else:
            p, star = np.nan, 'ns'

        ymax = df['pct_detected'].max()
        ax.text(0.5, ymax * 0.85, star, ha='center', va='bottom', fontsize=32)

        ax.set_title(gene)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(['Normal\nAdjacent', 'Primary\nTumor'])
        ax.set_xlabel('')
        ax.set_ylabel('% cells detected' if ax == axes[0] else '')
        ax.spines[['top', 'right']].set_visible(False)

        stats_list.append({'Gene': gene, 'n_pairs': len(tumor_vals), 'Wilcoxon_p': p})

    for ax in axes[len(genes):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()

    if return_stats:
        return pd.DataFrame(stats_list)


def plot_genes_pct_expressing_luad(
    adata,
    genes,
    palette=('tab:grey', 'tab:red'),
    celltype='Epithelial cell',
    figsize_per_gene=(6, 5),
    nrows=1,
    test_mode='nonparametric',
    return_stats=False,
    title='',
    save_path=None,
):
    """
    Plot the percent of cells per donor expressing each gene (expression > 0),
    comparing normal adjacent vs primary tumor in paired LUAD donors.

    Complements plot_genes_paired_luad — percent detection is robust to
    zero-inflation and captures presence/absence signal independently of
    mean expression magnitude.

    Supports any cell type in ann_coarse. For epithelial cells, tumor cells
    are restricted to ann_fine == 'Cancer cells'. For all other cell types,
    all cells of that type in tumor_primary are included.

    Parameters
    ----------
    adata : AnnData
        Atlas object. Subsetting to celltype and LUAD is performed internally.
    genes : list of str
        Gene symbols to plot (matched via var['feature_name']).
    palette : tuple
        Colors for (normal, tumor).
    celltype : str
        ann_coarse label to subset to.
    figsize_per_gene : tuple
        (width, height) per panel.
    nrows : int
        Number of rows in the figure grid.
    test_mode : str
        'nonparametric' — Wilcoxon signed-rank (default, recommended).
        'parametric'    — paired t-test.
        'auto'          — Shapiro-Wilk normality test per gene, then select.
    return_stats : bool
        If True, return a DataFrame of per-gene statistics.
    title : str
        Optional figure-level title.
    save_path : Path or str
        If provided, save figure to this path.
    """
    from scipy.stats import shapiro, wilcoxon, ttest_rel

    sub = adata[
        (adata.obs['ann_coarse'] == celltype) &
        (adata.obs['disease'].astype(str).str.lower().str.replace('_', ' ')
         == 'lung adenocarcinoma')
    ].copy()
    sub.obs['origin']   = sub.obs['origin'].astype(str).str.strip().str.lower()
    sub.obs['ann_fine'] = sub.obs['ann_fine'].astype(str).str.strip().str.lower()

    normal_cells = sub[sub.obs['origin'] == 'normal_adjacent'].copy()

    if 'epithelial' in celltype.lower():
        tumor_cells = sub[
            (sub.obs['origin'] == 'tumor_primary') &
            (sub.obs['ann_fine'] == 'cancer cells')
        ].copy()
    else:
        tumor_cells = sub[sub.obs['origin'] == 'tumor_primary'].copy()

    sub = ad.concat([normal_cells, tumor_cells], axis=0, merge='same')

    ncols     = int(np.ceil(len(genes) / nrows))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(figsize_per_gene[0] * ncols, figsize_per_gene[1] * nrows),
        sharey=False,
    )
    axes = np.ravel(axes)

    stats_list = []

    for ax, gene in zip(axes, genes):
        try:
            gene_id = sub.var.loc[sub.var['feature_name'] == gene].index[0]
        except IndexError:
            print(f'{gene} not found in var["feature_name"] — skipping')
            ax.axis('off')
            continue

        x = sub[:, gene_id].X
        x = x.toarray().ravel() if hasattr(x, 'toarray') else np.asarray(x).ravel()

        # fraction of cells per donor with any detected expression
        df = (
            pd.DataFrame({
                'donor_id': sub.obs['donor_id'].astype(str).values,
                'origin':   sub.obs['origin'].values,
                'detected': (x > 0).astype(float),
            })
            .groupby(['donor_id', 'origin'], observed=True)['detected']
            .mean()
            .mul(100.0)
            .reset_index()
            .rename(columns={'detected': 'pct_detected'})
        )

        donor_counts  = df.groupby('donor_id')['origin'].nunique()
        paired_donors = donor_counts[donor_counts == 2].index
        df = df[df['donor_id'].isin(paired_donors)]
        if df.empty:
            print(f'No paired donors for {gene} — skipping')
            ax.axis('off')
            continue

        order = ['normal_adjacent', 'tumor_primary']

        sns.violinplot(
            data=df, x='origin', y='pct_detected', hue='origin',
            order=order, palette=palette,
            inner=None, linewidth=1.2, cut=0, fill=False,
            ax=ax, legend=False,
        )
        sns.stripplot(
            data=df, x='origin', y='pct_detected', hue='origin',
            order=order, palette=palette,
            dodge=False, size=6, alpha=0.7,
            ax=ax, legend=False,
        )

        normal_vals, tumor_vals = [], []
        for did, vals in df.groupby('donor_id'):
            norm_val  = vals.loc[vals['origin'] == 'normal_adjacent', 'pct_detected'].values[0]
            tumor_val = vals.loc[vals['origin'] == 'tumor_primary',   'pct_detected'].values[0]
            ax.plot([0, 1], [norm_val, tumor_val], color='gray', alpha=0.4, linewidth=0.8)
            normal_vals.append(norm_val)
            tumor_vals.append(tumor_val)

        diff   = np.array(tumor_vals) - np.array(normal_vals)
        p_norm = np.nan

        if test_mode == 'auto':
            if len(diff) >= 3:
                _, p_norm = shapiro(diff)
            if np.isnan(p_norm) or p_norm <= 0.05:
                test_name, (stat, p) = 'Wilcoxon', wilcoxon(tumor_vals, normal_vals)
            else:
                test_name, (stat, p) = 'Paired t-test', ttest_rel(tumor_vals, normal_vals)
        elif test_mode == 'parametric':
            test_name, (stat, p) = 'Paired t-test', ttest_rel(tumor_vals, normal_vals)
        elif test_mode == 'nonparametric':
            test_name, (stat, p) = 'Wilcoxon', wilcoxon(tumor_vals, normal_vals)
        else:
            raise ValueError("test_mode must be 'auto', 'parametric', or 'nonparametric'")

        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ymax  = df['pct_detected'].max()
        yoff  = (df['pct_detected'].max() - df['pct_detected'].min()) * 0.15
        ax.text(0.5, ymax + yoff, star, ha='center', va='bottom',
                fontsize=28, fontweight='bold')
        ax.set_ylim(top=ymax + yoff * 2)
        ax.set_title(gene)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Normal\nAdjacent', 'Primary\nTumor'])
        ax.set_xlabel('')
        ax.set_ylabel('% cells expressing' if ax == axes[0] else '')
        ax.spines[['top', 'right']].set_visible(False)

        stats_list.append({
            'Gene':        gene,
            'n_pairs':     len(diff),
            'Normality_p': p_norm,
            'Test':        test_name,
            'Stat':        stat,
            'p_value':     p,
        })

    for ax in axes[len(genes):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=24, fontweight='bold', y=1.03)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()

    if return_stats:
        return pd.DataFrame(stats_list)

def plot_celltype_comparison_luad(
    adata,
    genes,
    tissue='tumor_primary',
    celltypes=('Epithelial cell', 'Macrophage/Monocyte'),
    palette=('#9DD9D2', '#D44D5C'),
    figsize_per_gene=(6, 5),
    nrows=1,
    test_mode='nonparametric',
    return_stats=False,
    title='',
    save_path=None,
):
    """
    Compare gene expression between two cell types within the same tissue
    region (e.g. epithelial vs macrophage in primary tumor or NAT).

    Donors are paired — only donors with both cell types represented in the
    specified tissue are included. Useful for showing that MHC II expression
    in malignant epithelial cells is comparable to professional APCs.

    Parameters
    ----------
    adata : AnnData
        Atlas object containing LUAD cells.
    genes : list of str
        Gene symbols to plot (matched via var['feature_name']).
    tissue : str
        Origin value to filter to ('tumor_primary' or 'normal_adjacent').
    celltypes : tuple of str
        Two ann_coarse labels to compare.
    palette : tuple or str
        Colors for each cell type, or a seaborn palette name.
    figsize_per_gene : tuple
        (width, height) per panel.
    nrows : int
        Number of rows in the figure grid.
    test_mode : str
        'nonparametric' — Wilcoxon signed-rank (default, recommended).
        'parametric'    — paired t-test.
        'auto'          — Shapiro-Wilk normality test per gene, then select.
    return_stats : bool
        If True, return a DataFrame of per-gene statistics.
    title : str
        Optional figure-level title.
    save_path : Path or str
        If provided, save figure to this path.
    """
    from scipy.stats import shapiro, wilcoxon, ttest_rel

    # subset to LUAD and specified tissue region
    sub = adata[
        (adata.obs['origin'].str.lower() == tissue.lower()) &
        (adata.obs['disease'].astype(str).str.lower().str.replace('_', ' ')
         == 'lung adenocarcinoma')
    ].copy()
    sub.obs['ann_coarse'] = sub.obs['ann_coarse'].astype(str).str.strip()

    ncols     = int(np.ceil(len(genes) / nrows))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(figsize_per_gene[0] * ncols, figsize_per_gene[1] * nrows),
        sharey=False,
    )
    axes = np.ravel(axes)

    stats_list = []

    for ax, gene in zip(axes, genes):
        try:
            gene_id = sub.var.loc[sub.var['feature_name'] == gene].index[0]
        except IndexError:
            print(f'{gene} not found in var["feature_name"] — skipping')
            ax.axis('off')
            continue

        x = sub[:, gene_id].X
        x = x.toarray().ravel() if hasattr(x, 'toarray') else np.asarray(x).ravel()

        df = (
            pd.DataFrame({
                'donor_id': sub.obs['donor_id'].astype(str).values,
                'celltype': sub.obs['ann_coarse'].values,
                gene:       x,
            })
            .loc[lambda d: d['celltype'].isin(celltypes)]
            .groupby(['donor_id', 'celltype'], observed=True)[gene]
            .mean()
            .reset_index()
        )

        # retain donors with both cell types represented
        donor_counts  = df.groupby('donor_id')['celltype'].nunique()
        paired_donors = donor_counts[donor_counts == 2].index
        df = df[df['donor_id'].isin(paired_donors)]
        if df.empty:
            print(f'No paired donors for {gene} — skipping')
            ax.axis('off')
            continue

        sns.violinplot(
            data=df, x='celltype', y=gene, hue='celltype',
            order=celltypes, palette=palette,
            inner=None, linewidth=1.2, cut=0, fill=False,
            ax=ax, legend=False,
        )
        sns.stripplot(
            data=df, x='celltype', y=gene, hue='celltype',
            order=celltypes, palette=palette,
            dodge=False, size=7, alpha=0.7,
            ax=ax, legend=False,
        )

        vals1, vals2 = [], []
        for did, vals in df.groupby('donor_id'):
            v1 = vals.loc[vals['celltype'] == celltypes[0], gene].values[0]
            v2 = vals.loc[vals['celltype'] == celltypes[1], gene].values[0]
            ax.plot([0, 1], [v1, v2], color='gray', alpha=0.4, linewidth=0.8)
            vals1.append(v1)
            vals2.append(v2)

        diff   = np.array(vals2) - np.array(vals1)
        p_norm = np.nan

        if test_mode == 'auto':
            if len(diff) >= 3:
                _, p_norm = shapiro(diff)
            if np.isnan(p_norm) or p_norm <= 0.05:
                test_name, (stat, p) = 'Wilcoxon', wilcoxon(vals2, vals1)
            else:
                test_name, (stat, p) = 'Paired t-test', ttest_rel(vals2, vals1)
        elif test_mode == 'parametric':
            test_name, (stat, p) = 'Paired t-test', ttest_rel(vals2, vals1)
        elif test_mode == 'nonparametric':
            test_name, (stat, p) = 'Wilcoxon', wilcoxon(vals2, vals1)
        else:
            raise ValueError("test_mode must be 'auto', 'parametric', or 'nonparametric'")

        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ymax  = df[gene].max()
        yoff  = (df[gene].max() - df[gene].min()) * 0.2
        ax.text(0.5, ymax + yoff, star, ha='center', va='bottom', fontsize=26)
        ax.set_ylim(top=ymax + yoff * 2)
        ax.set_title(f'{gene} ({tissue})')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(celltypes))
        ax.set_xlabel('')
        ax.set_ylabel('Mean expression' if ax == axes[0] else '')
        ax.spines[['top', 'right']].set_visible(False)

        stats_list.append({
            'Gene':        gene,
            'tissue':      tissue,
            'n_pairs':     len(diff),
            'Test':        test_name,
            'Stat':        stat,
            'p_value':     p,
            'Normality_p': p_norm,
        })

    for ax in axes[len(genes):]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=24, fontweight='bold', y=1.03)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()

    if return_stats:
        return pd.DataFrame(stats_list)


def plot_ciita_s100p_paired(wide, figsize=(8, 3.8), dpi=150):
    """
    Two-panel paired dot + delta figure comparing CIITA expression in S100P+
    vs S100P- epithelial cells, separately for normal adjacent and tumor.

    Left panel of each pair: per-sample paired dot plot (S100P- vs S100P+).
    Right panel: within-sample delta (S100P+ minus S100P-) with Wilcoxon test.

    Parameters
    ----------
    wide : pd.DataFrame
        Output of ciita_expr_by_s100p_strata_per_sample.
    figsize : tuple
        Figure dimensions in inches.
    dpi : int
        Figure resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    origins = ['normal_adjacent', 'tumor_primary']
    labels  = ['Normal adjacent', 'Tumor']

    COLOR_NEG  = '#C4A882'
    COLOR_POS  = '#7B2D8B'
    ALPHA_LINE = 0.35
    DOT_SIZE   = 4

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs  = fig.add_gridspec(
        1, 5,
        width_ratios=[1.6, 0.85, 0.25, 1.6, 0.85],
        left=0.08, right=0.97, top=0.85, bottom=0.15,
        wspace=0.08,
    )
    ax_slots = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[0, 4]),
    ]

    for i, (origin, label) in enumerate(zip(origins, labels)):
        ax_paired = ax_slots[i * 2]
        ax_delta  = ax_slots[i * 2 + 1]

        sub = wide[wide['origin'] == origin].dropna(
            subset=['CIITA_mean_S100Pneg', 'CIITA_mean_S100Ppos', 'delta_pos_minus_neg']
        )
        deltas   = sub['delta_pos_minus_neg'].values
        stat, p  = wilcoxon(deltas)
        s        = sig_label(p)
        med      = np.median(deltas)

        # paired dot plot
        rng    = np.random.default_rng(42 + i)
        jitter = rng.uniform(-0.08, 0.08, len(sub))

        for xn, xp, yn, yp in zip(
            jitter, 1 + jitter,
            sub['CIITA_mean_S100Pneg'],
            sub['CIITA_mean_S100Ppos'],
        ):
            ax_paired.plot([xn, xp], [yn, yp],
                           color='grey', lw=0.6, alpha=ALPHA_LINE, zorder=1)

        ax_paired.scatter(jitter,     sub['CIITA_mean_S100Pneg'],
                          color=COLOR_NEG, s=DOT_SIZE, zorder=3)
        ax_paired.scatter(1 + jitter, sub['CIITA_mean_S100Ppos'],
                          color=COLOR_POS, s=DOT_SIZE, zorder=3)

        ax_paired.set_xticks([0, 1])
        ax_paired.set_xticklabels(['S100P-', 'S100P+'], fontsize=8)
        ax_paired.set_xlim(-0.4, 1.4)
        ax_paired.set_ylabel('Mean CIITA (CIITA+ cells)', fontsize=7.5)
        ax_paired.set_title(label, fontsize=9, fontweight='bold', pad=6)
        ax_paired.spines[['top', 'right']].set_visible(False)
        ax_paired.tick_params(labelsize=7)

        # delta strip plot
        rng2 = np.random.default_rng(99 + i)
        jit2 = rng2.uniform(-0.18, 0.18, len(deltas))
        ymin = min(deltas) - np.ptp(deltas) * 0.08
        ymax = max(deltas) + np.ptp(deltas) * 0.45

        ax_delta.set_ylim(ymin, ymax)
        ax_delta.axhline(0, color='black', lw=0.8, ls='--', zorder=1)
        ax_delta.scatter(jit2, deltas, color=COLOR_POS, s=DOT_SIZE, alpha=0.75, zorder=3)
        ax_delta.plot([-0.25, 0.25], [med, med], color='black', lw=2, zorder=4)

        ax_delta.text(0.5, 0.97, f'{s}  p={p:.2e}',
                      ha='center', va='top', fontsize=6.5,
                      transform=ax_delta.transAxes)
        ax_delta.text(0.5, 0.02, f'n={len(sub)}',
                      ha='center', va='bottom', fontsize=6.5, color='grey',
                      transform=ax_delta.transAxes)

        ax_delta.set_xlim(-0.5, 0.5)
        ax_delta.set_xticks([])
        ax_delta.set_ylabel('')
        ax_delta.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        ax_delta.tick_params(axis='y', labelsize=7, left=False, labelleft=False)

    fig.suptitle('CIITA expression in S100P+ vs S100P- epithelial cells',
                 fontsize=9, y=0.97)
    return fig


def plot_dual_metric_panel(
    adata,
    genes_dict,
    cell_types,
    group_col='MHC2_clustering',
    group_order=['MHC class II High', 'MHC class II Low'],
    palette={'MHC class II High': '#FF8811FF', 'MHC class II Low': '#462255FF'},
    fig_path=None,
    title=None,
):
    """
    Plot percent expressing and mean expression (among positive cells)
    side by side for each gene × cell type combination.

    Parameters
    ----------
    adata : AnnData
    genes_dict : dict
        {gene_symbol: [ensembl_id, ...]}
    cell_types : list of str
    group_col : str
    group_order : list of str
    palette : dict
    fig_path : Path or None
    title : str or None

    Returns
    -------
    stats_df : pd.DataFrame
    """
    gene_list = list(genes_dict.keys())
    stats_records = []
    pct_data  = {}
    mean_data = {}

    for cell_type in cell_types:
        subset = adata[adata.obs['cell_type_major'] == cell_type]
        pct_data[cell_type]  = {}
        mean_data[cell_type] = {}

        for gene, gene_ens_list in genes_dict.items():
            gene_ens = gene_ens_list[0]
            x = subset.to_df()[gene_ens]
            expr = x.to_frame(name='expr')
            expr['sample']  = subset.obs['sample'].values
            expr['cluster'] = subset.obs[group_col].values

            # % expressing
            pct_df = (
                (expr.assign(detected=(expr['expr'] > 0).astype(int))
                 .groupby(['sample', 'cluster'], observed=True)['detected']
                 .mean() * 100)
                .reset_index()
                .rename(columns={'detected': 'expr'})
            )
            pct_data[cell_type][gene] = pct_df

            # mean among positive cells
            mean_df = (
                expr[expr['expr'] > 0]
                .groupby(['sample', 'cluster'], observed=True)['expr']
                .mean()
                .reset_index()
            )
            mean_data[cell_type][gene] = mean_df

            # stats — run on % expressing
            g1 = pct_df.loc[pct_df['cluster'] == group_order[0], 'expr']
            g2 = pct_df.loc[pct_df['cluster'] == group_order[1], 'expr']
            p  = mannwhitneyu(g1, g2, alternative='two-sided')[1] if len(g1) > 0 and len(g2) > 0 else np.nan
            stats_records.append({'cell_type': cell_type, 'gene': gene, 'p_value': p})

    stats_df = pd.DataFrame(stats_records)
    _, fdr, _, _ = multipletests(stats_df['p_value'].fillna(1), method='fdr_bh')
    stats_df['FDR_p']     = fdr
    stats_df['sig_label'] = stats_df['FDR_p'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )

    # figure — 2 columns per gene (pct + mean), rows = cell types
    nrows = len(cell_types)
    ncols = len(gene_list) * 2

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3 * ncols, 4 * nrows),
                             sharey=False)
    axes = np.atleast_2d(axes)

    for r, cell_type in enumerate(cell_types):
        for c_gene, gene in enumerate(gene_list):
            for c_metric, (metric_label, data) in enumerate([
                ('% expressing', pct_data[cell_type][gene]),
                ('mean expr\n(pos cells)', mean_data[cell_type][gene]),
            ]):
                col = c_gene * 2 + c_metric
                ax  = axes[r, col]

                sns.boxplot(
                    data=data, x='cluster', y='expr', hue='cluster',
                    order=group_order, palette=palette,
                    ax=ax, fill=False, showfliers=False, legend=False,
                )
                sns.stripplot(
                    data=data, x='cluster', y='expr', hue='cluster',
                    order=group_order, palette=palette,
                    ax=ax, edgecolor='k', linewidth=1,
                    size=3, alpha=0.6, legend=False,
                )

                if r == 0:
                    ax.set_title(f'{gene}\n{metric_label}', fontsize=14, pad=10)
                if c_gene == 0 and c_metric == 0:
                    ax.set_ylabel(cell_type, fontsize=14)
                else:
                    ax.set_ylabel('')

                ax.set_xlabel('')
                ax.set_xticklabels([])
                ax.spines[['top', 'right']].set_visible(False)

                # significance on % expressing column only
                if c_metric == 0:
                    sig = stats_df.loc[
                        (stats_df['cell_type'] == cell_type) &
                        (stats_df['gene'] == gene), 'sig_label'
                    ].values[0]
                    if sig:
                        ax.text(0.5, 0.78, sig, ha='center', va='bottom',
                                transform=ax.transAxes, fontsize=36)

    handles = [
        plt.Line2D([0], [0], color=palette[g], marker='o', linestyle='',
                   markersize=8, markeredgecolor='k', markeredgewidth=1, label=g)
        for g in group_order
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=14)

    if title:
        fig.suptitle(title, fontsize=16, y=1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight')

    plt.show()
    return stats_df
    