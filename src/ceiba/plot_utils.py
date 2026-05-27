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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu


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