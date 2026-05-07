# Project context for Claude

## Repository
- Name: mhc2-luad-spatial
- Path: /home/gh8sj/projects/mhc2-luad-paper
- GitHub: private, will be made public at submission

## Package
- `src/ceiba/` — custom analysis package
  - `plot_utils.py` — plotting utilities (sig_label, draw_boxstrip_panel, plot_scrna_group_comparison etc.)
  - `halo_utils.py` — HALO export processing
  - `tma_grid.py` — TMA core grid assignment

## Configuration
- `data/paths_config.yml` — all data paths, committed to repo
- `data/cbioportal_query.md` — instructions for re-downloading ORIEN data

## Notebook style
- Match formatting of reference notebook: `s100p-protein-mif-analysis.ipynb`
- Markdown headers: `##` for major sections, descriptive paragraph below each
- Code: clean, commented inline, no redundant cells, no exploratory scaffolding
- Save paths: always use `fig_out / 'filename.pdf'` from paths config
- Font settings: `sns.set(font_scale=1.8)`, `sns.set_style('ticks')`
- PDF fonts: `plt.rcParams['pdf.fonttype'] = 42`
- Colors: `cmap_low_high`, `cmap_high_low` defined in setup cell

## Completed notebooks
- `notebooks/scrna/figure2-mhc2-tumor-normal.ipynb`
- `notebooks/scrna/figure2-supp-celltype-comparison.ipynb`
- `notebooks/scrna/mhc2-patient-classification.ipynb`
- `notebooks/scrna/figure2e-at2-club-scores.ipynb`
- `notebooks/bulk_rnaseq/figure1ab-bulk-rnaseq-survival.ipynb`
- `notebooks/IHC_UVA_cohort/figure1d-uva-cohort-oncoprint.ipynb`

## In progress
- `notebooks/cosmx/` — 5 notebooks to clean and organize:
  - cosmx_celltyping
  - ihc_groups_cell_type_enrichment
  - mhc2_ihc_deg
  - umap_cosmx
  - filtering_and_merging_metadata

## Folder structure
notebooks/
├── scrna/
├── bulk_rnaseq/
├── IHC_UVA_cohort/
├── cosmx/                ← in progress
└── cross_modality/       ← after cosmx

## Key data objects
- `luad_epithelial_harmony.h5ad` — LUAD epithelial cells, Harmony UMAP
- `luad.h5ad` — full LUAD all cell types
- `luad_mhc2_classified.h5ad` — full LUAD with MHC2_clustering in obs
- `cancer_cells_mhc2_classification.h5ad` — used for figure 2e scoring

## Pre-submission checklist
See `CHECKLIST.md` in repo root