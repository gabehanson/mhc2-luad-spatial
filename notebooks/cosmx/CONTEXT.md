# CosMx context

## CosMx raw data notes
The CosMx data was exported from the NanoString platform as Seurat RDS objects
and uploaded to AWS S3. The R script `notebooks/cosmx/seurat_rds_to_mtx_extraction.R`
converts these RDS files to MTX format for loading into Python/AnnData.
Paths to the RDS files and extracted outputs are managed via `paths_config.yml`
under `datasets.cosmx.rds` and `datasets.cosmx.extracted`.

## Pipeline order
1. `prepare_salcher_cosmx_reference` — one-time setup, builds Salcher reference subset with multi-gene pseudo-probes
2. `filtering_and_merging_metadata` — loads MTX files, QC filtering, saves `combined_adata_qc_filtered.h5ad`
3. `umap_cosmx` — tonsil removal, normalization, UMAP + Leiden, saves `cosmx_umap.h5ad`
4. `cosmx_cell_typing` — marker gene scoring, cluster-level cell type assignment, saves `tumor_data_scored.h5ad`
5. `mhc2_ihc_deg` — pseudobulk DESeq2 on epithelial cells, volcano, MA plot
6. `ihc_groups_cell_type_enrichment` — cell type fraction comparison by IHC group
7. `cancer_cell_pathway_analysis` — GSEA with Hallmark and CosMx pathway sets

## Key processed files
- `tumor_data_scored.h5ad` — primary output, input for notebooks 5-7
- `epithelial.h5ad` — epithelial subset with UMAP, input for notebooks 5 and 7
- `salcher_cosmx_gene_set.h5ad` — Salcher reference subset, input for notebooks 3-4