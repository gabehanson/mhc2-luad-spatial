## CosMx raw data notes

The CosMx data was exported from the NanoString platform as Seurat RDS objects
and uploaded to AWS S3. The R script `notebooks/cosmx/seurat_rds_to_mtx_extraction.R`
converts these RDS files to MTX format for loading into Python/AnnData.
Paths to the RDS files and extracted outputs are managed via `paths_config.yml`
under `datasets.cosmx.rds` and `datasets.cosmx.extracted`.