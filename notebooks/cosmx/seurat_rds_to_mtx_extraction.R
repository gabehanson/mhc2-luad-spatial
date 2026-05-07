# extract_seurat_to_mtx.R
# ------------------------
# Extracts CosMx expression matrices and metadata from Seurat RDS objects
# exported by the NanoString spatial biology core.
#
# Required because the flat CSV files in the core export were uploaded as
# empty files (0 bytes). This script reads the Seurat objects directly and
# exports count matrices in MTX format for loading into Python/AnnData.
#
# Input:  Seurat RDS files in rds_dir
# Output: Per-slide directories in out_dir containing:
#           - {slide}_counts.mtx       — sparse count matrix
#           - {slide}_genes.csv        — gene names
#           - {slide}_barcodes.csv     — cell barcodes
#           - {slide}_metadata_file.csv — full cell metadata
#
# Usage: Rscript extract_seurat_to_mtx.R
# Or run interactively in RStudio

library(SeuratObject)
library(Matrix)
library(yaml)

cfg <- read_yaml('/home/gh8sj/projects/mhc2-luad-paper/data/paths_config.yml')

rds_dir <- cfg$datasets$cosmx$rds
out_dir  <- cfg$datasets$cosmx$extracted

dir.create(out_dir, showWarnings = FALSE)

slides <- c('2014.1', '2014.2', '2014.3', '2014.4')

for (slide in slides) {
    cat('Processing slide', slide, '\n')

    rds_path <- file.path(rds_dir, paste0('seuratObject_Lung.Ad.Ca.', slide, '.RDS'))
    obj <- readRDS(rds_path)

    slide_name <- paste0('LungAdCa', gsub('\\.', '', slide))
    slide_out  <- file.path(out_dir, slide_name)
    dir.create(slide_out, showWarnings = FALSE)

    # export raw count matrix
    counts <- GetAssayData(obj, assay='RNA', layer='counts')
    writeMM(counts, file.path(slide_out, paste0(slide_name, '_counts.mtx')))

    # export gene names and barcodes
    write.csv(data.frame(gene=rownames(counts)),
              file.path(slide_out, paste0(slide_name, '_genes.csv')),
              row.names=FALSE, quote=FALSE)
    write.csv(data.frame(barcode=colnames(counts)),
              file.path(slide_out, paste0(slide_name, '_barcodes.csv')),
              row.names=FALSE, quote=FALSE)

    # export full cell metadata
    write.csv(obj@meta.data,
              file.path(slide_out, paste0(slide_name, '_metadata_file.csv')),
              row.names=TRUE, quote=FALSE)

    cat('Done:', slide_name, '-', ncol(counts), 'cells x', nrow(counts), 'genes\n')
    rm(obj); gc()
}

cat('All slides extracted to', out_dir, '\n')