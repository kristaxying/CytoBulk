if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

if (!requireNamespace("dplyr", quietly = TRUE))
  install.packages("dplyr")

if (!requireNamespace("TxDb.Hsapiens.UCSC.hg38.knownGene", quietly = TRUE))
  BiocManager::install("TxDb.Hsapiens.UCSC.hg38.knownGene")

if (!requireNamespace("AnnotationDbi", quietly = TRUE))
  BiocManager::install("AnnotationDbi")

if (!requireNamespace("org.Hs.eg.db", quietly = TRUE))
  BiocManager::install("org.Hs.eg.db")

if (!requireNamespace("GenomicFeatures", quietly = TRUE))
  BiocManager::install("GenomicFeatures")

if (!requireNamespace("tibble", quietly = TRUE))
  install.packages("tibble")

library(TxDb.Hsapiens.UCSC.hg38.knownGene)
library(GenomicFeatures)
library(dplyr)
library(tibble)
library(AnnotationDbi)
library(org.Hs.eg.db)

setwd("D:/project/deconvolution_algorithm/coding/Spatial_mapping/CytoBulk/config/")

txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene

exons <- exonsBy(txdb, by = "gene")

exon_lengths <- sapply(exons, function(x) sum(width(x)))
# Get gene symbols
gene_ids <- names(exon_lengths)
gene_symbols <- mapIds(org.Hs.eg.db, keys = gene_ids, column = "SYMBOL", keytype = "ENTREZID", multiVals = "first")

# Merge lengths and symbols
exon_lengths_df <- data.frame(geneID = gene_ids, lengths = exon_lengths, row.names = NULL)
exon_lengths_df$SYMBOL <- gene_symbols

# Remove geneID column and reorder columns
exon_lengths_df <- exon_lengths_df[, c("SYMBOL", "lengths")]


write.table(exon_lengths_df, file = "gene_length.txt", sep = "\t", quote = FALSE, row.names = FALSE)