options(stringsAsFactors = F)

setwd("~/Documents/GitHub/risk/experiment/data/genetech/")
tpm <- read.table("Genetech_expression_TPM.txt", sep="\t")
tpm <- t(tpm)
tpm <- as.data.frame(tpm)

setwd("~/Documents/GitHub/risk/models/")
genes <- read.table("batch_norm.geneset.txt", header = T)[,1]

tpm <- tpm[,(colnames(tpm) %in% genes)]

