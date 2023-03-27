library("anndata")
library("scalop")

args = commandArgs(trailingOnly=TRUE)

data_path <- args[1]
n_clusters <- strtoi(args[2])
results_dir <- args[3]

ad <- anndata::read_h5ad(data_path)
matrix <- ad$X
matrix <- t(as.matrix(matrix))
matrix_tpm <- apply(matrix, 2, function(x) x/sum(as.numeric(x)) * 10^6)
idx <- (log2(rowMeans(matrix_tpm)+1) > 4.)
matrix_tpm <- matrix_tpm[idx,]
matrix_log_tpm <- log2((matrix_tpm/10.)+1)
matrix_log_tpm <- scalop::rowcenter(matrix_log_tpm)
matrix_log_tpm[matrix_log_tpm < 0.] <- 0.

prog.obj = list()
for(sample_id in unique(ad$obs[["sample_id"]])){
    print(sample_id)
    flush.console()
    idx = ad$obs["sample_id"]==sample_id
    input_matrix = matrix_log_tpm[, idx]
    res = scalop::programs(input_matrix)
    if(is.null(res)){
        print(paste0("No differentially expresed groups found for: ", sample_id))
        next
    }
    prog.obj = c(prog.obj, setNames(list(res),sample_id))
}

names(prog.obj) = paste0(names(prog.obj), ".")

tumour_programs = sapply(prog.obj, `[[`, 'programs', simplify = F) %>%
unlist(., recursive = F)

tumour_profiles = sapply(prog.obj, `[[`, 'profiles', simplify = F) %>%
unlist(., recursive = F)

tumour_groups = sapply(prog.obj, `[[`, 'groups', simplify = F) %>%
unlist(., recursive = F)


df = read.csv("/cluster/work/boeva/scRNAdata/annotations/cc_genes_2.csv")
cc_genes = gsub("\\s", "", c(df[, 1], df[, 2]))
cc_genes=cc_genes[cc_genes!=""]
print("Removing high cc signatures")

n_cc_genes = list()
for(name in names(metaprograms)){
    n_cc_genes[[name]] = length(intersect(metaprograms[[name]], cc_genes))
}

if(any(n_cc_genes > 0.25 * lengths(metaprograms))){
    noncc_cluster_names <- names(which.min(n_cc_genes))
    print("Found a meta-signature associated with cell cylce")
    noncc_program_names = clust_2[[noncc_cluster_names]]
}else{
    print("No meta-signature associated with cell cycle found. Retaining all sigantures.")
    noncc_program_names = names(tumour_programs)
}


noncc_programs = tumour_programs[noncc_program_names]
noncc.matrix = Jaccard(noncc_programs)


clust_4 = scalop::hca_groups(noncc.matrix,
cor.method="none",
k=n_clusters,
min.size=0,
max.size=1)

# 6: Define meta-programs from program clusters
# Sort by frequency across programs in cluster (>= 15)
mp_freqs = sapply(clust_4, function(k) sort(table(unlist(noncc_programs[k])),decreasing = T),simplify = F)
metaprograms = sapply(mp_freqs, function(tab) head(names(tab)[tab >= 2], 200), simplify = F)
names(metaprograms) = sprintf("metaprogram%d", 1:n_clusters)

print("Writing metasignatures.")
dir.create(results_dir, recursive = TRUE)
for(i in 1:length(metaprograms)){
    df <- as.data.frame(metaprograms[[i]])
    name <-paste0("/metaprogram", i)
    colnames(df) <- name
    write.table(metaprograms[[i]], paste0(results_dir, name, ".csv"), row.names = FALSE, col.names=FALSE)
}
