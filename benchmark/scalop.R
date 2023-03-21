library("anndata")
library("scalop")

args = commandArgs(trailingOnly=TRUE)

data_path <- args[1]
n_clusters <- strtoi(args[2])
results_dir <- args[3]

ad <- anndata::read_h5ad(data_path)
prog.obj = list()
for(sample_id in unique(ad$obs[["sample_id"]])){
    print(sample_id)
    flush.console()
    matrix = ad[ad$obs["sample_id"]==sample_id]$X
    matrix = t(as.matrix(matrix))
    matrix = apply(matrix, 2, function(x) x/sum(as.numeric(x)) * 10^4)
    if(ncols(matrix)<50){next}
    res = scalop::programs(scalop::rowcenter(matrix))
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
no_cc_programs = c()
for(program_name in names(tumour_programs)){
    if(length(intersect(tumour_programs[[program_name]], cc_genes))<25){
        no_cc_programs = append(no_cc_programs, program_name)
    }
}

noncc_programs = tumour_programs[no_cc_programs]
noncc.matrix = Jaccard(noncc_programs)
noncc.matrix = scalop::hca_reorder(noncc.matrix)

# 5: Retrieve non-cycling program clusters
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
    name <-paste0("metaprogram", i)
    colnames(df) <- name
    write.table(metaprograms[[i]], paste0(results_dir, name, ".csv"), row.names = FALSE, col.names=FALSE)
}
