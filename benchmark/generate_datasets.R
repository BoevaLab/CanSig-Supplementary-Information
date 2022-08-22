library("splatter")
library("scater")
library("ggplot2")
library("Matrix")
setwd("/cluster/work/boeva/scRNAdata/benchmark")


counts <- data.matrix(read.csv("~/counts.csv"))
counts <- t(counts)

params <- splatEstimate(counts)


dir.create("raw_datasets", showWarnings = FALSE)
dir.create("raw_datasets_nbe", showWarnings = FALSE)

batch <- 2*c(750, 750, 750, 750, 500, 500, 500, 500, 250, 250, 250, 250, 125, 125)

batchloc <- c(0.125, 0.15, 0.175)
be_level <- c("weak", "medium", "strong")
deprob <- list(c(0.1125, 0.1125, 0.125, 0.15, 0.25), c(0.125, 0.125, 0.15, 0.175, 0.25), c(0.175, 0.175, 0.175, 0.2, 0.25))
defacloc <- list(c(0.15, 0.15, 0.15, 0.175, 0.25), c(0.15, 0.15, 0.17, 0.2, 0.25), c(0.15, 0.15, 0.17, 0.2, 0.25))
dg_level <- c("weak", "medium", "strong")

datasets <- list()
k<-1
for(i in 1:3){
  for(j in 1:3){
    dataset <- list(batchCells=batch, batchloc=batchloc[j], deprob=deprob[[i]], defacloc=defacloc[[i]])

  data_path <- paste0("raw_datasets/be_", be_level[j], "dg_", dg_level[i])
  dir.create(data_path, showWarnings = FALSE)
  params.groups <- setParam(params, "batchCells", dataset$batchCells)
  params.groups <- setParam(params.groups, "nGenes", 4000)
  sim1 <- splatSimulateGroups(params.groups,
                             group.prob = c(0.225, 0.25, 0.275, 0.15, 0.1),
                             batch.facLoc = dataset$batchloc,
                             de.prob = dataset$deprob,
                             de.facLoc =dataset$defacloc,
                             verbose = FALSE)

  if(class(counts(sim1)) == "dgCMatrix"){
    writeMM(counts(sim1), paste0(data_path, "/sim_counts.mtx"))
  }else{
    write.csv(counts(sim1), paste0(data_path, "/sim_counts.csv"))
  }

  write.csv(rowData(sim1), paste0(data_path, "/sim_genes.csv"))
  write.csv(colData(sim1), paste0(data_path, "/sim_cells.csv"))
 }
  data_path_nbe <- paste0("raw_datasets_nbe/", "dg_", dg_level[i])
  dir.create(data_path_nbe, showWarnings = FALSE)
  params.groups <- setParam(params, "batchCells", dataset$batchCells)
  params.groups <- setParam(params.groups, "nGenes", 4000)
  sim2 <- splatSimulateGroups(params.groups,
                             group.prob = c(0.225, 0.25, 0.275, 0.15, 0.1),
                             batch.facLoc = dataset$batchloc,
                             de.prob = dataset$deprob,
                             de.facLoc =dataset$defacloc ,
                             batch.rmEffect = TRUE,
                             verbose = FALSE)

  if(class(counts(sim2)) == "dgCMatrix"){
    writeMM(counts(sim2), paste0(data_path_nbe, "/sim_counts.mtx"))
  }else{
    write.csv(counts(sim2), paste0(data_path_nbe, "/sim_counts.csv"))
  }

  write.csv(rowData(sim2), paste0(data_path_nbe, "/sim_genes.csv"))
  write.csv(colData(sim2), paste0(data_path_nbe, "/sim_cells.csv"))

}
