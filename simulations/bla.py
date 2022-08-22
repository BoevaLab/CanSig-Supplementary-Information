from simul.cna.profiles import Genome
import pandas as pd
import infercnvpy



gene_df = pd.read_csv("~/BoevaLab/annotations/gene_order_all.csv")


Genome(gene_df)
