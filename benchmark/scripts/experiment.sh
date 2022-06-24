export DATA="data/data.h5ad"
export OUTPUTDIR="benchmark-results"
export SCRIPT="scripts/run.py"


# Run Scanorama with different number of principal components
for LATENT in 2 3 5 10 20 50;
do
	python $SCRIPT $DATA --output-dir $OUTPUTDIR --method scanorama --latent $LATENT
done

# Run BBKNN
# For now we don't use BBKNN as it returns distance matrix rather than
# latent representations.
# python $SCRIPT $DATA --output-dir $OUTPUTDIR --method bbknn


# Run scVI and CanSig
for LATENT in 3 5 10;
do
	for LAYER in 1 2 3;
	do
		python $SCRIPT $DATA --output-dir $OUTPUTDIR --method scvi --latent $LATENT --scvi-layers $LAYER
		python $SCRIPT $DATA --output-dir $OUTPUTDIR --method cansig --latent $LATENT --scvi-layers $LAYER
	done
done
