export DATA="data/data.h5ad"
export OUTPUTDIR="run-results"
export SCRIPT="scripts/run.py"


# Run Scanorama with different number of principal components
for LATENT in 5 10 50;
do
	python $SCRIPT $DATA --output-dir $OUTPUTDIR --method scanorama --latent $LATENT
done

# Run BBKNN
python $SCRIPT $DATA --output-dir $OUTPUTDIR --method bbknn


# Run scVI
for LATENT in 5 10;
do
	for LAYER in 1 3;
	do
		python $SCRIPT $DATA --output-dir $OUTPUTDIR --method scvi --latent $LATENT --scvi-layers $LAYER
	done
done

