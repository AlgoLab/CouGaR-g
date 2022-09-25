snakemake -s crossval_nn.smk --config KFOLD=1 -p -c1
snakemake -s crossval_nn.smk --config KFOLD=2 -p -c1
snakemake -s crossval_nn.smk --config KFOLD=3 -p -c1
snakemake -s crossval_nn.smk --config KFOLD=4 -p -c1
snakemake -s crossval_nn.smk --config KFOLD=5 -p -c1