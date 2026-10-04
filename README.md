# ContextSplice
Data-driven computational model to predict context-dependent splicing

## Install

1. Run "make install" in the repository home after cloning.

or

2. Run the following lines:
```
conda create -n conextsplice python=3.11 -y

conda activate contextsplice

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

pip install lightning

conda install -c conda-forge pandas numpy hydra-core omegaconf wandb gputil matplotlib beartype h5py pytables -y
```
## Datasets
### Pangolin original data
Pangolin_train github repo provides the following intermediate files:
- splice_table_Human.test.txt  
- splice_table_Human.txt  
- splice_table_Macaque.txt  
- splice_table_Mouse.txt  
- splice_table_Rat.txt
- orthologs.txt
- paralogs.txt

These files can be used to generate the original training data instead of repeating all data processing pipelines starting from RNA-seq raw reads by the following steps:

1. Download genome and annotation data by running "make download_pangolin_genomes"

2. Move the files to data/pangolin

3. Ideally, one only needs to run one script: create_files.sh in data/pangolin. Some additional python pacakges may need to be installed. You should get:
- dataset_train_all.h5 (>400G)
- dataset_test_1.h5 (>50G)

Note that some python scripts from Pangolin_train have been modified in order to run in newer python versions.

### EnTEX RNA-seq and DNase-seq data


## Data Processing Methods

### Spliser processing



### Majiq processing

