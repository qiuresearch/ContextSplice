.ONESHELL:
SHELL := /bin/bash
.DEFAULT_GOAL := help
# .PHONY: all help clean test_pangolin test_human pangolin_test
# This tells make to always run the recipe for the requested targets,
# regardless of whether a file with that name exists, thus no needs for .PHONY declaration.
MAKEFLAGS += --always-make

# Use ENV_VAR if set, otherwise default to "default_value"
CONDA_PREFIX?=$(HOME)/miniconda3
CONDA_SH_PATH?=$(CONDA_PREFIX)/etc/profile.d/conda.sh
CONDA_ENV_NAME?=contextsplice

gpus=0
debug=false
wandb=online
train_len=null
val_len=null
test_len=null
predict_len=200000

sbatch=false
partition=small-gpu
time=7-00:00:00

help: ## Display this help message
	@echo "Usage: make <target>"
	@echo
	@echo "Available targets:"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-42s\033[0m %s\n", $$1, $$2}'

sbatch_redirect: ## Redirect the action to sbatch instead of interactive running
	@if [ -n "$${SLURM_JOB_ID}" ] ; then exit 0 ; fi
	@GOALOPT="gpus=$(gpus) train_len=$(train_len) val_len=$(val_len) test_len=$(test_len) predict_len=$(predict_len) debug=$(debug)"
	@if [ "$(sbatch)" = "true" ] ; then \
		sbatch_file=sbatch ; sbatch_cmds=(echo Starting...) ; \
		for goal in $(MAKECMDGOALS) ; do sbatch_file=$${sbatch_file}_$${goal} ; sbatch_cmds+=(\; make $${goal} $${GOALOPT}) ; done ; \
		sbatch_brew.sh -p $(partition) -t $(time) -o "$${sbatch_file}.sh" "$${sbatch_cmds[*]}" ; \
		echo "Submitting via sbatch ..." ; \
		sbatch "$${sbatch_file}.sh" ; \
		exit 1 ; \
	fi

install: ## Install python dependencies under conda environment
	source $(CONDA_SH_PATH)
	conda create -n $(CONDA_ENV_NAME) python=3.13 -y
	conda activate $(CONDA_ENV_NAME)
	# Install PyTorch with CUDA 12.8 support
	pip install torch==2.7.1 torchvision==0.22.0 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
	pip install lightning
	pip install lightning[extra]
	conda install -c conda-forge pandas numpy hydra-core omegaconf wandb gputil matplotlib beartype h5py pytables -y
	conda install mappy -c bioconda -y

test_pangolinomni3_pangolinsolo123: ## Evaluate PangolinOmni3 model trained on PangolinSolo123 dataset
	ckpt_dir=checkpoints/pangolinomni3.pangolinsolo123_2025-11-11_00-04-02_97d417l3
	ckpt_dir=checkpoints/pangolinomni3.pangolinsolo123_2025-11-19_20-34-04_pwa6rdke
	ckpt_files=($$(ls $${ckpt_dir}/valoss*.ckpt))
	ckpt_path=$${ckpt_files[0]}
	tissues=(liver)
	tissues=(heart liver brain testis)
	tissues=(heart liver brain testis "heart,liver,brain,testis")	
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinomni3 \
			model.state_dict_path=null \
			litrun.resume_from_ckpt=$${ckpt_path} \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_epoch_length=$(predict_len) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			+dataset.tissue_embedding_path="data/tissue_avg_pca_embeddings.csv" \
			debug=$(debug)
		wait
	done

test_pangolinomni2_pangolinsolo123: ## Evaluate PangolinOmni2 model trained on PangolinSolo123 dataset
	ckpt_dir=checkpoints/pangolinomni2.pangolinsolo123_2025-10-27_07-27-47_933otwx4
	ckpt_path="$${ckpt_dir}/best.ckpt"
	tissues=(liver)
	tissues=(heart liver brain testis)
	tissues=(heart liver brain testis "heart,liver,brain,testis")	
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinomni2 \
			model.state_dict_path=null \
			litrun.resume_from_ckpt="$${ckpt_path}" \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_epoch_length=$(predict_len) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			+dataset.tissue_embedding_path="data/tissue_avg_pca_embeddings.csv" \
			debug=$(debug)
		wait
	done

test_pangolinomni_pangolinsolo123: ## Evaluate PangolinOmni model trained on PangolinSolo123 dataset
	ckpt_dir=checkpoints/pangolinomni.pangolinsolo123_2025-10-21_23-25-37_bsi1y0zt
	tissues=(liver)
	tissues=("heart,liver,brain,testis")	
	tissues=(heart liver brain testis "heart,liver,brain,testis")
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinomni \
			model.state_dict_path=null \
			litrun.resume_from_ckpt=$${ckpt_dir}/best.ckpt \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_epoch_length=$(predict_len) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			+dataset.tissue_embedding_path="data/tissue_avg_pca_embeddings.csv" \
			debug=$(debug)
		wait
	done

test_pangolinsolo_pangolinsolo4: ## Evaluate PangolinSolo model trained on PangolinSolo testis dataset (single tissue input and output)
	ckpt_dir=checkpoints/pangolinsolo.pangolinsolo4_2025-10-27_16-15-32_8l8omvzn
	ckpt_path="$${ckpt_dir}/best.ckpt"
	tissues=(liver)
	tissues=(heart liver brain testis "heart,liver,brain,testis")
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinsolo \
			model.state_dict_path=null \
			litrun.resume_from_ckpt=$${ckpt_path} \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_epoch_length=$(predict_len) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			debug=$(debug)
		wait
	done

test_pangolinsolo_pangolinsolo3: ## Evaluate PangolinSolo model trained on PangolinSolo testis dataset (single tissue input and output)
	ckpt_dir=checkpoints/pangolinsolo.pangolinsolo3_2025-10-29_03-43-28_l9t6a1nc
	ckpt_path="$${ckpt_dir}/best.ckpt"
	tissues=(liver)
	tissues=(heart liver brain testis "heart,liver,brain,testis")
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinsolo \
			model.state_dict_path=null \
			litrun.resume_from_ckpt=$${ckpt_path} \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_epoch_length=$(predict_len) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			debug=$(debug)
		wait
	done

test_pangolinsolo_pangolinsolo2: ## Evaluate PangolinSolo model trained on PangolinSolo liver dataset (single tissue input and output)
	ckpt_dir=checkpoints/pangolinsolo.pangolinsolo2_2025-10-29_05-02-24_4wltwlym
	ckpt_path="$${ckpt_dir}/best.ckpt"
	tissues=(liver)
	tissues=(heart liver brain testis "heart,liver,brain,testis")
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinsolo \
			model.state_dict_path=null \
			litrun.resume_from_ckpt=$${ckpt_path} \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_epoch_length=$(predict_len) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			debug=$(debug)
		wait
	done

test_pangolinsolo_pangolinsolo1: ## Evaluate PangolinSolo model trained on PangolinSolo heart dataset (single tissue input and output)
# 	ckpt_dir=checkpoints/pangolinsolo_pangolin_2025-10-18_12-41-38_btn1jd24
	ckpt_dir=checkpoints/pangolinsolo.pangolinsolo1_2025-10-25_11-46-07_cqw6jfnv
	ckpt_path="$${ckpt_dir}/best.ckpt"
	tissues=(liver)
	tissues=(heart liver brain testis "heart,liver,brain,testis")
	for ((i=0; i<$${#tissues[@]}; i++)) ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run.py stage=eval \
			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-} \
			infer.save_level=2 infer.eval_dim=null \
			model.type=pangolinsolo \
			model.state_dict_path=null \
			litrun.resume_from_ckpt=$${ckpt_path} \
			dataset.type=pangolinsolo \
			dataset.train_path=null \
			dataset.predict_epoch_length=$(predict_len) \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			+dataset.tissue_types=[$${tissues[i]}] \
			debug=$(debug)
		wait
	done	
# 	ckpt_names=($$(ls $${ckpt_dir}/epoch*.ckpt))
# 	echo "Available checkpoints:"
# 	echo "$${ckpt_names[@]}"
# 	tissues=(heart liver brain testis)
# 	tissues+=("heart,liver,brain,testis")
# 	tissues=(heart)
# 	for ((i=0; i<$${#tissues[@]}; i++)) ; do
# 		for ((j=0; j<$${#ckpt_names[@]}; j++)) ; do
# 			epoch=$$(echo $${ckpt_names[j]} | rev | cut -d'-' -f3 | rev)
# 			echo "Evaluating checkpoint: $${ckpt_names[j]} (epoch: $${epoch})"
# 		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
# 		python -u run.py stage=eval \
# 			infer.save_prefix=$${ckpt_dir}/pangolin_test_$${tissues[i]//,/-}_epoch-$${epoch} \
# 			infer.save_level=2 infer.eval_dim=null \
# 			model.type=pangolinsolo \
# 			model.state_dict_path=null \
# 			litrun.resume_from_ckpt=$${ckpt_names[j]} \
# 			dataset.type=pangolinsolo \
# 			dataset.train_path=null \
# 			dataset.predict_epoch_length=$(predict_len) \
# 			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
# 			+dataset.tissue_types=[$${tissues[i]}] \
# 			debug=$(debug)
# 		wait
# 		done
# 	done

test_pangolinorig_ensemble_pangolin: ## Evaluate Pangolin original ensemble average
	# Run Pangolin model (final.modelnum.tissue.epoch) on Pangolin dataset
	model_dir=benchmarks/pangolin_models
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=eval \
	    infer.save_prefix=benchmarks/pangolin_ensemble/pangolin_test \
		infer.save_level=2 infer.eval_dim=-2 \
		model.type=pangolin \
		model.state_dict_path=null \
		litrun.resume_from_ckpt=null \
		ensemble.enable=true \
		ensemble.model.type=[pangolin,pangolin,pangolin] \
		ensemble.model.state_dict_path=[$${model_dir}/final.1.0.3,$${model_dir}/final.2.0.3,$${model_dir}/final.3.0.3,$${model_dir}/final.4.0.3,$${model_dir}/final.5.0.3] \
	    dataset.type=pangolin \
		dataset.predict_path=data/pangolin/dataset_test_1.h5 \
		dataset.predict_epoch_length=$(predict_len) \
		+dataset.tissue_types=[heart,liver,brain,testis] \
		debug=$(debug)

test_pangolin_pangolin: ## Evaluate Pangolin model trained on Pangolin dataset (quad inputs and outputs)
	ckpt_dir="checkpoints/pangolin.pangolin_2025-10-21_22-16-17_mnsgbck4"
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=eval \
		infer.save_prefix=$${ckpt_dir}/pangolin_test \
		infer.save_level=2 infer.eval_dim=-2 \
		model.type=pangolin \
		model.state_dict_path=null \
		litrun.resume_from_ckpt=$${ckpt_dir}/best.ckpt \
		dataset.type=pangolin \
		dataset.train_path=null \
		dataset.predict_epoch_length=$(predict_len) \
		dataset.predict_path=data/pangolin/dataset_test_1.h5 \
		+dataset.tissue_types=[heart,liver,brain,testis] \
		debug=$(debug)

test_pangolinorig_pangolin: ## Evaluate Pangolin original model
	# Run Pangolin model (final.modelnum.tissue.epoch) on Pangolin dataset
	#	dataset.predict_path=data/pangolin/dataset_train_all.h5 \
	models=(final.1.3.3 final.2.3.3 final.3.3.3 final.4.3.3 final.5.3.3 final.1.3.3.v2 final.2.3.3.v2 final.3.3.3.v2)
	models+=(final.1.4.3 final.2.4.3 final.3.4.3 final.4.4.3 final.5.4.3 final.1.4.3.v2 final.2.4.3.v2 final.3.4.3.v2)
	models=(final.1.0.3 final.2.0.3 final.3.0.3 final.4.0.3 final.5.0.3 final.1.0.3.v2 final.2.0.3.v2 final.3.0.3.v2)
# 	models=(final.1.1.3 final.2.1.3 final.3.1.3 final.4.1.3 final.5.1.3 final.1.1.3.v2 final.2.1.3.v2 final.3.1.3.v2)
# 	models+=(final.1.2.3 final.2.2.3 final.3.2.3 final.4.2.3 final.5.2.3 final.1.2.3.v2 final.2.2.3.v2 final.3.2.3.v2)	
	for model in $${models[@]} ; do
		conda run --no-capture-output --name $(CONDA_ENV_NAME) \
		python -u run.py stage=eval \
			infer.save_prefix=benchmarks/pangolin_$${model}/pangolin_test \
			infer.save_level=2 infer.eval_dim=-2 \
			model.type=pangolin \
			model.state_dict_path=benchmarks/pangolin_models/$${model} \
			litrun.resume_from_ckpt=null \
			ensemble.enable=false \
			dataset.type=pangolin \
			dataset.predict_path=data/pangolin/dataset_test_1.h5 \
			dataset.predict_epoch_length=$(predict_len) \
			+dataset.tissue_types=[heart,liver,brain,testis] \
			debug=$(debug)
	done

train_pangolinsolo_pangolinsolo123: ## Train PangolinSolo model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=train gpus=[$(gpus)] \
		run_name=pangolinsolo.pangolinsolo123 \
		model.type=pangolinsolo \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		+dataset.tissue_types=[heart,liver,brain] \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		datamodule.train_batch_size=96 \
		datamodule.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_pangolinsolo_pangolinsolo124: ## Train PangolinSolo model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=train gpus=[$(gpus)] \
		run_name=pangolinsolo.pangolinsolo124 \
		model.type=pangolinsolo \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		+dataset.tissue_types=[heart,liver,testis] \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		datamodule.train_batch_size=96 \
		datamodule.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_pangolinsolo_pangolinsolo1: ## Train PangolinSolo model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=train gpus=[$(gpus)] \
		run_name=pangolinsolo.pangolinsolo1 \
		model.type=pangolinsolo \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		+dataset.tissue_types=[heart] \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		datamodule.train_batch_size=96 \
		datamodule.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_pangolinsolo_pangolinsolo2: ## Train PangolinSolo model on PangolinSolo 2 dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=train gpus=[$(gpus)] \
		run_name=pangolinsolo.pangolinsolo2 \
		model.type=pangolinsolo \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		+dataset.tissue_types=[liver] \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		datamodule.train_batch_size=96 \
		datamodule.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_pangolinsolo_pangolinsolo3: ## Train PangolinSolo model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=train gpus=[$(gpus)] \
		run_name=pangolinsolo.pangolinsolo3 \
		model.type=pangolinsolo \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		+dataset.tissue_types=[brain] \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		datamodule.train_batch_size=96 \
		datamodule.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_pangolinsolo_pangolinsolo4: ## Train PangolinSolo model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=train gpus=[$(gpus)] \
		run_name=pangolinsolo.pangolinsolo4 \
		model.type=pangolinsolo \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		+dataset.tissue_types=[testis] \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		datamodule.train_batch_size=96 \
		datamodule.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)
		
train_pangolinomni_pangolinsolo123: ## Train PangolinOmni model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=train gpus=[$(gpus)] \
		run_name=pangolinomni.pangolinsolo123 \
		model.type=pangolinomni \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		+dataset.tissue_types=[heart,liver,brain] \
		+dataset.tissue_embedding_path="data/tissue_avg_pca_embeddings.csv" \
		datamodule.train_batch_size=96 \
		datamodule.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		litrun.wandb.mode=$(wandb) \
		debug=$(debug)

train_pangolinomni2_pangolinsolo123: sbatch_redirect ## Train PangolinOmni2 model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=train gpus=[$(gpus)] \
		run_name=pangolinomni2.pangolinsolo123 \
		model.type=pangolinomni2 \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		+dataset.tissue_types=[heart,liver,brain] \
		+dataset.tissue_embedding_path="data/tissue_avg_pca_embeddings.csv" \
		datamodule.train_batch_size=96 \
		datamodule.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_pangolinomni3_pangolinsolo123: sbatch_redirect ## Train PangolinOmni3 model on PangolinSolo dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=train gpus=[$(gpus)] \
		run_name=pangolinomni3.pangolinsolo123 \
		model.type=pangolinomni3 \
		model.state_dict_path=null \
		dataset.type=pangolinsolo \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		dataset.train_epoch_length=$(train_len) \
		dataset.val_epoch_length=$(val_len) \
		dataset.test_epoch_length=$(test_len) \
		dataset.predict_path=null \
		dataset.predict_epoch_length=$(predict_len) \
		+dataset.tissue_types=[heart,liver,brain] \
		+dataset.tissue_embedding_path="data/tissue_avg_pca_embeddings.csv" \
		datamodule.train_batch_size=96 \
		datamodule.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)		

train_pangolin_pangolin: sbatch_redirect ## Train Pangolin model on Pangolin dataset (all four tissues)
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=train gpus=[$(gpus)] \
		model.type=pangolin \
		model.state_dict_path=null \
		dataset.type=pangolin \
		dataset.train_path=data/pangolin/dataset_train_all.h5 \
		+dataset.tissue_types=[heart,liver,brain,testis] \
		datamodule.train_batch_size=96 \
		datamodule.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_splicesolo_genecrops: sbatch_redirect ## Train SpliceSolo on genecrops dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=train gpus=[$(gpus)] \
		model.type=SpliceSolo \
		model.state_dict_path=null \
		dataset.type=genecrops \
		dataset.train_path=data/gwsplice_genecrops.h5 \
		datamodule.train_batch_size=96 \
		datamodule.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)

train_splicesolo_genesites: sbatch_redirect ## Train SpliceSolo on gene sites dataset
	conda run --no-capture-output --name $(CONDA_ENV_NAME) \
	python -u run.py stage=train gpus=[$(gpus)] \
		model.type=splicesolo \
		model.state_dict_path=null \
		dataset.type=genesites \
		dataset.train_path=data/gwsplice_genesites.pkl \
		+dataset.stratified_sampling=chrom \
		+dataset.weighted_sampling=len \
		+dataset.dynamic_weights=false \
		datamodule.train_batch_size=96 \
		datamodule.val_batch_size=128 \
		litrun.resume_from_ckpt=null \
		debug=$(debug)