import os
import h5py
import tables
import torch
from torch.utils.data import Dataset
from omegaconf.listconfig import ListConfig
import numpy as np
import pandas as pd

from . import splice_feats
from . import utils

ilogger = utils.get_pylogger(__name__)


def get_pangolin_tissue_cols(tissue_types):
    # tissue order: heart, liver, brain, testis
    assert (
        isinstance(tissue_types, (list, tuple, ListConfig)) and len(tissue_types) > 0
    ), f"tissue_types: {tissue_types} must be a list or tuple with at least one element."
    tissue_types = [t.lower() if isinstance(t, str) else t for t in tissue_types]
    # starting column for each tissue in Y
    col_map = {
        "heart": 0,
        "liver": 3,
        "brain": 6,
        "testis": 9,
        1: 0,
        2: 3,
        3: 6,
        4: 9,
    }

    tissue_cols = []
    for t in tissue_types:
        if t in col_map:
            tissue_cols.append(col_map[t])
        else:
            raise ValueError(
                f"Invalid tissue_type: {t}. Must be one of {list(col_map.keys())}."
            )
    return tissue_cols


class PangolinSoloDataset(Dataset):
    """Adapted from Pangolin github repo
    Returns sample_feat containing only one tissue type per call, which is randomly picked
    based on tissue_types list.
    """

    def __init__(
        self,
        file_path,
        tissue_types=["heart"],
        tissue_embedding_path=None,
        epoch_length=None,
        **kwargs,
    ):
        super(PangolinSoloDataset, self).__init__()
        self.file_path = file_path
        self.data = h5py.File(file_path, "r", libver="latest")
        self.data_length = len(self.data) // 3
        assert len(self.data) % 3 == 0
        if epoch_length:
            self.epoch_length = epoch_length
        else:
            self.epoch_length = self.data_length * len(tissue_types)

        self.tissue_cols = get_pangolin_tissue_cols(tissue_types)
        self.tissue_types = [t.lower() for t in tissue_types]

        if tissue_embedding_path:
            ilogger.info(f"Loading tissue embeddings from {tissue_embedding_path}...")
            self.tissue_embeddings = pd.read_csv(tissue_embedding_path, index_col=0)
            self.tissue_embeddings.index = [
                str(idx).lower() for idx in self.tissue_embeddings.index
            ]

            for t in self.tissue_types:
                if t not in self.tissue_embeddings.index:
                    raise ValueError(
                        f"Tissue type {t} not found in tissue embeddings index."
                    )
        else:
            self.tissue_embeddings = None

        self.iterations = 0

    def __getitem__(self, idx):
        self.iterations += 1
        if self.iterations % 100000 == 0:  # workaround to avoid memory issues
            self.data.close()
            self.data = h5py.File(self.file_path, "r", libver="latest")

        if len(self.tissue_cols) > 1:  # multiple tissues, adjust idx accordingly
            idx = idx // len(self.tissue_cols)
            itissue = idx % len(self.tissue_cols)
        else:
            itissue = 0

        if idx >= self.data_length:
            ilogger.warning(f"idx: {idx} exceeds data length: {self.data_length}!")
            idx = idx % self.data_length

        tissue_col = self.tissue_cols[itissue]

        X = self.data[f"X{idx}"][:].T  # transpose to (4, L)
        Y = self.data[f"Y{idx}"][:].T  # transpose to (12, L)
        Z = self.data[f"Z{idx}"][:]  # (4,): [chrom, start, end, strand]

        # each tissue: unspliced, spliced, usage
        sample_feat = {
            "seq": splice_feats.decode_onehot(
                X, dim=0, idx2base=np.array(["A", "C", "G", "T", "N"])
            ),
            "seq_onehot": X.astype(np.float32),  # (4, L)
            "cls": np.argmax(
                Y[tissue_col : tissue_col + 2, :], axis=0
            ),  # 0: unspliced, 1: spliced (L)
            "psi": Y[tissue_col + 2, :].astype(np.float32),  # usage (L)
            "chrom": Z[0].decode(),
            "start": int(Z[1]),
            "end": int(Z[2]),
            "strand": Z[3].decode(),
        }

        if self.tissue_embeddings is not None:
            tissue_type = self.tissue_types[itissue]
            tissue_embedding = self.tissue_embeddings.loc[tissue_type].values.astype(
                np.float32
            )

            sample_feat["seq_onehot"] = np.concatenate(
                [
                    sample_feat["seq_onehot"],
                    np.tile(
                        tissue_embedding[:, np.newaxis],
                        (1, sample_feat["seq_onehot"].shape[1]),
                    ),
                ],
                axis=0,
            )

        return sample_feat

    def __len__(self):
        return self.epoch_length

    def __del__(self):
        try:
            self.data.close()
        except:
            ilogger.warning("Failed to close HDF5 file in PangolinSoloDataset.")


class PangolinDataset(Dataset):
    """Adapted from Pangolin github repo"""

    def __init__(
        self,
        file_path,
        tissue_types=["heart", "liver", "brain", "testis"],
        epoch_length=None,
        **kwargs,
    ):
        super(PangolinDataset, self).__init__()
        self.file_path = file_path
        self.data = h5py.File(file_path, "r", libver="latest")
        self.data_length = len(self.data) // 3
        assert len(self.data) % 3 == 0

        if epoch_length:
            self.epoch_length = epoch_length
        else:
            self.epoch_length = self.data_length

        self.tissue_cols = get_pangolin_tissue_cols(tissue_types)
        self.iterations = 0

    def __getitem__(self, idx):
        self.iterations += 1
        if self.iterations % 100000 == 0:  # workaround to avoid memory issues
            self.data.close()
            self.data = h5py.File(self.file_path, "r", libver="latest")

        if idx >= self.data_length:
            ilogger.warning(f"idx: {idx} exceeds data length: {self.data_length}!")
            idx = idx % self.data_length

        X = self.data[f"X{idx}"][:].T  # transpose to (4, L)
        Y = self.data[f"Y{idx}"][:].T  # transpose to (12, L)
        Z = self.data[f"Z{idx}"][:]  # (4,): [chrom, start, end, strand]

        # for each tissue: unspliced, spliced, usage
        cls_list, psi_list = [], []
        for tissue_col in self.tissue_cols:
            cls_list.append(
                np.argmax(Y[tissue_col : tissue_col + 2, :], axis=0)
            )  # 0: unspliced, 1: spliced
            psi_list.append(Y[tissue_col + 2, :])  # usage

        sample_feat = {
            "seq": splice_feats.decode_onehot(
                X, dim=0, idx2base=np.array(["A", "C", "G", "T", "N"])
            ),
            "seq_onehot": X.astype(np.float32),
            "cls": np.stack(cls_list, axis=0).astype(np.int64),  # (num_tissues, L)
            "psi": np.stack(psi_list, axis=0).astype(np.float32),  # (num_tissues, L)
            "chrom": Z[0].decode(),
            "start": int(Z[1]),
            "end": int(Z[2]),
            "strand": Z[3].decode(),
        }

        return sample_feat

    def __len__(self):
        return self.epoch_length

    def __del__(self):
        try:
            self.data.close()
        except:
            ilogger.warning("Failed to close HDF5 file in PangolinDataset.")


class GeneCropsDataset(Dataset):
    def __init__(self, data, file_path=None, indices=None, num_classes=None, **kwargs):
        super(GeneCropsDataset, self).__init__()

        self.data = data
        if isinstance(self.data, str):
            if not os.path.exists(self.data):
                raise FileNotFoundError(f"Data file {self.data} does not exist.")
            self.file_path = self.data
            self.data = tables.open_file(self.file_path, mode="r")
        elif self.data is None:  # load from file_path
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file {file_path} does not exist.")
            self.file_path = file_path
            self.data = tables.open_file(file_path, mode="r")
        elif isinstance(self.data, tables.File):
            self.file_path = file_path
        else:
            raise ValueError(
                "data must be a path to HDF5 file or a tables.File handle."
            )

        self.indices = indices
        self.num_classes = num_classes
        self.iterations = 0

    def __getitem__(self, idx):
        self.iterations += 1

        if self.iterations % 10000 == -1 and os.path.exists(
            self.file_path
        ):  # workaround to avoid memory issues
            self.data.close()
            self.data = tables.open_file(self.file_path, mode="r")

        if self.indices is not None:
            if idx < 0 or idx >= len(self.indices):
                raise IndexError(
                    f"Index:{idx} out of bounds for SeqCropsDataset (len(indices): {len(self.indices)})."
                )
            idx = self.indices[idx]

        if idx < 0 or idx >= len(self.data.root.crop_feats):
            raise IndexError(
                f"Index:{idx} out of bounds for SeqCropsDataset (len(data): {len(self.data.root.crop_feats)})."
            )

        crop_feat = self.data.root.crop_feats[idx]
        # Change dtypes as needed because of low-precision storage in HDF5 if possible
        sample_feat = {
            "seq": crop_feat["seq"],  # .decode(), # strings are saved in bytes in HDF5
            "seq_onehot": crop_feat["seq_onehot"].astype(np.float32),
            "cls": crop_feat["cls"].astype(np.int64),
            "psi": crop_feat["psi"],  # usage
            "chrom": crop_feat["chrom"].decode(),
            "gene_id": crop_feat["gene_id"].decode(),
            "gene_start": crop_feat["gene_start"][0],
            "gene_end": crop_feat["gene_end"][0],
            "strand": crop_feat["strand"].decode(),
            "start": crop_feat["crop_start"][0],
            "end": crop_feat["crop_end"][0],
        }

        if self.num_classes:
            sample_feat["cls"] = np.minimum(crop_feat["cls"], self.num_classes - 1)

        return sample_feat

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(self.data.root.crop_feats)

    def __del__(self):
        try:
            self.data.close()
        except:
            ilogger.warning("Failed to close HDF5 file in SeqCropsDataset.")


def get_gtex_tissue_cols(tissue_types):
    # tissue order: heart, liver, brain, testis
    assert (
        isinstance(tissue_types, (list, tuple, ListConfig)) and len(tissue_types) > 0
    ), f"tissue_types: {tissue_types} must be a list or tuple with at least one element."
    tissue_types = [t.lower() if isinstance(t, str) else t for t in tissue_types]

    gtex_tissues = [
        "Adipose Tissue",
        "Adrenal Gland",
        "Bladder",
        "Blood",
        "Blood Vessel",
        "Bone Marrow",
        "Brain",
        "Breast",
        "Cervix Uteri",
        "Colon",
        "Esophagus",
        "Fallopian Tube",
        "Heart",
        "Kidney",
        "Liver",
        "Lung",
        "Muscle",
        "Nerve",
        "Ovary",
        "Pancreas",
        "Pituitary",
        "Prostate",
        "Salivary Gland",
        "Skin",
        "Small Intestine",
        "Spleen",
        "Stomach",
        "Testis",
        "Thyroid",
        "Uterus",
        "Vagina",
    ]

    # starting column for each tissue in GTEx dataset 'cls' and 'psi' arrays
    col_map = {}
    for i, tissue in enumerate(gtex_tissues):
        col_map[tissue.replace(" ", "").lower()] = i

    tissue_cols = []
    for t in tissue_types:
        t = t.replace(" ", "").lower()
        if t in col_map:
            tissue_cols.append(col_map[t])
        else:
            raise ValueError(
                f"Invalid tissue_type: {t}. Must be one of {list(col_map.keys())}."
            )
    return tissue_cols


class GeneSitesDataset(Dataset):
    def __init__(
        self,
        genesites_df,
        num_classes=2,
        epoch_length=None,
        tissue_types=None,
        dataset_name=None,
        pipe_genecrops=False, # whether to exhaust all gene crops before sampling the next gene
        weighted_sampling='gene_span',
        dynamic_weights=False,
        min_quality=None,
        stratified_sampling=None,
        sampling_method=None,
        random_seed=42,
        training=True,
        summarize=False,
        **kwargs,
    ):
        """
        PyTorch Dataset for RNA splice sites at the gene level.

        Args:
            genesites_df (pd.DataFrame or str): Genesites DataFrame or path to CSV/pickle/parquet file.
            epoch_length (int, optional): Number of samples per epoch. Defaults to length of genesites_df.
            data_dir (str): Directory containing RNA structure files.
            pipe_genecrops (bool): Whether to exhaust all gene crops before sampling the next gene.
            weighted_sampling (str): Sampling strategy ('gene_span' or 'uniform').
            dynamic_weights (bool): Update weights dynamically during training.
            min_quality (float): Minimum quality threshold for samples.
            stratified_sampling (str, optional): Column name for stratified sampling. If None, no stratification.
            sampling_method (str, optional): Sampling method ('random' or 'sequential'). Not implemented.
            dataset_name (str, optional): Name of the dataset.
            seed_start (int): Starting seed for random sampling.
            training (bool): Whether the dataset is used for training.
            samples_per_seq (int): Number of samples to generate per sequence.
            **kwargs: Additional keyword arguments.
        """
        super(GeneSitesDataset, self).__init__()

        self.is_child_process = utils.get_rank()  # utils.is_child_process()

        if isinstance(genesites_df, str):
            if not os.path.exists(genesites_df):
                raise FileNotFoundError(f"Metadata file {genesites_df} does not exist.")
            if genesites_df.endswith("parquet"):
                genesites_df = pd.read_parquet(genesites_df)
            elif genesites_df.endswith("csv"):
                genesites_df = pd.read_csv(genesites_df)
            elif genesites_df.endswith("pkl"):
                genesites_df = pd.read_pickle(genesites_df)
            else:
                raise ValueError(f"Unsupported file format: {genesites_df}")
        elif not isinstance(genesites_df, pd.DataFrame):
            raise ValueError("genesites_df must be a pandas DataFrame or file name.")

        self.genesites_df = genesites_df.reset_index(drop=True)
        self.num_classes = num_classes
        self.epoch_length = len(genesites_df) if epoch_length is None else epoch_length
        self.random_seed = random_seed
        self.iterations = 0

        self.genecrops_cache = [] if pipe_genecrops else None
        self.dataset_name = dataset_name

        self.tissue_types = tissue_types
        if self.tissue_types is not None:
            self.tissue_cols = get_gtex_tissue_cols(self.tissue_types)
            if len(self.tissue_cols) == 0:
                ilogger.warning(f"tissue_types: {self.tissue_types} must contain at least one valid tissue type.")
                self.tissue_cols = None
            elif len(self.tissue_cols) > 1:
                ilogger.warning(f"Multiple tissue types specified: {self.tissue_types}. Using the first tissue type for sampling.")
                self.tissue_cols = self.tissue_cols[0]  # Use the first tissue column for sampling
            else:
                self.tissue_cols = self.tissue_cols[0]  # only one tissue type, use its column index

        self.training = training
        self.min_quality = min_quality

        # Set up customized sampling options
        if stratified_sampling:
            ilogger.info(
                f"Setup stratified sampling based on column: {stratified_sampling}"
            )
            meta_groups = self.genesites_df.groupby(stratified_sampling)
            if meta_groups.ngroups <= 1:
                raise ValueError(
                    f"Only one group found for stratified_sampling: {stratified_sampling}. Check your meta_df."
                )

            self.stratified_sampling = True
            group_indices = meta_groups.indices
            self.stratified_ngroups = len(group_indices)
            self.stratified_names = list(group_indices.keys())
            self.stratified_ilocs = list(group_indices.values())
            ilogger.info(
                f"Found {self.stratified_ngroups} groups for stratified sampling, e.g., {self.stratified_names[:5]}..."
            )
        else:
            self.stratified_sampling = False
            self.all_ilocs = np.arange(len(genesites_df))

        self.weighted_sampling = weighted_sampling
        if self.weighted_sampling:
            if self.weighted_sampling not in genesites_df.columns:
                raise ValueError(
                    "weighted_sampling is True but column <{}> is missing in meta_df.".format(
                        self.weighted_sampling
                    )
                )
            self.iweight = self.genesites_df.columns.get_loc(self.weighted_sampling)
            self.get_normalized_sampling_weights()
        else:
            self.iweight = None  # Default to None if not present

        self.dynamic_weights = dynamic_weights

        self.sampling_method = sampling_method  # not yet implemented
        if sampling_method:
            if sampling_method not in ["random", "sequential"]:
                raise ValueError("sampling_method must be 'random' or 'sequential'.")

        self.customized_sampling = (
            self.stratified_sampling or self.sampling_method or self.weighted_sampling
        )

        if not self.customized_sampling and self.epoch_length > len(genesites_df):
            ilogger.warning(
                f"epoch_size {self.epoch_length} is greater than sample_size {len(genesites_df)}, adjusting epoch_size."
            )
            self.epoch_length = len(genesites_df)

        if summarize and not self.is_child_process:
            print(f"SpliceDataset summary:")
            print(f"               training: {self.training}")
            print(f"           len(meta_df): {len(self.genesites_df)}")
            print(f"             epoch_size: {self.epoch_length}")
            print(f"            min_quality: {self.min_quality}")
            print(f"    customized_sampling: {self.customized_sampling}")
            print(f"    stratified_sampling: {self.stratified_sampling}")
            if self.stratified_sampling:
                print(
                    f"     stratified_ngroups: {self.stratified_ngroups}, e.g., {self.stratified_names[:5]}..."
                )
            print(f"      weighted_sampling: {self.weighted_sampling}")
            print(f"        dynamic_weights: {self.dynamic_weights}")
            # print(f"    sampling_method: {self.sampling_method if self.sampling_method else 'None'}")

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):

        if idx < 0 or idx >= self.epoch_length:
            raise IndexError(f"Index:{idx} out of bounds for GeneSitesDataset.")

        sample_feat = None
        while sample_feat is None:
            sample_feat = self.omni_sampler(idx)

        return sample_feat

    def get_normalized_sampling_weights(self):
        """Get normalized sampling weights for stratified or all indices

        Yields:
            self.stratified_sampling_weights: list of np.arrays for each stratified group
            self.sampling_weights: np.array for all indices
        """

        if self.stratified_sampling:
            self.stratified_sampling_weights = []
            for i, ilocs in enumerate(self.stratified_ilocs):
                weights = self.genesites_df.iloc[ilocs, self.iweight].values
                total_weight = weights.sum()
                if total_weight == 0:
                    ilogger.warning(
                        f"Total sampling weight is zero for stratified group: {self.stratified_names[i]}!"
                    )
                    self.stratified_sampling_weights.append(None)
                else:
                    self.stratified_sampling_weights.append(weights / total_weight)
        else:
            weights = self.genesites_df.iloc[self.all_ilocs, self.iweight].values
            total_weight = weights.sum()
            if total_weight == 0:
                ilogger.warning(
                    f"Total sampling weight is zero for the entire dataset."
                )
                self.sampling_weights = None
            else:
                self.sampling_weights = weights / total_weight

    def omni_sampler(self, idx):
        """Customized sampling logic based on the configuration"""
        self.iterations += 1

        if self.genecrops_cache is not None and len(self.genecrops_cache):
            sample_feat = self.genecrops_cache.pop(0)
        else:
            if self.customized_sampling:
                if self.stratified_sampling:  # randomly sample from stratified groups
                    igroup = np.random.randint(self.stratified_ngroups)
                    ilocs = self.stratified_ilocs[igroup]
                else:
                    ilocs = self.all_ilocs

                if self.weighted_sampling:
                    if self.stratified_sampling:
                        weights = self.stratified_sampling_weights[igroup]
                    else:
                        weights = self.sampling_weights
                    idx = np.random.choice(ilocs, p=weights)
                else:
                    idx = np.random.choice(ilocs)

            gene_ds = self.genesites_df.iloc[idx]
            gene_feat = splice_feats.sprinkle_sites_on_sequence(
                gene_ds,
                n_classes=self.num_classes,
                icol=self.tissue_cols,
                # training=self.training,
            )
            crop_feats = splice_feats.get_crop_feats_from_single_seq(
                gene_feat,
                num_crops=11 if self.genecrops_cache is not None else 1,
                crop_size=5000,
                flank_size=5000,
                min_sites=0,
                min_usage=0,
            )
            sample_feat = crop_feats[0]
            if self.genecrops_cache is not None:
                self.genecrops_cache.extend(crop_feats[1:])

        if sample_feat is None:
            if self.weighted_sampling and self.dynamic_weights:
                self.genesites_df.iloc[idx, self.iweight] = 0.0
            ilogger.warning(f"Warning: Sample features for index {idx} are None, skipping.")
            return None

        if self.training:
            if self.dynamic_weights:
                # Todo: update sampling weights based on sample quality
                if self.weighted_sampling and self.iterations % self.epoch_length == 0:
                    # self.get_normalized_sampling_weights()
                    pass

        return sample_feat