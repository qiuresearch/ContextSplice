import math
import os
import numpy as np
from beartype.typing import Any, Dict, Optional

import torch
from torch.utils.data.distributed import DistributedSampler, dist
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule

class OmniDataModule(LightningDataModule):
    def __init__(self, train_dataset=None, val_dataset=None, test_dataset=None, predict_dataset=None,
                 batch_size=16, num_workers=4, batch_sampler=None, collater=None,
                 **kwargs):
        super().__init__()
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.collater = collater # not used yet

        self.cfg = kwargs
        self.save_hyperparameters(logger=False, ignore=['train_dataset', 'val_dataset', 'test_dataset', 'predict_dataset', 'batch_sampler', 'collater'])

    def train_dataloader(self, rank=None, num_replicas=None):
        if self.train_dataset is None:
            return None

        if not self.batch_sampler:
            dataloader_cfg = {
                'shuffle': self.cfg.get('train_shuffle', True),
                'batch_size': self.cfg.get('train_batch_size', self.batch_size),
            }
        elif self.batch_sampler.upper() == 'RNALength'.upper():
            dataloader_cfg = {
                'batch_sampler': RNALengthBatcher(
                sampler_cfg=self.cfg,
                sample_lengths=self.train_dataset.get_sample_lengths(),
                pad_multiples_of=self.cfg.get('pad_multiples_of', None),
                rank=rank,
                num_replicas=num_replicas,
                )
            }
        else:
            raise ValueError(f"Unknown batch_sampler: {self.batch_sampler}")
        
        return DataLoader(self.train_dataset, 
                          num_workers=self.cfg.get('train_num_workers', self.num_workers),
                          prefetch_factor=self.cfg.get('train_prefetch_factor', 2),
                          pin_memory=self.cfg.get('train_pin_memory', True),
                          persistent_workers=self.cfg.get('train_persistent_workers', True),
                          drop_last=self.cfg.get('train_drop_last', True),
                          **dataloader_cfg
                          )
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        if dist.is_available() and dist.is_initialized():
            dataloader_cfg = {
                'sampler': DistributedSampler(self.val_dataset, shuffle=False),
            }
        else:
            dataloader_cfg = {}

        return DataLoader(self.val_dataset, 
                          shuffle=self.cfg.get('val_shuffle', False),
                          batch_size=self.cfg.get('val_batch_size', self.batch_size),
                          num_workers=self.cfg.get('val_num_workers', self.num_workers),
                          prefetch_factor=self.cfg.get('val_prefetch_factor', 2),
                          pin_memory=self.cfg.get('val_pin_memory', True),
                          persistent_workers=self.cfg.get('val_persistent_workers', True),
                          **dataloader_cfg,
                          )
    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(self.test_dataset,
                          shuffle=self.cfg.get('test_shuffle', False),
                          batch_size=self.cfg.get('test_batch_size', self.batch_size),
                          num_workers=self.cfg.get('test_num_workers', self.num_workers),
                          prefetch_factor=self.cfg.get('test_prefetch_factor', 2),
                          pin_memory=self.cfg.get('test_pin_memory', False),
                          persistent_workers=self.cfg.get('test_persistent_workers', True),
                          )
    def set_predict_dataset(self, predict_dataset):
        self.predict_dataset = predict_dataset
        
    def predict_dataloader(self):
        if self.predict_dataset is None:
            return None
        return DataLoader(self.predict_dataset, 
                          shuffle=self.cfg.get('predict_shuffle', False),
                          batch_size=self.cfg.get('predict_batch_size', self.batch_size),
                          num_workers=self.cfg.get('predict_num_workers', self.num_workers),
                          prefetch_factor=self.cfg.get('predict_prefetch_factor', 2),
                          pin_memory=self.cfg.get('predict_pin_memory', False),
                          persistent_workers=self.cfg.get('predict_persistent_workers', True),
                          )
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any] = None):
        """Things to do when loading checkpoint."""
        pass

"""
Taken from
https://github.com/microsoft/protein-frame-flow/blob/main/data/pdb_dataloader.py#L162
"""
class RNALengthBatcher:
    def __init__(
            self,
            sampler_cfg,
            sample_lengths,
            pad_multiples_of=None,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
        ):
        super().__init__()
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                self.num_replicas = 1
            else:
                self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                self.rank = 0
            else:
                self.rank = dist.get_rank()
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self.sample_lengths = sample_lengths
        if pad_multiples_of is not None:
            self.sample_lengths = np.ceil(self.sample_lengths / pad_multiples_of) * pad_multiples_of
        self.num_samples = len(self.sample_lengths)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._sampler_cfg['max_batch_size']
        
        # Group indices by lengths for efficient batching (akin to pandas groupby)
        sorted_indices = np.argsort(self.sample_lengths)
        sorted_lengths = self.sample_lengths[sorted_indices]
        unique_indices = np.where(sorted_lengths[1:] != sorted_lengths[:-1])[0]
        if len(unique_indices) == 0:
            print("Warning: all samples have the same length.")
            unique_indices = np.array([0, len(sorted_lengths)])
        else:
            unique_indices = np.concatenate(([0], unique_indices+1, [len(sorted_lengths)]))

        length_group_indices = {}
        for i in range(len(unique_indices) - 1):
            length = sorted_lengths[unique_indices[i]]
            length_group_indices[length] = sorted_indices[unique_indices[i]:unique_indices[i+1]]
            
        self.length_group_indices = length_group_indices
        
        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        # self.num_batches = int(np.ceil(self.num_samples / self.num_replicas))
        if self._sampler_cfg['linear_effect']:
            self.num_batches = self.sample_lengths.sum() / self._sampler_cfg['max_num_res_squared'] / self.num_replicas
        else:
            self.num_batches = (self.sample_lengths ** 2).sum() / self._sampler_cfg['max_num_res_squared'] / self.num_replicas
        self.num_batches = max(
            math.ceil(self.num_batches) + len(self.length_group_indices),
            math.ceil(self.num_samples / self.max_batch_size),
        )

    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.
        rng = np.random.default_rng(self.seed + self.epoch)
        
        # Each batch contains multiple RNAs of the same length.
        sample_batches = []
        for grp_len, grp_indices in self.length_group_indices.items():

            if self.shuffle:
                indices2grp = rng.permutation(len(grp_indices))
            else:
                indices2grp = np.arange(len(grp_indices))

            if len(grp_indices) > self.num_replicas > 1:
                replica_indices = grp_indices[indices2grp[self.rank::self.num_replicas]]
            else:
                replica_indices = grp_indices[indices2grp]

            if self._sampler_cfg['linear_effect']:
                max_batch_size = min(
                    self.max_batch_size,
                    self._sampler_cfg['max_num_res_squared'] // grp_len + 1,
                )
            else:
                max_batch_size = min(
                    self.max_batch_size,
                    self._sampler_cfg['max_num_res_squared'] // grp_len**2 + 1,
                )
                
            max_batch_size = int(max_batch_size)
            num_batches = math.ceil(len(replica_indices) / max_batch_size)
            for i in range(num_batches):
                batch_indices = replica_indices[i*max_batch_size:(i+1)*max_batch_size]
                assert all(self.sample_lengths[batch_indices] == self.sample_lengths[batch_indices[0]]), \
                    f"Batch contains different lengths: {self.sample_lengths[batch_indices]}"
                sample_batches.append(batch_indices)
        
        # Remove any length bias
        rng.shuffle(sample_batches)
        return sample_batches

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947

        all_batches = []
        num_augments = -1
        while len(all_batches) < self.num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if len(all_batches) >= self.num_batches:
            all_batches = all_batches[:self.num_batches]
        # print(all_batches[-5:0])
        self.sample_batches = all_batches

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_batches)

    def __len__(self):
        if hasattr(self, "sample_batches"):
            return len(self.sample_batches)
        else:
            return self.num_batches

# class MyBatchCollator:
#     """ batch_dim: whether batch dim already exists in the input 
#         set_length: the length of the set, if None, use the max length in the batch
#         multiple_of: if set_length is None, use the multiple of this number as set_length
#     """
#     def __init__(self, batch_dim=True, set_length=None, multiple_of=64) -> None:
#         """ batch_dim: whether batch dim already exists in the input """
#         self.batch_dim = batch_dim
#         self.set_length = set_length
#         self.multiple_of = multiple_of
#         if set_length is not None and multiple_of is not None:
#             logger.warning(f"set_length and multiple_of are both specified, multiple_of will be ignored")
    
#     def __call__(self, samples):

#         if self.set_length is not None or self.multiple_of is not None or len(samples) > 1:
#             samples = tailor_data_shapes(samples,
#                                    set_length=self.set_length, 
#                                    multiple_of=self.multiple_of, 
#                                    batch_dim=self.batch_dim)

#         if self.batch_dim:
#             stack_fn = partial(torch.concat, dim=0)
#         else:
#             stack_fn = partial(torch.stack, dim=0)

#         return dicts_aggregate(stack_fn, samples)


class MyDataLoader(DataLoader):
    def __init__(self, *args, config, stage="train", generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.stage = stage
        self.generator = generator
        self._prep_batch_properties_probs()

    def _prep_batch_properties_probs(self):
        keyed_probs = []
        stage_cfg = self.config[self.stage]

        max_iters = self.config.common.max_recycling_iters

        # this block is present in opencomplex code but not in openfold code
        # if(stage_cfg.supervised):
        #     clamp_prob = self.config.supervised.clamp_prob
        #     keyed_probs.append(
        #         ("use_clamped_fape", [1 - clamp_prob, clamp_prob])
        #     )

        if(stage_cfg.uniform_recycling):
            recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
        else:
            recycling_probs = [
                0. for _ in range(max_iters + 1)
            ]
            recycling_probs[-1] = 1.

        keyed_probs.append(
            ("no_recycling_iters", recycling_probs)
        )

        keys, probs = zip(*keyed_probs)
        max_len = max([len(p) for p in probs])
        padding = [[0.] * (max_len - len(p)) for p in probs]

        self.prop_keys = keys
        self.prop_probs_tensor = torch.tensor(
            [p + pad for p, pad in zip(probs, padding)],
            dtype=torch.float32,
        )

    def _add_batch_properties(self, batch):
        return batch

    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._add_batch_properties(batch)

        return _batch_prop_gen(it)
