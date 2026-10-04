import os
import time
import copy
import datetime
from beartype.typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

import gc
import torch
from pytorch_lightning import Callback, LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Timer, LearningRateMonitor, RichProgressBar, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from . import visuals
from . import utils
from . import loss_metrics

ilogger = utils.get_pylogger(__name__)


def peekaboo_epoch_metrics(epoch_metrics, prefix='', logger=None, logger_prefix='val/epoch_'):
    """ Display averaged batch metrics (a list of dicts) as a table. """

    if not epoch_metrics or len(epoch_metrics) == 0:
        print("No epoch metrics to summarize.")
        return

    print(f"\n{prefix} Epoch Metrics:")

    avg_metrics = copy.deepcopy(epoch_metrics[0])
    for batch_metrics in epoch_metrics[1:]:
        for k, v in batch_metrics.items():
            if k in avg_metrics:
                avg_metrics[k] += v
            else:
                avg_metrics[k] = v
    avg_metrics = {k: v / len(epoch_metrics) for k, v in avg_metrics.items()}

    for k, v in avg_metrics.items():
        print(f"{k:>28}: {float(v):>9.4f}")

    if logger is not None:
        for metric_name, metric_val in avg_metrics.items():
            logger(f'{logger_prefix}{metric_name}',
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    rank_zero_only=False,
            )
    return avg_metrics


def save_epoch_metrics(epoch_metrics, save_path):
    """ Save epoch metrics (a list of dicts) to a CSV file without Pandas. """
    if not epoch_metrics or len(epoch_metrics) == 0:
        print("No epoch metrics to save.")
        return

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    with open(save_path, 'w') as f:
        keys = list(epoch_metrics[0].keys())
        f.write(','.join(keys) + '\n')
        for batch_metrics in epoch_metrics:
            f.write(','.join(str(batch_metrics[k].item()) for k in keys) + '\n')


def save_model_outputs(feats, preds, save_path, pruning=True):
    """ Save model outputs (features and predictions) to a file. """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if pruning:
        feats = feats.copy()
        preds = preds.copy()
        if 'seq_onehot' in feats:
            feats.pop('seq_onehot')
        if 'cls_logits' in preds and 'cls' in preds:
            preds.pop('cls')
        if 'psi_logits' in preds and 'psi' in preds:
            preds.pop('psi')

    ilogger.info(f"Saving model outputs to {save_path} ...")
    torch.save((feats, preds), save_path)
    ilogger.info(f"Model outputs saved to {save_path}")


def save_eval_results(feats, preds, save_prefix=None, save_level=2, eval_dim=None, pruning=True):
    """ Save evaluation outputs and metrics.
    Args:
        feats: dict of tensors, ground truth features/labels
        preds: dict of tensors, model predictions
        save_prefix: str, prefix for saving files
        save_level: int, level of saving detail (0: none, 1: agg metrics, 2: outputs + agg metrics, 3: individual metrics)
        split_dim: int, dimension to split individual outputs for visualization only
        pruning: bool, whether to prune large tensors before saving
    """

    if save_level >= 2 and save_prefix:
        save_path = f"{save_prefix}_eval_outputs.pt"
        save_model_outputs(feats, preds, save_path, pruning=True)

    ilogger.info("Calculating aggregate metrics ...")
    agg_metrics = loss_metrics.calc_benchmark(preds, feats, exclude_dim=None)

    if save_level >= 1 and save_prefix:
        metrics_path = f"{save_prefix}_agg_metrics.yaml"
        utils.to_yaml(agg_metrics, yaml_path=metrics_path)
        ilogger.info(f"Aggregate metrics saved to {metrics_path}")
        visuals.plot_agg_metrics(agg_metrics, save_prefix=save_prefix+'_agg_metrics', display=False)

    if save_level >= 3 and save_prefix:
        ilogger.info("Calculating individual sample metrics ...")
        sam_metrics = loss_metrics.calc_benchmark(preds, feats, exclude_dim=[0])

        metrics_path = f"{save_prefix}_ind_metrics.csv"
        ilogger.info(f"Saving individual sample metrics to {metrics_path} ...")
        loss_metrics.save_sample_metrics(sam_metrics, metrics_path)
        ilogger.info(f"Individual sample metrics saved to {metrics_path}")
    else:
        sam_metrics = None

    if save_prefix and eval_dim is not None:
        dim_size = preds['cls_logits'].shape[eval_dim]
        for i in tqdm(range(dim_size), total=dim_size, desc=f"Saving eval outputs for dim {eval_dim} ..."):
            dim_preds = {'cls_logits': preds['cls_logits'].select(eval_dim, i),
                        'psi_logits': preds['psi_logits'].select(eval_dim, i),
                        'cls': preds['cls'].select(eval_dim, i),
                        'psi': preds['psi'].select(eval_dim, i),
                        }
            dim_labels = {'cls': feats['cls'].select(eval_dim, i),
                        'psi': feats['psi'].select(eval_dim, i),
                        }
            agg_metrics = loss_metrics.calc_benchmark(dim_preds, dim_labels, exclude_dim=None)
            metrics_path = f"{save_prefix}_agg_metrics_dim{i}.yaml"
            utils.to_yaml(agg_metrics, yaml_path=metrics_path)
            ilogger.info(f"Aggregate metrics saved to {metrics_path}")
            visuals.plot_agg_metrics(agg_metrics, save_prefix=f'{save_prefix}_agg_metrics_dim{i}', display=False)

    return agg_metrics, sam_metrics


def load_model_state_dict(model, state_dict_path):
    pass


class NanGradientCallback(Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Called right before an optimizer step to check gradients for NaNs."""
        if self._any_nan_gradients(pl_module):
            trainer.logger.log_metrics({"nan_gradient_detected": 1}, step=trainer.global_step)
            pl_module.zero_grad(set_to_none=True)  # Clear gradients
            print(f"NaN gradients detected at global step {trainer.global_step}, skipping optimizer step")

    def _any_nan_gradients(self, pl_module):
        """Check if any gradient is NaN in the model"""
        for param in pl_module.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                return True
        return False    
    

class OmniRunModule(LightningModule):
    def __init__(self, model, cfg=None, **kwargs):
        super().__init__()

        self.is_child_process = utils.get_rank() # utils.is_child_process()

        self.model = model
        self.cfg = {} if cfg is None else cfg
        self.extra_cfg = kwargs

        self.train_epoch_metrics = []
        self.train_epoch_samples = []
        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.test_epoch_metrics = []
        self.test_epoch_samples = []
        self.predict_epoch_metrics = []
        self.predict_epoch_samples = []

        self._train_steps = 0
        self._validation_steps = 0
        self._test_steps = 0
        self._predict_steps = 0

        self._epoch_start_time = time.time()        
        self.last_lr_step = -1 # not yet used
        self.save_hyperparameters(ignore=['model'])

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def resume_last_lr_step(self, lr_step):
        self.last_lr_step = lr_step


    def resume_from_ckpt(self, ckpt_path=None):
        """ not yet used """
        cfg = self.cfg
        if cfg.resume_from_ckpt and not cfg.resume_model_weights_only: # actual parameter loading is done by the trainer
            if(os.path.isdir(cfg.resume_from_ckpt)):
                pass
                # last_global_step = get_global_step_from_zero_checkpoint(cfg.resume_from_ckpt)
            else:
                sd = torch.load(cfg.resume_from_ckpt, map_location=torch.device("cpu"))
                last_global_step = int(sd['global_step'])
            self.resume_last_lr_step(last_global_step)
            ilogger.info(f"Successfully loaded last lr step: {self.last_lr_step}")
            
        if(cfg.resume_from_ckpt and cfg.resume_model_weights_only):
            if(os.path.isdir(cfg.resume_from_ckpt)):
                pass
                # sd = get_fp32_state_dict_from_zero_checkpoint(cfg.resume_from_ckpt)
            else:
                ilogger.info("Loading model weights only...")
                sd = torch.load(cfg.resume_from_ckpt, map_location=torch.device("cpu"))

            if 'state_dict' in sd:
                sd = {k[len("net."):]:v for k,v in sd['state_dict'].items() if k.startswith("net.")}
            elif 'module' in sd:
                sd = {k[len("_forward_module.net."):]:v for k,v in sd['module'].items() if k.startswith("_forward_module.net.")}
            else:
                ilogger.warning("No state_dict or module found in checkpoint, loading the whole checkpoint...")

            # print the state_dict keys of self.net
            # print(self.net.state_dict().keys())

            self.model.load_state_dict(sd, strict=False)
            ilogger.info(f"Successfully loaded model weights: {cfg.resume_from_ckpt}")

        # if(args.resume_from_jax_params):
        #     self.net.load_from_jax(args.resume_from_jax_params)
        #     logging.info(f"Successfully loaded JAX parameters at {args.resume_from_jax_params}...")

        # if(args.resume_from_jax_params is not None and args.resume_from_ckpt is not None):
            # raise ValueError("Choose between loading pretrained Jax-weights and a checkpoint-path")
            
        # TorchScript components of the model (commented out by QiuResearch)
        # if(args.script_modules):
        #     script_preset_(self)        

    def forward(self, batch_feats):
        return self.model(batch_feats)
    
    def configure_optimizers(self):

        optimizer_cfg = self.cfg.get('optimizer', {})

        lr_scheduler = optimizer_cfg.get('lr_scheduler', None)
        learning_rate = optimizer_cfg.get('learning_rate', 3e-4)
        eps = optimizer_cfg.get('eps', 1e-8)

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, eps=eps)

        #  if self.last_lr_step != -1:
        #     for group in optimizer.param_groups:
        #         if 'initial_lr' not in group:
        #             group['initial_lr'] = learning_rate   

        if lr_scheduler is None:
            return optimizer
        
        if lr_scheduler.upper() == "AlphaFoldLRScheduler".upper():
            lr_scheduler = "AlphaFoldLRScheduler"
            # LRScheduler = AlphaFoldLRScheduler(
            #     optimizer,
            #     base_lr=0.0,
            #     max_lr=learning_rate,
            #     last_epoch=self.last_lr_step,
            #     warmup_no_steps=self.args.train_epoch_len * self.args.lr_warmup_end,
            #     start_decay_after_n_steps=self.args.train_epoch_len * self.args.lr_decay_start,
            #     decay_every_n_steps=self.args.train_epoch_len * self.args.lr_decay_inteval,
            #     decay_factor=self.args.lr_decay_factor,
            # )
        elif lr_scheduler.upper() == "CosineAnnealingWarmRestarts".upper():
            LRScheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=optimizer_cfg.get('lr_T_0', 3), # in epochs
                T_mult=optimizer_cfg.get('lr_T_mult', 2), # T_i+1 = T_i * T_mult
                eta_min=0.0, # minimum learning rate
                last_epoch=optimizer_cfg.get('last_lr_step', -1),
            )
        elif lr_scheduler.upper() == "StepLR".upper():
            LRScheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=optimizer_cfg.get('lr_step_size', 10), # in epochs
                gamma=optimizer_cfg.get('lr_gamma', 0.1), # decay rate
                last_epoch=optimizer_cfg.get('last_lr_step', -1),
            )
        elif lr_scheduler.upper() == "MultiStepLR".upper():
            LRScheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=optimizer_cfg.get('lr_milestones', [10, 20]),
                gamma=optimizer_cfg.get('lr_gamma', 0.1),
                last_epoch=optimizer_cfg.get('last_lr_step', -1),
            )
        elif lr_scheduler.upper() == "CosineAnnealingLR".upper():
            LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=optimizer_cfg.get('lr_T_max', 50), # in epochs
                eta_min=optimizer_cfg.get('lr_eta_min', 0),
                last_epoch=optimizer_cfg.get('last_lr_step', -1),
            )
        elif lr_scheduler.upper() == "ReduceLROnPlateau".upper():
            LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=optimizer_cfg.get('lr_gamma', 0.1),
                patience=optimizer_cfg.get('lr_patience', 10),
                threshold=optimizer_cfg.get('lr_threshold', 1e-4),
                threshold_mode='rel',
                cooldown=optimizer_cfg.get('lr_cooldown', 0),
                min_lr=optimizer_cfg.get('lr_min', 0),
                eps=optimizer_cfg.get('lr_eps', 1e-8),
                verbose=True,
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")

        ilogger.info(f'Using {lr_scheduler} learning rate scheduler...')
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LRScheduler,
                "interval": optimizer_cfg.get('lr_interval', 'epoch'), # 'step' or 'epoch'
                "name": f'trainer/{lr_scheduler}',
            }
        }

    def on_train_epoch_start(self):
        self._epoch_start_time = time.time()
        ## if <1% 
        # if not self.use_f1 and self.current_epoch >= self.switch_epoch:
            # self.use_f1 = True
            # print(f"🔁 Switching to soft F1 loss at epoch {self.current_epoch}")
        return super().on_train_epoch_start()

    def training_step(self, batch_feats, batch_idx=None):
        if self._train_steps == 0 and not self.is_child_process:
            utils.peekaboo_tensors(batch_feats, prefix='Training Step Input')

        preds = self.model(batch_feats)
        if self._train_steps == 0 and not self.is_child_process:
            utils.peekaboo_tensors(preds, prefix='Training Step Forward')

        train_loss, avg_losses = self.model.calc_loss(preds, batch_feats)
        avg_losses.update(self.model.calc_metric(preds, batch_feats))
        if self._train_steps == 0 and not self.is_child_process:
            utils.peekaboo_tensors(avg_losses, prefix='Training Step Metrics')
        
        self.log("train_loss", train_loss, prog_bar=True)
        self.train_epoch_metrics.append(avg_losses)
        self._train_steps += 1

        if self._train_steps % 1000 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        return train_loss
    
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            rank_zero_only=False,
        )
        # self.log("trainer/epoch", self.current_epoch, prog_bar=False, logger=True)

        if len(self.train_epoch_metrics) > 0 and not self.is_child_process:
            peekaboo_epoch_metrics(
                self.train_epoch_metrics,
                prefix='Training',
                logger=self._log_scalar,
                logger_prefix='train/epoch_'
            )
            self.train_epoch_metrics.clear()                

    def validation_step(self, batch_feats, batch_idx=None):
        if self._validation_steps == 0 and not self.is_child_process:
            utils.peekaboo_tensors(batch_feats, prefix='Validation Step Input')

        preds = self.model(batch_feats)

        if self._validation_steps == 0 and not self.is_child_process:
            utils.peekaboo_tensors(preds, prefix='Validation Step Forward')

        train_loss, avg_losses = self.model.calc_loss(preds, batch_feats)
        avg_losses.update(self.model.calc_metric(preds, batch_feats))

        if self._validation_steps == 0 and not self.is_child_process:
            utils.peekaboo_tensors(avg_losses, prefix='Validation Step Metrics')

        self.log("val_loss", train_loss, prog_bar=True)
        self.validation_epoch_metrics.append(avg_losses)
        self._validation_steps += 1
        return train_loss

    def on_validation_epoch_start(self):
        self._epoch_start_time = time.time()
        return super().on_validation_epoch_start()
        
    def on_validation_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'val/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            rank_zero_only=False,
        )

        if len(self.validation_epoch_metrics) > 0 and not self.is_child_process:
            peekaboo_epoch_metrics(
                self.validation_epoch_metrics,
                prefix='Validation',
                logger=self._log_scalar,
                logger_prefix='val/epoch_'
            )
            self.validation_epoch_metrics.clear()

        return super().on_validation_epoch_end()

    def test_step(self, batch_feats, batch_idx=None):
        if self._test_steps == 0 and not self.is_child_process:
            utils.peekaboo_tensors(batch_feats, prefix='Test Step Input')
            
        preds = self.model(batch_feats)
        if self._test_steps == 0 and not self.is_child_process:
            utils.peekaboo_tensors(preds, prefix='Test Step Output')

        train_loss, avg_losses = self.model.calc_loss(preds, batch_feats)
        avg_losses.update(self.model.calc_metric(preds, batch_feats))
        if self._test_steps == 0 and not self.is_child_process:
            utils.peekaboo_tensors(avg_losses, prefix='Test Loss Items')

        self.log("test/loss", train_loss, prog_bar=True)
        self.test_epoch_metrics.append(avg_losses)

        pred_labels = self.model.preds_to_labels(preds)
        self._test_steps += 1
        # Return is NOT collected by trainer.test()! So not useful at all!
        return (batch_feats, pred_labels)

    def on_test_epoch_start(self):
        self._epoch_start_time = time.time()
        return super().on_test_epoch_start()
    
    def on_test_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'test/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            rank_zero_only=False,
        )

        if len(self.test_epoch_metrics) > 0 and not self.is_child_process:
            peekaboo_epoch_metrics(
                self.test_epoch_metrics,
                prefix='Test',
                logger=self._log_scalar,
                logger_prefix='test/epoch_'
            )
            self.test_epoch_metrics.clear()

        return super().on_test_epoch_end()

    def on_predict_epoch_start(self):
        if not self.is_child_process:
            self.predict_epoch_metrics.clear()
        return super().on_predict_epoch_start()
    
    def predict_step(self, batch_feats, batch_idx=None):
        if self._predict_steps == 0:
            utils.peekaboo_tensors(batch_feats, prefix='Predict Step Input')

        preds = self.model(batch_feats)
        self.model.preds_to_labels(preds)
        if self._predict_steps == 0:
            utils.peekaboo_tensors(preds, prefix='Predict Step Output')

        train_loss, avg_losses = self.model.calc_loss(preds, batch_feats)
        avg_losses.update(self.model.calc_metric(preds, batch_feats))
        if avg_losses:
            self.predict_epoch_metrics.append(avg_losses)
            if self._predict_steps == 0:
                utils.peekaboo_tensors(avg_losses, prefix='Predict Step Metrics')

        self._predict_steps += 1
        return (batch_feats, preds)

    def on_predict_epoch_end(self):
        if len(self.predict_epoch_metrics) > 0 and not self.is_child_process:
            peekaboo_epoch_metrics(
                self.predict_epoch_metrics,
                prefix='Predict',
                logger=None,
            )
        return super().on_predict_epoch_end()

    def fit(self, datamodule, cfg=None, accelerator="cuda", devices="auto", save_cfg=None, debug=False, **kwargs):
        """ Fit the model using PyTorch Lightning Trainer.
        Args:
            datamodule: LightningDataModule
            cfg: DictConfig, configuration for training, if None, use self.cfg
            devices: list of int, GPU device ids to use, if None, use all available
            save_cfg: DictConfig, configuration for the entire run, soley for the purpose of saving
        """

        cfg = self.cfg if cfg is None else \
            OmegaConf.merge(self.cfg, cfg, OmegaConf.create(kwargs))

        if cfg.get('resume_from_ckpt') and cfg.get('resume_cfg_override'):
            ilogger.info(f"Resume from {cfg.resume_from_ckpt} with config override...")
            pass # to be implemented
        elif cfg.get('resume_from_ckpt'):
            ilogger.info(f"Resume from {cfg.resume_from_ckpt} without config override...")
        elif cfg.get('resume_model_weights'):
            pass

        checkpoints_cfg = cfg['checkpoints']
        date_string =  datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        callbacks = []
        ckpt_home = checkpoints_cfg.get('dirpath', 'checkpoints') # keep the original ckpt home

        if debug: # set wand
            ilogger.info("Debug mode (without wandb and callbacks)...")
            wandb_logger = None
            # cfg.wandb.mode = 'offline'
            save_dir = os.path.join(ckpt_home, f'debug_{date_string}')
            checkpoints_cfg['dirpath'] = save_dir
        else:
            # cfg.wandb.name = f'{torch_model.__class__.__name__}_{date_string}'
            cfg.wandb.name = f'{cfg.wandb.name}_{date_string}'
            wandb_logger = WandbLogger(**cfg.wandb)
            ilogger.info(f"Wandb logger initialized with name: {wandb_logger.experiment.name}, id: {wandb_logger.experiment.id}")
            save_dir = os.path.join(ckpt_home, f'{cfg.wandb.name}_{wandb_logger.experiment.id}')

            if cfg.resume_from_ckpt:
                save_dir = os.path.dirname(cfg.resume_from_ckpt)
                # ckpt_dir = os.path.join(ckpt_dir, wandb_logger.experiment.id, cfg.wandb.name)

        # Model checkpoints
        checkpoints_cfg['dirpath'] = save_dir
        ilogger.info(f"Checkpoints saved to {save_dir}")
        callbacks.append(ModelCheckpoint(**checkpoints_cfg))
        callbacks.append(LearningRateMonitor(**cfg.lr_monitor))
        callbacks.append(EarlyStopping(**cfg.early_stopping))
        callbacks.append(NanGradientCallback())
        # if cfg.run.trainer.progress_bar_refresh_rate > 0:
            # callbacks.append(RichProgressBar(refresh_rate=cfg.run.trainer.progress_bar_refresh_rate))
            # callbacks.append(TQDMProgressBar(refresh_rate=cfg.run.trainer.progress_bar_refresh_rate))
        # if cfg.run.trainer.detect_anomaly:
            # torch.autograd.set_detect_anomaly(True)
            # ilogger.warning("Anomaly detection is enabled. Training will be slow!")
            #

        # save_conf is solely for saving purposes
        if save_cfg is not None:
            save_cfg = OmegaConf.merge(save_cfg, OmegaConf.create({'litrun': cfg}))
        else:
            save_cfg = cfg

        # Save config to wandb and as a yaml file
        if wandb_logger is not None and isinstance(wandb_logger.experiment.config, wandb.sdk.wandb_config.Config):
            cfg_dict = OmegaConf.to_container(save_cfg, resolve=True)
            flat_cfg = dict(utils.flatten_dict(cfg_dict))
            wandb_logger.experiment.config.update(flat_cfg)

        os.makedirs(save_dir, exist_ok=True)
        if not self.is_child_process:
            cfg_path = os.path.join(save_dir, 'train.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=save_cfg, f=f.name)

        checkpoints_cfg['dirpath'] = ckpt_home # restore the orignal home

        try:
            get_ipython
            cfg.trainer.strategy = 'auto'
            cfg.trainer.deterministic = False
            # cfg.run.trainer.fast_dev_run = True
        except NameError:
            pass

        trainer = Trainer(
            **cfg.trainer,
            callbacks=callbacks,
            logger=wandb_logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            accelerator=accelerator,
            devices=devices,
        )
        # strategy="ddp_find_unused_parameters_false" if len(devices) > 1 else None,
        # strategy="ddp" if len(devices) > 1 else None,
        # accumulate_grad_batches=main_cfg.trainer.accumulate_grad_batches,

        trainer.fit(
            model=self,
            datamodule=datamodule,
            ckpt_path=cfg.get('resume_from_ckpt', None),
        )

        return trainer

    def evaluate(self, datamodule, save_prefix=None, save_level=2, eval_dim=None,
                 accelerator="cuda", devices="auto", debug=False, **kwargs):
        """ Test the model using PyTorch Lightning Trainer.
        Args:
            datamodule: LightningDataModule
            devices: list of int, GPU device ids to use, if None, use all available
            debug: bool, whether to enable debug mode
            **kwargs: additional keyword arguments
        Returns:
            predictions: list of (batch_feats, preds) tuples from test_step
        """

        trainer = Trainer(**self.cfg.trainer,
                          enable_progress_bar=True,
                          enable_model_summary=True,
                          accelerator=accelerator,
                          devices=devices)
        
        eval_outputs = trainer.predict(
            self,
            datamodule=datamodule,
            ckpt_path=self.cfg.get('resume_from_ckpt', None),
        )

        if save_level >= 1 and save_prefix and not self.is_child_process and self.predict_epoch_metrics:
            metrics_path = f"{save_prefix}_batch_metrics.csv"
            save_epoch_metrics(self.predict_epoch_metrics, metrics_path)
            ilogger.info(f"Batch metrics saved to {metrics_path}")

        # concat all batches to two dicts of tensors (feats, preds)
        ilogger.info('Concatenate input and prediction batches...')
        eval_feats, eval_preds = utils.concat_dicts_outputs(eval_outputs)
        del eval_outputs

        agg_metrics, sam_metrics = save_eval_results(eval_feats, eval_preds, save_prefix=save_prefix,
            save_level=save_level, eval_dim=eval_dim, pruning=True)

        return {'feats': eval_feats, 'preds': eval_preds, 'agg_metrics': agg_metrics, 'sam_metrics': sam_metrics}

    def predict(self, datamodule, save_prefix=None, save_level=2, accelerator="cuda", devices="auto", debug=False, **kwargs):
        """ Inference using PyTorch Lightning Trainer.
        Args:
            datamodule: LightningDataModule
            devices: list of int, GPU device ids to use, if None, use all available
            debug: bool, whether to enable debug mode
            **kwargs: additional keyword arguments
        Returns:
            predictions: list of (batch_feats, preds) tuples from predict_step
        """

        trainer = Trainer(enable_progress_bar=True,
                          enable_model_summary=True,
                          accelerator=accelerator,
                          devices=devices)
        
        pred_outputs = trainer.predict(
            self,
            datamodule=datamodule,
            ckpt_path=self.cfg.get('resume_from_ckpt', None),
        )

        # concat all batches to two dicts of tensors (batch_feats, preds)
        input_feats, pred_targets = utils.concat_dicts_outputs(pred_outputs)

        if save_prefix is not None and not self.is_child_process:
            if save_level >= 1 and self.predict_epoch_metrics and len(self.predict_epoch_metrics) > 0:
                metrics_path = f"{save_prefix}_batch_metrics.csv"
                save_epoch_metrics(self.predict_epoch_metrics, metrics_path)
                ilogger.info(f"Batch metrics saved to {metrics_path}")            

            if save_level >= 2:
                save_path = f"{save_prefix}_pred_outputs.pt"
                save_model_outputs(input_feats, pred_targets, save_path, pruning=True)

        return {'feats': input_feats, 'preds': pred_targets}
