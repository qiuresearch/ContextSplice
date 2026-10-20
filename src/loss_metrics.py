import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from . import utils

ilogger = utils.get_pylogger(os.path.basename(__name__))


def calc_topk_roc_prc_curves(y_pred, y_true, ks=(0.5, 1, 2, 4), multiples_of_true=False, eps=1e-8):
    """ Compute top-k accuracy, precision, recall, f1, auroc, auprc
        y_pred: 1D or 2D numpy array of predicted scores (probabilities)
        y_true: 1D or 2D numpy array of ground truth binary labels (0 or 1)
        ks: list of k values to compute top-k metrics
        multiples_of_true: if True, k is interpreted as multiples of the number of positive samples in y_true
    """

    if y_true.ndim > 1:
        y_true = y_true.flatten()
    idx_true = torch.where(y_true > 0)[0]
    if idx_true.size == 0:
        ilogger.warning("No positive samples in y_true.")

    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    argsorted_y_pred = torch.argsort(y_pred)  # ascending order
    sorted_y_pred = y_pred[argsorted_y_pred]

    all_metrics = {'k0': ks}

    if multiples_of_true:
        ks = [int(np.ceil(k * len(idx_true))) for k in ks]
    else:
        ks = [int(np.ceil(k)) for k in ks]

    for k in ks:
        metric = {}
        if k < 1 or k > len(y_true):
            ilogger.warning(f"Invalid k={k} for y_true of length {len(y_true)}.")
            continue

        if k * len(idx_true) < len(y_true): # small k 
            # tp = torch.intersect1d(argsorted_y_pred[-k:], idx_true)
            tp = torch.tensor(len(idx_true) + k - len(torch.cat([idx_true, argsorted_y_pred[-k:]]).unique()))
        else:
            tp = y_true[argsorted_y_pred[-k:]].sum()

        metric['k'] = k
        metric['topk_threshold'] = sorted_y_pred[-k-1:-k+1].mean() if k > 1 else sorted_y_pred[-1]
        metric['topk_accuracy'] = tp / min(k, len(idx_true)) # from pangolin, can we do this?
        metric['topk_precision'] = tp / k
        metric['topk_recall'] = tp / len(idx_true)

        if metric['topk_precision'] + metric['topk_recall'] > 0:
            metric['topk_f1'] = 2 * metric['topk_precision'] * metric['topk_recall'] / (metric['topk_precision'] + metric['topk_recall'])
        else:
            metric['topk_f1'] = 0.0

        for key in metric:
            if key in all_metrics:
                all_metrics[key].append(metric[key])
            else:
                all_metrics[key] = [metric[key]]

    # convert all metrics to numpy arrays (HaHa)
    for key in all_metrics:
        all_metrics[key] = np.array(all_metrics[key])

    # auprc (TODO: group equal scores before computing precision-recall curve?)
    # quantile indices
    idx_thresholds = np.linspace(0, len(y_pred) - 1, num=min(len(y_true), 101), dtype=int)
    # linear space indices
    idx_linspace = np.array([np.searchsorted(sorted_y_pred, t, side='left') for t in np.linspace(0.01, 0.99, num=99)])
    idx_linspace = idx_linspace[(idx_linspace > 0) & (idx_linspace < len(y_pred))]
    # top-k indices
    idx_topks = (len(sorted_y_pred) - np.linspace(0, 3, num=101, dtype=float) * len(idx_true) - 1).astype(int)
    idx_topks = idx_topks[(idx_topks > 0) & (idx_topks < len(y_pred))]

    idx_thresholds = np.sort(np.unique(np.concatenate((idx_thresholds, idx_linspace, idx_topks))))
    thresholds = np.concatenate([sorted_y_pred[idx_thresholds], [1.0]])
    idx_thresholds = np.concatenate([idx_thresholds, [len(y_true)]])

    tprs = np.zeros_like(thresholds)
    fprs = np.zeros_like(thresholds)
    precisions = np.zeros_like(thresholds)
    recalls = np.zeros_like(thresholds)
    tps = np.zeros_like(thresholds)
    fps = np.zeros_like(thresholds)
    fns = np.zeros_like(thresholds)

    for i, idx in enumerate(idx_thresholds):
        tps[i] = y_true[argsorted_y_pred[idx:]].sum()
        fps[i] = len(y_true) - idx - tps[i]
        fns[i] = len(idx_true) - tps[i]

    tprs = (tps + eps) / (len(idx_true) + eps)
    fprs = (fps + eps) / (len(y_true) - len(idx_true) + eps)

    precisions = (tps + eps) / (tps + fps + eps)
    recalls = (tps + eps) / (tps + fns + eps)

    f1s = (2 * precisions * recalls + eps) / (precisions + recalls + eps)

    sorted_indices = np.arange(len(recalls)) # [np.argsort(recalls)]
    # compute AUC using the trapezoidal rule with numpy
    roc_auc = -np.trapz(tprs[sorted_indices], fprs[sorted_indices])
    prc_auc = -np.trapz(precisions[sorted_indices], recalls[sorted_indices])
    # alternatively, we can also use scipy.integrate to compute AUC
    # roc_auc = integrate.trapz(tprs[sorted_indices], fprs[sorted_indices])
    # prc_auc = integrate.trapz(precisions[sorted_indices], recalls[sorted_indices])

    all_metrics.update({
        'N': len(y_true),
        'idx_threshold': idx_thresholds,
        'threshold': thresholds[sorted_indices],
        'tp': tps,
        'fp': fps,
        'fn': fns,
        'tpr': tprs[sorted_indices],
        'fpr': fprs[sorted_indices],
        'recall': recalls[sorted_indices],
        'precision': precisions[sorted_indices],
        'f1': f1s[sorted_indices],
        'auprc': prc_auc,
        'auroc': roc_auc,
    })

    return all_metrics


def _get_remaining_dims(excluded_dims, ndim):
    """ excluded_dims must be a list or tuple of integers"""
    if excluded_dims:
        return tuple(i for i in range(ndim) if i not in excluded_dims and i - ndim not in excluded_dims)
    else:
        return tuple(range(ndim))
    
def _get_dim_index(dim, dims, ndim=None):
    if ndim:
        if dim < 0:
            dim = ndim + dim
        dims = sorted([i if i >= 0 else ndim + i for i in dims])

    return dims.index(dim) if dim in dims else None
    

def calc_loss(preds, labels, exclude_dim=None):
    """ 
        labels are the ground truth in the batch_feats
    """
    # TODO::
    # 1) Many sequences are shorter than crop_size, so we need to mask out the padded regions
    # cls has shape (B, 4, crop_size)
    # psi has shape (B, 1, crop_size)
    # 2) cls_odds and psi_std are not used currently

    loss_items = {}
    if 'cls' not in labels or 'psi' not in labels:
        ilogger.warning("No cls or psi in labels.")
        return -1, loss_items
    
    if exclude_dim is not None and utils.is_scalar(exclude_dim):
        exclude_dim = [int(exclude_dim)]    

    # four classes: non, acceptor, donor, hybrid
    if 'cls_logits' in preds:
        loss_items['cls_loss'] = F.cross_entropy(
            preds['cls_logits'],
            labels['cls'],
            ignore_index=-100,
            reduction='none' if exclude_dim else 'mean',
            )
    elif 'cls' in preds:
        loss_items['cls_loss'] = F.cross_entropy(
            torch.log(preds['cls'] + 1e-8),  # add a small value to avoid log(0)
            labels['cls'],
            ignore_index=-100,
            reduction='none' if exclude_dim else 'mean',
            )
    else:
        raise ValueError("No cls or cls_logits in preds.")

    if exclude_dim:
        loss_items['cls_loss'] = loss_items['cls_loss'].mean(
            dim=_get_remaining_dims(exclude_dim, loss_items['cls_loss'].ndim)
        )
    # psi is a probability between 0 and 1
    if 'psi_logits' in preds:
        loss_items['psi_loss'] = F.binary_cross_entropy_with_logits(
            preds['psi_logits'],
            labels['psi'],
            weight=labels['psi'] >= 0,  # mask out negative values
            reduction='none' if exclude_dim else 'mean')
    elif 'psi' in preds:
        psi_label = labels['psi'].float()
        weight = (psi_label >= 0).float()  # mask out negative values
        psi_label[psi_label < 0] = 0  # set negative values to 0 for BCE loss (an error will be raised otherwise)

        loss_items['psi_loss'] = F.binary_cross_entropy(
            preds['psi'].float(),
            psi_label,
            weight=weight,
            reduction='none' if exclude_dim else 'mean')
    else:
        raise ValueError("No psi or psi_logits in preds.")

    if exclude_dim:
        loss_items['psi_loss'] = loss_items['psi_loss'].mean(
            dim=_get_remaining_dims(exclude_dim, loss_items['psi_loss'].ndim)
        )

    loss = loss_items['cls_loss'] + loss_items['psi_loss']
    loss_items['loss'] = loss

    for k in loss_items:
        loss_items[k] = loss_items[k].cpu().numpy()

    return loss, loss_items


def calc_metric(preds, labels, C_dim=1, exclude_dim=None, to_numpy=True, eps=1e-8):
    """ preds are the output of forward() without any activation
        labels are the ground truth in the batch_feats
    Args:
        C_dim: the dimension of classes in preds['cls'], default is 1 (B, C, ..., L)
               preds['psi'] and labels['psi'] have NO C_dim as they are scalar regression!!!
        exclude_dim: list of dimensions to exclude when computing metrics, e.g., batch dimension
        to_numpy: whether to convert the output metrics to numpy arrays
    """
    metric_items = {}
    if 'cls' not in labels or 'psi' not in labels:
        return metric_items

    if exclude_dim is None or exclude_dim is False:
        exclude_dim = []
    else:
        if utils.is_scalar(exclude_dim):
            exclude_dim = [int(exclude_dim)]
    assert C_dim not in exclude_dim, "C_dim cannot be in exclude_dim."

    # compute precision, recall, f1 for cls

    # cls_logits in (B, num_classes/num_channels, crop_size)
    if 'cls' in preds:
        cls_pred = preds['cls']
    elif 'cls_logits' in preds:
        cls_pred = F.softmax(preds['cls_logits'], dim=C_dim)
    else:
        raise ValueError("No cls or cls_logits in preds.")

    cls_label = torch.movedim(F.one_hot(labels['cls'], num_classes=cls_pred.shape[C_dim]), -1, C_dim)

    dims_to_sum = _get_remaining_dims(exclude_dim + [C_dim], cls_pred.ndim)
    C_dim_new = _get_dim_index(C_dim, exclude_dim + [C_dim], ndim=cls_pred.ndim)

    # print(cls_pred.shape, cls_label.shape, dims_to_sum)

    cls_tp = (cls_pred * cls_label).sum(dim=dims_to_sum)  # (num_classes,) or (B, num_classes)
    cls_fp = (cls_pred * (1 - cls_label)).sum(dim=dims_to_sum)  # (num_classes,) or (B, num_classes)
    cls_fn = ((1 - cls_pred) * cls_label).sum(dim=dims_to_sum)  # (num_classes,) or (B, num_classes)

    cls_precision = cls_tp / (cls_tp + cls_fp + eps)
    cls_recall = cls_tp / (cls_tp + cls_fn + eps)
    cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall + eps)

    metric_items['cls_precision'] = cls_precision.mean(dim=C_dim_new)
    metric_items['cls_recall'] = cls_recall.mean(dim=C_dim_new)
    metric_items['cls_f1'] = cls_f1.mean(dim=C_dim_new)

    # compute mse for psi
    if 'psi' in preds:
        psi_pred = preds['psi']
    elif 'psi_logits' in preds:
        psi_pred = torch.sigmoid(preds['psi_logits'])  # (B, crop_size)
    else:
        raise ValueError("No psi or psi_logits in preds.")

    metric_items['psi_mse'] = F.mse_loss(psi_pred, labels['psi'], reduction='none' if exclude_dim else 'mean')
    if exclude_dim:
        metric_items['psi_mse'] = metric_items['psi_mse'].mean(
            dim=_get_remaining_dims(exclude_dim, metric_items['psi_mse'].ndim)
        )

    if to_numpy:
        for key in metric_items:
            metric_items[key] = metric_items[key].cpu().numpy()

    return metric_items


def calc_benchmark(preds, labels, C_dim=1, exclude_dim=None, eps=1e-8):
    """ preds are the output of forward() without any activation
        labels are the ground truth in the batch_feats
    Args:
        C_dim: the dimension of classes in preds['cls'], default is 1 (B, C, ..., L)
        exclude_dim: list of dimensions to exclude when computing metrics, e.g., batch dimension
        eps: small value to avoid division by zero
    Returns:
        all_metrics: dict of metrics, each value is a scalar or a numpy array
    """
    
    assert preds['cls'].ndim == labels['cls'].ndim + 1, "Dimension mismatch between preds['cls'] and labels['cls']+1."
    
    if exclude_dim is not None and utils.is_scalar(exclude_dim):
        exclude_dim = [int(exclude_dim)]

    if exclude_dim:
        assert len(exclude_dim) == 1, "Only support excluding one dimension for individual metrics."
        dim_size = labels['cls'].shape[exclude_dim[0]]
        E_dim = exclude_dim[0]
        assert E_dim >= 0, "exclude_dim must be non-negative."

        C_dim = (C_dim + preds['cls'].ndim) % preds['cls'].ndim
        assert E_dim != C_dim, "Cannot exclude the class dimension."

        if C_dim > E_dim:
            C_dim -= 1  # after excluding E_dim, C_dim shifts left by 1

        _, all_metrics = calc_loss(preds, labels, exclude_dim=exclude_dim)
        all_metrics.update(calc_metric(preds, labels, exclude_dim=exclude_dim, to_numpy=True, eps=eps))

        for i in tqdm(range(dim_size), total=dim_size, desc="Calculating per-sample metrics"):

            # y_pred = (preds['cls'][i].argmax(dim=0).flatten() > 0.5).to(torch.float32)
            y_pred = preds['cls'].select(dim=E_dim, index=i).select(dim=C_dim, index=-1).flatten()
            y_true = (labels['cls'].select(dim=E_dim, index=i).flatten() > 0.5).to(torch.float32)
            metric = calc_topk_roc_prc_curves(y_pred, y_true, ks=(0.5, 1, 2, 4), multiples_of_true=True)
            if metric is None:
                continue
            # only collect scalar metrics
            for key in metric:
                if hasattr(metric[key], '__len__') and len(metric[key]) > 1:
                    continue
                if key in all_metrics:
                    all_metrics[key].append(metric[key])
                else:
                    all_metrics[key] = [metric[key]]
        # convert lists to numpy arrays
        for key in all_metrics:
            if isinstance(all_metrics[key], list):
                all_metrics[key] = np.array(all_metrics[key])
    else:
        ilogger.info("Calculating aggregate loss and metrics ...")
        _, all_metrics = calc_loss(preds, labels)
        all_metrics.update(calc_metric(preds, labels, exclude_dim=None, to_numpy=True, eps=eps))        

        ilogger.info("Calculating ROC and PRC metrics...")
        # y_pred = (preds['cls'].argmax(dim=C_dim).flatten() > 0.5).to(torch.float32)
        y_pred = preds['cls'].select(dim=C_dim, index=-1).flatten()
        y_true = (labels['cls'].flatten() > 0.5).to(torch.float32)
        all_metrics.update(calc_topk_roc_prc_curves(y_pred, y_true, ks=(0.5, 1, 2, 4), multiples_of_true=True))
        all_metrics['num_samples'] = labels['cls'].shape[0]

    return all_metrics


def save_summary_metrics(metrics, save_path):
    """ Save summary metrics to a yaml file.
        metrics: dict containing summary metrics
        save_path: path to save the yaml file
    """
    pass


def save_sample_metrics(metrics, save_path):
    """ Save individual sample metrics to a CSV file without Pandas.
        metrics: list of dicts, each dict contains metrics for a batch
        save_path: path to save the CSV file
    """
    keys = list(metrics.keys())
    csv_lines = [','.join(keys)]
    for i in range(len(metrics[keys[0]])):
        csv_lines.append(','.join([str(metrics[key][i]) for key in keys]))

    with open(save_path, 'w') as f:
        f.writelines('\n'.join(csv_lines))    
    return
