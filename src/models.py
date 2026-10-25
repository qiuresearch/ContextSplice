import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

# number of channels in the convolutional layers
L = 32
# convolution window size in residual units
W = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                21, 21, 21, 21, 41, 41, 41, 41])
# atrous rate in residual units
AR = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                 10, 10, 10, 10, 25, 25, 25, 25])

CL = 5000  # context length on each side

class ResBlock(nn.Module):
    def __init__(self, L, W, AR, pad=True):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(L)
        s = 1 # stride
        # padding calculation: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/2
        if pad:
            padding = int(1 / 2 * (1 - L + AR * (W - 1) - s + L * s))
        else:
            padding = 0
        self.conv1 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)
        self.bn2 = nn.BatchNorm1d(L)
        self.conv2 = nn.Conv1d(L, L, W, dilation=AR, padding=padding)

    def forward(self, x):
        out = self.bn1(x)
        out = torch.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = out + x
        return out


class SpliceBaseModule(nn.Module):
    def __init__(self, **kwargs):
        super(SpliceBaseModule, self).__init__()
        self.extra_cfg = kwargs

    def calc_loss(self, preds, labels):
        """ preds are the output of forward() without any activation
            labels are the ground truth in the batch_feats
        """

        # ::TODO::
        # 1) Many sequences are shorter than crop_size, so we need to mask out the padded regions
        # cls has shape (B, num_classes, ..., crop_size)
        # psi has shape (B, ,..., crop_size)
        # 2) cls_odds and psi_std are not used currently

        avg_losses = {}
        if 'cls' not in labels or 'psi' not in labels:
            print("Missing labels for loss calculation, returning -1")
            return -1, avg_losses
        
        # four classes: non, acceptor, donor, hybrid
        avg_losses['cls_loss'] = F.cross_entropy(
            preds['cls_logits'],
            labels['cls'],
            ignore_index=-100)

        # psi is a probability between 0 and 1
        avg_losses['psi_loss'] = F.binary_cross_entropy_with_logits(
            preds['psi_logits'],
            labels['psi'],
            weight=labels['psi'] >= 0,  # mask out negative values
            reduction='mean')

        train_loss = avg_losses['cls_loss'] + avg_losses['psi_loss']
        avg_losses['loss'] = train_loss

        for k in avg_losses: # Move to CPU here, because of long epoch
            avg_losses[k] = avg_losses[k].detach().cpu()

        return train_loss, avg_losses

    @torch.no_grad()
    def calc_metric(self, preds, labels, eps=1e-8):
        """ Designed to be called at EVERY STEP of train, validation and test
        Args:
            preds are the output of forward() without any activation
            labels are the ground truth in the batch_feats
            return a dict of metric items
        """
        avg_metrics = {}
        if 'cls' not in labels or 'psi' not in labels:
            return avg_metrics
        
        # compute precision, recall, f1 for cls
        cls_pred = torch.movedim(F.softmax(preds['cls_logits'], dim=1), 1, -1)  # (B, [num_tissues], crop_size, num_classes)
        cls_label = F.one_hot(labels['cls'], num_classes=cls_pred.shape[-1]).to(cls_pred.dtype)  # (B, [num_tissues], crop_size, num_classes)

        all_but_last_dims = tuple(range(len(cls_pred.shape) - 1))
        cls_tp = (cls_pred * cls_label).sum(dim=all_but_last_dims)  # (num_classes,)
        cls_fp = (cls_pred * (1 - cls_label)).sum(dim=all_but_last_dims)  # (num_classes,)
        cls_fn = ((1 - cls_pred) * cls_label).sum(dim=all_but_last_dims)  # (num_classes,)

        cls_precision = cls_tp / (cls_tp + cls_fp + eps)
        cls_recall = cls_tp / (cls_tp + cls_fn + eps)
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall + eps)

        avg_metrics['cls_precision'] = cls_precision.mean()
        avg_metrics['cls_recall'] = cls_recall.mean()
        avg_metrics['cls_f1'] = cls_f1.mean()
        
        # compute mse for psi
        psi_pred = torch.sigmoid(preds['psi_logits'])  # (B, crop_size)
        avg_metrics['psi_mse'] = F.mse_loss(psi_pred, labels['psi'], reduction='mean')

        for k in avg_metrics: # Move to CPU here, because of long epoch
            avg_metrics[k] = avg_metrics[k].detach().cpu()

        return avg_metrics

    def preds_to_labels(self, preds):
        """ Convert the raw network outputs to discrete labels
            preds are the output of forward() without any activation
        """
        cls_probs = F.softmax(preds['cls_logits'], dim=1)  # (B, num_classes, ..., crop_size)
        psi_vals = torch.sigmoid(preds['psi_logits'])  # (B, ..., crop_size)

        preds['cls'] = cls_probs # torch.argmax(cls_probs, dim=1)  # (B, crop_size)
        preds['psi'] = psi_vals  # (B, crop_size)

        return preds

    def predict(self, batch_feats):
        pred_logits = self.forward(batch_feats)
        pred_labels = self.preds_to_labels(pred_logits)
        return pred_labels
    

class SpliceSolo(SpliceBaseModule):
    """ Only one change from PangolinSolo: output channels increased from 2 to 4 for CLS """
    def __init__(self, L=L, W=W, AR=AR, **kwargs):
        super(SpliceSolo, self).__init__()

        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1)
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = nn.Conv1d(L, 4, 1)
        self.conv_last2 = nn.Conv1d(L, 1, 1)
        
        self.extra_cfg = kwargs
        self.init_weights()
        
    def init_weights(self):
        # Bias final binary conv layers so initial sigmoid ≈ 0
        nn.init.constant_(self.conv_last2.bias, -5.0)
    
        # Optionally init weights to zero for those layers too
        nn.init.constant_(self.conv_last2.weight, 0.0)

    def forward(self, batch_feats):
        x = batch_feats['seq_onehot']

        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        #CL = 2 * np.sum(AR * (W - 1)) # CL = 1,0000
        #skip = F.pad(skip, (-CL // 2, -CL // 2)) 
        # take the middle segment only (this is the same as F.pad with negative padding)
        skip = skip[:, :, CL: -CL]
        out1 = self.conv_last1(skip)
        out2 = self.conv_last2(skip)
        return {'cls_logits': out1, 'psi_logits': out2.squeeze(dim=1)}


class ContextSpliceFiLM(SpliceBaseModule):
    """ Multi-layer FiLM: a shared context embedding Z_c (2-layer MLP over the
        regulator vector) is projected per residual block into (gamma_l, beta_l),
        which modulate that block's output: H_l = gamma_l * H_l + beta_l.
        ResBlock internals are unchanged. """
    def __init__(self, L, W, AR, context_dim=128, film_hidden=64, **kwargs):
        super(ContextSpliceFiLM, self).__init__()
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1) # in_channels, out_channels, kernel_size
        # self.conv2 = nn.Conv1d(32, L, 1)

        self.bn1 = nn.BatchNorm1d(L)
        # self.bn2 = nn.BatchNorm1d(L)

        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
                
        self.film_mlp = nn.Sequential(
            nn.Linear(context_dim, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, film_hidden), nn.ReLU())
        self.film_projs = nn.ModuleList(
            [nn.Linear(film_hidden, 2 * L) for _ in range(len(W))])

        self.conv_last1 = nn.Conv1d(L, 2, 1)
        self.conv_last2 = nn.Conv1d(L, 1, 1)
        self.init_weights()
        
    def init_weights(self):
        nn.init.constant_(self.conv_last2.weight, 0.0)
        nn.init.constant_(self.conv_last2.bias, -5.0)
        # start FiLM as identity: gamma = 1, beta = 0
        for proj in self.film_projs:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, batch_feats):
        x = batch_feats['seq_onehot']

        reg_gene = batch_feats['reg_context']  # (B, context_dim)
        if reg_gene.dim() == 3:
            reg_gene = reg_gene.mean(dim=-1)
        z_c = self.film_mlp(reg_gene)
            
        conv = self.bn1(self.conv1(x[:, :4, :]))
        # conv += self.bn2(self.conv2(x[:, 4:, :]))

        skip = self.skip(conv)
        j = 0
        for i in range(len(W)):
            conv = self.resblocks[i](conv)
            gamma, beta = self.film_projs[i](z_c).chunk(2, dim=-1)
            conv = (1 + gamma).unsqueeze(-1) * conv + beta.unsqueeze(-1)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        #CL = 2 * np.sum(AR * (W - 1))
        #skip = F.pad(skip, (-CL // 2, -CL // 2))
        skip = skip[:, :, CL: -CL]
        out1 = self.conv_last1(skip)
        out2 = self.conv_last2(skip)

        return {'cls_logits': out1, 
                'psi_logits': out2.squeeze(dim=1),}


class PangolinOmni3(SpliceBaseModule):
    """ Only one change from PangolinSolo: input channels increased from 4 to 36 """
    def __init__(self, L, W, AR, **kwargs):
        super(PangolinOmni3, self).__init__()
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1) # in_channels, out_channels, kernel_size
        self.conv2 = nn.Conv1d(32, L, 1)

        self.bn1 = nn.BatchNorm1d(L)
        # self.bn2 = nn.BatchNorm1d(L)

        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = nn.Conv1d(L, 2, 1)
        self.conv_last2 = nn.Conv1d(L, 1, 1)
        self.init_weights()
        
    def init_weights(self):
        nn.init.constant_(self.conv_last2.weight, 0.0)
        nn.init.constant_(self.conv_last2.bias, -5.0)

    def forward(self, batch_feats):
        x = batch_feats['seq_onehot']
            
        conv = 0.5 * self.bn1(self.conv1(x[:, :4, :]))
        conv += 0.5 * F.normalize(self.conv2(x[:, 4:, :]), p=2, dim=1)

        skip = self.skip(conv)
        j = 0
        for i in range(len(W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        #CL = 2 * np.sum(AR * (W - 1))
        #skip = F.pad(skip, (-CL // 2, -CL // 2))
        skip = skip[:, :, CL: -CL]
        out1 = self.conv_last1(skip)
        out2 = self.conv_last2(skip)

        return {'cls_logits': out1, 
                'psi_logits': out2.squeeze(dim=1),}
    

class PangolinOmni(SpliceBaseModule):
    """ Only one change from PangolinSolo: input channels increased from 4 to 36 """
    def __init__(self, L, W, AR, **kwargs):
        super(PangolinOmni, self).__init__()
        self.n_chans = L
        self.conv1 = nn.Conv1d(4 + 32, L, 1) # in_channels, out_channels, kernel_size
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = nn.Conv1d(L, 2, 1)
        self.conv_last2 = nn.Conv1d(L, 1, 1)
        self.init_weights()
        
    def init_weights(self):
        nn.init.constant_(self.conv_last2.weight, 0.0)
        nn.init.constant_(self.conv_last2.bias, -5.0)

    def forward(self, batch_feats):
        x = batch_feats['seq_onehot']
            
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        #CL = 2 * np.sum(AR * (W - 1))
        #skip = F.pad(skip, (-CL // 2, -CL // 2))
        skip = skip[:, :, CL: -CL]
        out1 = self.conv_last1(skip)
        out2 = self.conv_last2(skip)

        return {'cls_logits': out1, 
                'psi_logits': out2.squeeze(dim=1),}
    

class PangolinSolo(SpliceBaseModule):
    """ Only one change from Pangolin: output one tissue type only """
    def __init__(self, L, W, AR, **kwargs):
        super(PangolinSolo, self).__init__()
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1) # in_channels, out_channels, kernel_size
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = nn.Conv1d(L, 2, 1)
        self.conv_last2 = nn.Conv1d(L, 1, 1)
        self.init_weights()
        
    def init_weights(self):
        nn.init.constant_(self.conv_last2.weight, 0.0)
        nn.init.constant_(self.conv_last2.bias, -5.0)

    def forward(self, batch_feats):
        x = batch_feats['seq_onehot']
            
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        #CL = 2 * np.sum(AR * (W - 1))
        #skip = F.pad(skip, (-CL // 2, -CL // 2))
        skip = skip[:, :, CL: -CL]
        out1 = self.conv_last1(skip)
        out2 = self.conv_last2(skip)

        return {'cls_logits': out1, 
                'psi_logits': out2.squeeze(dim=1),}
    

class Pangolin(SpliceBaseModule):
    """ Orignal Pangolin model with multiple tissue types """
    def __init__(self, L, W, AR, **kwargs):
        super(Pangolin, self).__init__()
        self.n_chans = L
        self.conv1 = nn.Conv1d(4, L, 1) # in_channels, out_channels, kernel_size
        self.skip = nn.Conv1d(L, L, 1)
        self.resblocks, self.convs = nn.ModuleList(), nn.ModuleList()
        for i in range(len(W)):
            self.resblocks.append(ResBlock(L, W[i], AR[i]))
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                self.convs.append(nn.Conv1d(L, L, 1))
        self.conv_last1 = nn.Conv1d(L, 2, 1)
        self.conv_last2 = nn.Conv1d(L, 1, 1)
        self.conv_last3 = nn.Conv1d(L, 2, 1)
        self.conv_last4 = nn.Conv1d(L, 1, 1)
        self.conv_last5 = nn.Conv1d(L, 2, 1)
        self.conv_last6 = nn.Conv1d(L, 1, 1)
        self.conv_last7 = nn.Conv1d(L, 2, 1)
        self.conv_last8 = nn.Conv1d(L, 1, 1)
        self.init_weights()
        
    def init_weights(self):
        # Bias final binary conv layers so initial sigmoid ≈ 0
        nn.init.constant_(self.conv_last2.bias, -5.0)
        nn.init.constant_(self.conv_last4.bias, -5.0)
        nn.init.constant_(self.conv_last6.bias, -5.0)
        nn.init.constant_(self.conv_last8.bias, -5.0)
    
        # Optionally init weights to zero for those layers too
        nn.init.constant_(self.conv_last2.weight, 0.0)
        nn.init.constant_(self.conv_last4.weight, 0.0)
        nn.init.constant_(self.conv_last6.weight, 0.0)
        nn.init.constant_(self.conv_last8.weight, 0.0)

    def forward(self, batch_feats):
        x = batch_feats['seq_onehot']
            
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i in range(len(W)):
            conv = self.resblocks[i](conv)
            if (((i + 1) % 4 == 0) or ((i + 1) == len(W))):
                dense = self.convs[j](conv)
                j += 1
                skip = skip + dense
        #CL = 2 * np.sum(AR * (W - 1))
        #skip = F.pad(skip, (-CL // 2, -CL // 2))
        skip = skip[:, :, CL: -CL]
        out1 = self.conv_last1(skip)
        out2 = self.conv_last2(skip)
        out3 = self.conv_last3(skip)
        out4 = self.conv_last4(skip)
        out5 = self.conv_last5(skip)
        out6 = self.conv_last6(skip)
        out7 = self.conv_last7(skip)
        out8 = self.conv_last8(skip)

        # out1 = F.softmax(self.conv_last1(skip), dim=1)
        # out2 = torch.sigmoid(self.conv_last2(skip))
        # out3 = F.softmax(self.conv_last3(skip), dim=1)
        # out4 = torch.sigmoid(self.conv_last4(skip))
        # out5 = F.softmax(self.conv_last5(skip), dim=1)
        # out6 = torch.sigmoid(self.conv_last6(skip))
        # out7 = F.softmax(self.conv_last7(skip), dim=1)
        # out8 = torch.sigmoid(self.conv_last8(skip))

        # heart, liver, brain, testis
        return {'cls_logits': torch.stack([out1, out3, out5, out7], dim=2), # (B, n_classes, n_tissues, crop_size)
                'psi_logits': torch.cat([out2, out4, out6, out8], dim=1), # (B, n_tissues, crop_size)
                }
