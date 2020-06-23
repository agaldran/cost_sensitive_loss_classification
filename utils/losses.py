import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats2
import sys
try:
    from kornia.losses import FocalLoss as focal_loss
except:
    pass

def get_gauss_label(label, n_classes, amplifier, noise=0):
    n = n_classes*amplifier
    half_int = amplifier/2
    label_noise = np.random.uniform(low=-noise, high=noise)
    if label == 0:
        label_noise = np.abs(label_noise)
    if label == 4:
        label_noise = -np.abs(label_noise)
    label += label_noise
    label_new = half_int + label*amplifier
    gauss_label = stats2.norm.pdf(np.arange(n), label_new, half_int/2)
    gauss_label/=np.sum(gauss_label)
    return gauss_label

def get_gaussian_label_distribution(n_classes, std=0.5):
    cls = []
    for n in range(n_classes):
        cls.append(stats2.norm.pdf(range(n_classes), 0, std))
    dists = np.stack(cls, axis=0)
    return dists
    # if n_classes == 3:
    #     CL_0 = stats2.norm.pdf([0, 1, 2], 0, std)
    #     CL_1 = stats2.norm.pdf([0, 1, 2], 1, std)
    #     CL_2 = stats2.norm.pdf([0, 1, 2], 2, std)
    #     dists = np.stack([CL_0, CL_1, CL_2], axis=0)
    #     return dists
    # if n_classes == 5:
    #     CL_0 = stats2.norm.pdf([0, 1, 2, 3, 4], 0, std)
    #     CL_1 = stats2.norm.pdf([0, 1, 2, 3, 4], 1, std)
    #     CL_2 = stats2.norm.pdf([0, 1, 2, 3, 4], 2, std)
    #     CL_3 = stats2.norm.pdf([0, 1, 2, 3, 4], 3, std)
    #     CL_4 = stats2.norm.pdf([0, 1, 2, 3, 4], 4, std)
    #     dists = np.stack([CL_0, CL_1, CL_2, CL_3, CL_4], axis=0)
    #     return dists
    # if n_classes == 6:
    #     CL_0 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 0, std)
    #     CL_1 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 1, std)
    #     CL_2 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 2, std)
    #     CL_3 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 3, std)
    #     CL_4 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 4, std)
    #     CL_5 = stats2.norm.pdf([0, 1, 2, 3, 4, 5], 5, std)
    #     dists = np.stack([CL_0, CL_1, CL_2, CL_3, CL_4, CL_5], axis=0)
    #     return dists
    # else:
    #     raise NotImplementedError

def cross_entropy_loss_one_hot(logits, target, reduction='mean'):
    logp = F.log_softmax(logits, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

def one_hot_encoding(label, n_classes):
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(
        1, label.view(-1, 1), 1)

def label_smoothing_criterion(alpha=0.1, distribution='uniform', std=0.5, reduction='mean'):
    def _label_smoothing_criterion(logits, labels):
        n_classes = logits.size(1)
        device = logits.device
        # manipulate labels
        one_hot = one_hot_encoding(labels, n_classes).float().to(device)
        if distribution == 'uniform':
            uniform = torch.ones_like(one_hot).to(device)/n_classes
            soft_labels = (1 - alpha)*one_hot + alpha*uniform
        elif distribution == 'gaussian':
            dist = get_gaussian_label_distribution(n_classes, std=std)
            soft_labels = torch.from_numpy(dist[labels.cpu().numpy()]).to(device)
        else:
            raise NotImplementedError

        loss = cross_entropy_loss_one_hot(logits, soft_labels.float(), reduction)

        return loss

    return _label_smoothing_criterion

def cost_sensitive_loss(input, target, M):
    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))
    device = input.device
    M = M.to(device)
    return (M[target, :]*input.float()).sum(axis=-1)
    # return torch.diag(torch.matmul(input, M[:, target]))

class CostSensitiveLoss(nn.Module):
    def __init__(self,  n_classes, exp=1, normalization='softmax', reduction='mean'):
        super(CostSensitiveLoss, self).__init__()
        if normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        elif normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = None
        self.reduction = reduction
        x = np.abs(np.arange(n_classes, dtype=np.float32))
        M = np.abs((x[:, np.newaxis] - x[np.newaxis, :])) ** exp
        M /= M.max()
        self.M = torch.from_numpy(M)

    def forward(self, logits, target):
        preds = self.normalization(logits)
        loss = cost_sensitive_loss(preds, target, self.M)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError('`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

class CostSensitiveRegularizedLoss(nn.Module):
    def __init__(self,  n_classes=5, exp=2, normalization='softmax', reduction='mean', base_loss='ce', lambd=10):
        super(CostSensitiveRegularizedLoss, self).__init__()
        if normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        elif normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = None
        self.reduction = reduction
        x = np.abs(np.arange(n_classes, dtype=np.float32))
        M = np.abs((x[:, np.newaxis] - x[np.newaxis, :])) ** exp
        #
        # M_oph = np.array([
        #                 [1469, 4, 5,  0,  0],
        #                 [58, 62,  5,  0,  0],
        #                 [22, 3, 118,  1,  0],
        #                 [0, 0,   13, 36,  1],
        #                 [0, 0,    0,  1, 15]
        #                 ], dtype=np.float)
        # M_oph = M_oph.T
        # # Normalize M_oph to obtain M_difficulty:
        # M_difficulty = 1-np.divide(M_oph, np.sum(M_oph, axis=1)[:, None])
        # # OPTION 1: average M and M_difficulty:
        # M = 0.5 * M + 0.5 * M_difficulty
        # ################
        # # OPTION 2: replace uninformative entries in M_difficulty by entries of M:
        # # M_difficulty[M_oph == 0] = M[M_oph == 0]
        # # M = M_difficulty

        M /= M.max()
        self.M = torch.from_numpy(M)
        self.lambd = lambd
        self.base_loss = base_loss

        if self.base_loss == 'ce':
            self.base_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
        elif self.base_loss == 'ls':
            self.base_loss = label_smoothing_criterion(distribution='uniform', reduction=reduction)
        elif self.base_loss == 'gls':
            self.base_loss = label_smoothing_criterion(distribution='gaussian', reduction=reduction)
        elif self.base_loss == 'focal_loss':
            kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": reduction}
            self.base_loss = focal_loss(**kwargs)
        else:
            sys.exit('not a supported base_loss')

    def forward(self, logits, target):
        base_l = self.base_loss(logits, target)
        if self.lambd == 0:
            return self.base_loss(logits, target)
        else:
            preds = self.normalization(logits)
            loss = cost_sensitive_loss(preds, target, self.M)
            if self.reduction == 'none':
                return base_l + self.lambd*loss
            elif self.reduction == 'mean':
                return base_l + self.lambd*loss.mean()
            elif self.reduction == 'sum':
                return base_l + self.lambd*loss.sum()
            else:
                raise ValueError('`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

def get_cost_sensitive_criterion(n_classes=5, exp=2):
    train_criterion = CostSensitiveLoss(n_classes, exp=exp, normalization='softmax')
    val_criterion = CostSensitiveLoss(n_classes, exp=exp, normalization='softmax')
    return train_criterion, val_criterion

def get_cost_sensitive_regularized_criterion(base_loss='ce', n_classes=5, lambd=1, exp=2):
    train_criterion = CostSensitiveRegularizedLoss(n_classes, exp=exp, normalization='softmax', base_loss=base_loss, lambd=lambd)
    val_criterion = CostSensitiveRegularizedLoss(n_classes, exp=exp, normalization='softmax', base_loss=base_loss, lambd=lambd)

    return train_criterion, val_criterion
