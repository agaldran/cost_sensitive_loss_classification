import argparse
from datetime import datetime
import json
import operator

from models.get_model import get_arch
from utils.get_loaders import get_train_val_loaders, modify_dataset
from utils.losses import get_cost_sensitive_criterion, get_cost_sensitive_regularized_criterion
from utils.evaluation import eval_predictions_multi, ewma
from utils.reproducibility import set_seeds
from utils.model_saving_loading import write_model, load_model
from tqdm import trange
import numpy as np
import torch

import os.path as osp
import os
import sys


def str2bool(v):
    # as seen here: https://stackoverflow.com/a/43357954/3208255
    if isinstance(v, bool):
       return v
    if v.lower() in ('true','yes'):
        return True
    elif v.lower() in ('false','no'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')

def compare_op(metric):
    '''
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    '''
    if metric == 'auc':
        return operator.gt, 0.5
    elif metric == 'kappa':
        return operator.gt, 0
    elif metric == 'kappa_auc_avg':
        return operator.gt, 0.25
    elif metric == 'loss':
        return operator.lt, np.inf
    elif metric == 'dice':
        return operator.gt, 0
    else:
        raise NotImplementedError

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def reduce_lr(optimizer, epoch, factor=0.1, verbose=True):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = old_lr * factor
        param_group['lr'] = new_lr
        if verbose:
            print('Epoch {:5d}: reducing learning rate'
                  ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

parser = argparse.ArgumentParser()
parser.add_argument('--csv_train', type=str, default='train.csv', help='path to training data csv')
parser.add_argument('--model_name', type=str, default='resnet18', help='selected architecture')
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='from pretrained weights')
parser.add_argument('--load_checkpoint', type=str, default='no', help='path to pre-trained checkpoint')
parser.add_argument('--base_loss', type=str, default='ce', help='base loss function (ce)')
parser.add_argument('--lambd', type=float, default=10., help='lagrange multiplier for ot_loss')
parser.add_argument('--exp', type=float, default=2, help='matrix exponentiation M**exp')
parser.add_argument('--n_classes', type=int, default=5, help='number of target classes (5)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--optimizer', type=str, default='sgd', help='sgd/adam')
parser.add_argument('--oversample', type=str, default='1/2/1/6/7', help='oversampling per-class proportions')
parser.add_argument('--n_epochs', type=int, default=50, help='total max epochs (1000)')
parser.add_argument('--patience', type=int, default=5, help='epochs until early stopping (20)')
parser.add_argument('--decay_f', type=float, default=0.1, help='decay factor after 3/4 of patience epochs (0=no decay)')
parser.add_argument('--metric', type=str, default='kappa', help='which metric to monitor (kappa/auc/loss/kappa_auc_avg)')
parser.add_argument('--save_model', type=str2bool, nargs='?', const=True, default=True, help='avoid saving anything (False)')
parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')

args = parser.parse_args()

def run_one_epoch_cls(loader, model, criterion, optimizer=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = optimizer is not None
    model.train() if train else model.eval()
    probs_all, preds_all, labels_all = [], [], []
    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for i_batch, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(inputs)
            probs = torch.nn.Softmax(dim=1)(logits)
            _, preds = torch.max(probs, 1)
            loss = criterion(logits, labels)
            if train:  # only in training mode
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            ll = loss.item()
            del loss
            probs_all.extend(probs.detach().cpu().numpy())
            preds_all.extend(preds.detach().cpu().numpy())
            labels_all.extend(labels.detach().cpu().numpy())

            # Compute running loss
            running_loss += ll * inputs.size(0)
            n_elems += inputs.size(0)
            run_loss = running_loss / n_elems
            if train: t.set_postfix(tr_loss="{:.4f}".format(float(run_loss)))
            else: t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()

    return np.stack(preds_all), np.stack(probs_all), np.stack(labels_all), run_loss

def train_cls(model, optimizer, train_criterion, val_criterion, train_loader, val_loader,
          oversample, n_epochs, metric, patience, decay_f, exp_path):
    counter_since_checkpoint = 0
    tr_losses, tr_aucs, tr_ks, vl_losses, vl_aucs, vl_ks = [], [], [], [], [], []
    stats = {}
    is_better, best_monitoring_metric = compare_op(metric)
    best_kappa, best_auc = 0,0
    for epoch in range(n_epochs):
        print('\n EPOCH: {:d}/{:d}'.format(epoch+1, n_epochs))
        if oversample == [1, 1, 1]:
            tr_preds, tr_probs, tr_labels, tr_loss = run_one_epoch_cls(train_loader, model, train_criterion, optimizer)
        else:
            csv_train_path = train_loader.dataset.csv_path
            train_loader_MOD = modify_dataset(train_loader, csv_train_path=csv_train_path, keep_samples=oversample)
            # train one epoch
            tr_preds, tr_probs, tr_labels, tr_loss = run_one_epoch_cls(train_loader_MOD, model, train_criterion, optimizer)

        # validate one epoch, note no optimizer is passed
        with torch.no_grad():
            vl_preds, vl_probs, vl_labels, vl_loss = run_one_epoch_cls(val_loader, model, val_criterion)
        tr_k, tr_auc, tr_acc = eval_predictions_multi(tr_labels, tr_preds, tr_probs)
        print('\n')
        vl_k, vl_auc, vl_acc = eval_predictions_multi(vl_labels, vl_preds, vl_probs)
        print('Train/Val. Loss: {:.4f}/{:.4f} -- Kappa: {:.4f}/{:.4f} -- AUC: {:.4f}/{:.4f} -- LR={:.6f}'.format(
                tr_loss, vl_loss, tr_k, vl_k, tr_auc, vl_auc, get_lr(optimizer)).rstrip('0'))
        # store performance for this epoch
        tr_losses.append(tr_loss)
        tr_aucs.append(tr_auc)
        tr_ks.append(tr_k)
        vl_losses.append(vl_loss)
        vl_aucs.append(vl_auc)
        vl_ks.append(vl_k)

        #  smooth val values with a moving average before comparing
        vl_auc = ewma(vl_aucs, window=3)[-1]
        vl_loss = ewma(vl_losses, window=3)[-1]
        vl_k = ewma(vl_ks, window=3)[-1]

        # check if performance was better than anyone before and checkpoint if so
        if metric =='auc': monitoring_metric = vl_auc
        elif metric =='loss': monitoring_metric = vl_loss
        elif metric == 'kappa': monitoring_metric = vl_k
        elif metric == 'kappa_auc_avg': monitoring_metric = 0.5*(vl_k+vl_auc)
        else: sys.exit('Not a suitable metric for this task')

        if is_better(monitoring_metric, best_monitoring_metric):
             print('Best (smoothed) val {} attained. {:.4f} --> {:.4f}'.format(
                 metric, best_monitoring_metric, monitoring_metric))
             best_auc, best_kappa = vl_auc, vl_k
             if exp_path != None:
                 print(15*'-',' Checkpointing ', 15*'-')
                 write_model(exp_path, model, optimizer, stats)

             best_monitoring_metric = monitoring_metric
             stats['tr_losses'], stats['vl_losses'] = tr_losses, vl_losses
             stats['tr_aucs'], stats['vl_aucs'] = tr_aucs, vl_aucs
             stats['tr_ks'], stats['vl_ks'] = tr_aucs, vl_aucs
             counter_since_checkpoint = 0  # reset patience
        else:
            counter_since_checkpoint += 1

        if decay_f != 0 and counter_since_checkpoint == 3*patience//4:
            reduce_lr(optimizer, epoch, factor=decay_f, verbose=False)
            print(8 * '-', ' Reducing LR now ', 8 * '-')

        # early stopping if no improvement happened for `patience` epochs
        if counter_since_checkpoint == patience:
            print('\n Early stopping the training, trained for {:d} epochs'.format(epoch))
            del model
            torch.cuda.empty_cache()
            return best_kappa, best_auc

    del model
    torch.cuda.empty_cache()
    return best_kappa, best_auc

if __name__ == '__main__':
    '''
    Example:
    python train.py --base_loss focal_loss 
    '''
    data_path = 'data'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # gather parser parameters
    args = parser.parse_args()
    model_name = args.model_name
    pretrained = args.pretrained
    load_checkpoint = args.load_checkpoint
    base_loss = args.base_loss
    lambd = args.lambd
    exp = args.exp
    lr, bs, optimizer_choice = args.lr, args.batch_size, args.optimizer
    csv_train = args.csv_train
    csv_train = osp.join(data_path, csv_train)
    csv_val = csv_train.replace('train', 'val')
    oversample = args.oversample.split('/')
    oversample = list(map(float, oversample))

    n_epochs, patience, decay_f, metric = args.n_epochs, args.patience, args.decay_f, args.metric
    save_model = str2bool(args.save_model)
    n_classes = args.n_classes
    if len(oversample) != n_classes: sys.exit('oversample must be a tuple of len {:d}'.format(n_classes))

    if save_model:
        save_path = args.save_path
        if save_path == 'date_time':
            save_path = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        experiment_path = osp.join('experiments', save_path)
        args.experiment_path = experiment_path  # store experiment path
        os.makedirs(experiment_path, exist_ok=True)
        config_file_path = osp.join(experiment_path,'config.cfg')
        # args.config_file_path = config_file_path
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    else: experiment_path=None

    print('* Instantiating model {}, pretrained={}'.format(model_name, pretrained))
    model, mean, std = get_arch(model_name, pretrained=pretrained, n_classes=n_classes)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if load_checkpoint != 'no':
        print('* Loading weights from previous checkpoint={}'.format(load_checkpoint))
        model, stats, optimizer_state_dict = load_model(model, load_checkpoint, device='cpu', with_opt=True)
    model = model.to(device)


    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    train_loader, val_loader = get_train_val_loaders(csv_path_train=csv_train, csv_path_val=csv_val,
                                                     batch_size=bs, mean=mean, std=std)

    if optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        sys.exit('not a valid optimizer choice')
    if load_checkpoint != 'no': optimizer.load_state_dict(optimizer_state_dict)


    print('* Instantiating base loss function {}, lambda={}, exp={}'.format(
            base_loss, str(lambd).rstrip('0'), str(exp).rstrip('0')))
    if base_loss == 'no':
        train_crit, val_crit = get_cost_sensitive_criterion(n_classes=n_classes, exp=exp)
    else:
        train_crit, val_crit = get_cost_sensitive_regularized_criterion(base_loss=base_loss, n_classes=n_classes, lambd=lambd, exp=exp)

    print('* Starting to train\n','-' * 10)
    m1, m2 = train_cls(model, optimizer, train_crit, val_crit, train_loader, val_loader,
              oversample, n_epochs, metric, patience, decay_f, experiment_path)
    print("kappa: %f" % m1)
    print("auc: %f" % m2)


    if save_model:
        file = open(osp.join(experiment_path, 'val_metrics.txt'), 'w')
        file.write(str(m1))
        file.write('\n')
        file.write(str(m2))
        file.close()
