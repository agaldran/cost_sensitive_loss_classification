import argparse
import pandas as pd
from models.get_model import get_arch
from utils.get_loaders import get_test_loader

from utils.evaluation import eval_predictions_multi
from utils.reproducibility import set_seeds
from utils.model_saving_loading import load_model
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

parser = argparse.ArgumentParser()
parser.add_argument('--csv_test', type=str, default='data/test_eyepacs.csv', help='path to test data csv')
parser.add_argument('--model_name', type=str, default='resnext50', help='selected architecture')
parser.add_argument('--load_path', type=str, default='experiments/gls_reg_1e2_resnext50_exp2', help='path to saved model')
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='from pretrained weights')
parser.add_argument('--dihedral_tta', type=int, default=2, help='dihedral group cardinality (2)')
parser.add_argument('--n_classes', type=int, default=5, help='number of target classes (5)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--csv_out', type=str, default='results/results.csv', help='path to output csv')

args = parser.parse_args()

def run_one_epoch_cls(loader, model, optimizer=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = optimizer is not None
    model.train() if train else model.eval()
    probs_all, preds_all, labels_all = [], [], []
    with trange(len(loader)) as t:
        for i_batch, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(inputs)
            probs = torch.nn.Softmax(dim=1)(logits)
            _, preds = torch.max(probs, 1)
            probs_all.extend(probs.detach().cpu().numpy())
            preds_all.extend(preds.detach().cpu().numpy())
            labels_all.extend(labels.detach().cpu().numpy())
            run_loss = 0
            t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()

    return np.stack(preds_all), np.stack(probs_all), np.stack(labels_all)

def test_cls_tta_dihedral(model, test_loader, n=3):
    probs_tta = []
    prs = [0, 1]
    import torchvision
    test_loader.dataset.transforms.transforms.insert(-1, torchvision.transforms.RandomRotation(0))
    rotations = np.array([i * 360 // n for i in range(n)])
    for angle in rotations:
        for p2 in prs:
            test_loader.dataset.transforms.transforms[2].p = p2  # pr(vertical flip)
            test_loader.dataset.transforms.transforms[-2].degrees = [angle, angle]
            # validate one epoch, note no optimizer is passed
            with torch.no_grad():
                test_preds, test_probs, test_labels = run_one_epoch_cls(test_loader, model)
                probs_tta.append(test_probs)

    probs_tta = np.mean(np.array(probs_tta), axis=0)
    preds_tta = np.argmax(probs_tta, axis=1)

    test_k, test_auc, test_acc = eval_predictions_multi(test_labels, preds_tta, probs_tta)
    print('Test Kappa: {:.4f} -- AUC: {:.4f} -- Balanced Acc: {:.4f}'.format(test_k, test_auc, test_acc))

    del model
    torch.cuda.empty_cache()
    return probs_tta, preds_tta, test_labels

def test_cls(model, test_loader):
    # validate one epoch, note no optimizer is passed
    with torch.no_grad():
        preds, probs, labels = run_one_epoch_cls(test_loader, model)
    print(labels)
    vl_k, vl_auc, vl_acc = eval_predictions_multi(labels, preds, probs)
    print('Val. Kappa: {:.4f} -- AUC: {:.4f}'.format(vl_k, vl_auc).rstrip('0'))

    del model
    torch.cuda.empty_cache()
    return probs, preds, labels

if __name__ == '__main__':
    '''
    Example:
    python test.py --dihedral_tta 2 --load_path experiments/ce_reg_1e2 --csv_out results/ce_reg_1e2.csv
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
    load_path = args.load_path
    pretrained = args.pretrained
    bs = args.batch_size
    csv_test = args.csv_test
    n_classes = args.n_classes
    dihedral_tta = args.dihedral_tta
    csv_out = args.csv_out

    print('* Instantiating model {}, pretrained={}'.format(model_name, pretrained))
    model, mean, std = get_arch(model_name, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test,  batch_size=bs, mean=mean, std=std)

    if dihedral_tta==0:
        probs, preds, labels = test_cls(model, test_loader)
    elif dihedral_tta>0:
        probs, preds, labels = test_cls_tta_dihedral(model, test_loader, n=dihedral_tta)
    df = pd.DataFrame(zip(list(test_loader.dataset.im_list),
                      probs[:, 0], probs[:, 1], probs[:, 2],
                      probs[:, 3], probs[:, 4], labels),
                      columns=['image_id', 'dr0', 'dr1', 'dr2', 'dr3', 'dr4', 'gt'])
    df.to_csv(csv_out, index=False)

