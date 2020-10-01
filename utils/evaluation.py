import numpy as np
import sys

from sklearn.metrics import cohen_kappa_score as kappa, roc_auc_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score



def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """
    pretty print for confusion matrixes
    https://gist.github.com/zachguo/10296432
    """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

def ewma(data, window=5):
    # exponetially-weighted moving averages
    data = np.array(data)
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def eval_predictions_multi(y_true, y_pred, y_proba, print_conf=True):
    acc = balanced_accuracy_score(y_true, y_pred)
    k = kappa(y_true, y_pred, weights='quadratic')
    classes = ['DR0', 'DR1', 'DR2', 'DR3', 'DR4']
    # mean_auc = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
    # ovo should be better, but average is not clear from docs
    # mean_auc = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovo')
    mean_auc = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovo')
    if print_conf:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        print_cm(cm, classes)

    return k, mean_auc, acc
