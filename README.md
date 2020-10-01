# Cost-Sensitive loss for multi-class classification
This is a repository containing our implementation of cost-sensitive loss functions for classification tasks in pytorch, as presented in:

```
Cost-Sensitive Regularization for Diabetic Retinopathy Grading from Eye Fundus Images
Adrian Galdran, José Dolz, Hadi Chakor, Hervé Lombaert, Ismail Ben Ayed
Medical Image Computing and Computer Assisted Intervention, 2020 (accepted)
```

If you find the code here useful in your research, we appreciate if you can cite our work. Thanks!

### Introduction

The proposed idea is quite simple. If you want to penalize different kinds of errors while training your model to perform multi-class classification, you first neeed to encode those penalties into a penalty (or confusion) matrix. In a silly example, imagine you have a problem with `n=3` classes, and you are very worried that your model mis-classifies instances of class `2` as class `0`, but you don't care at all about any of the other possible types of errors. You would build a matrix like the following:

<a href="https://www.codecogs.com/eqnedit.php?latex=\displaystyle&space;M&space;=&space;\begin{bmatrix}&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0\\&space;10&space;&&space;0&space;&&space;0\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\displaystyle&space;M&space;=&space;\begin{bmatrix}&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0\\&space;10&space;&&space;0&space;&&space;0\end{bmatrix}" title="\displaystyle M = \begin{bmatrix} 0 & 0 & 0\\ 0 & 0 & 0\\ 10 & 0 & 0\end{bmatrix}" /></a>

Given an example with a label `y=k` and a one-hot encoded prediction `U(x)=[x_1, x_2, x_3]` (being `U` your neural network or whatever model you are using), a cost-sensitive loss would be computed by simply taking the scalar product of the `k`-th row of `M` and `U(x)`. As you can see, if your example has `y=2` and you have a (correct) prediction `U(x)=[0,0,1]`, then the loss is `L=<[10, 0, 0], [0,0,1]> = 0`. However, if you have an incorrect prediction `U(x)=[1,0,0]`, you get `L=10`. And the funny part, if you have a prediction `U(x)=[0,1,0]`, you still get `L=0`, since you did not penalize this kind of error within `M`.

Enough talking, if you want to use this loss function, you just need to import it and instantiate it as follows:
```
from utils.losses import CostSensitiveLoss

n_classes = 3
criterion = CostSensitiveLoss(n_classes)
```

Please have into account that `criterion` expects raw outputs of your network, i.e., pre-softmax activations. It will apply a softmax internally (you can change that with the `normalization` parameter).

By default, `criterion` implements a cost matrix that penalizes faraway predictions more than closeby predictions, which is a useful thing to have in image grading/ordinal classification problems:

```
print(criterion.M)
[[0.0000, 0.5000, 1.0000],
[0.5000, 0.0000, 0.5000],
[1.0000, 0.5000, 0.0000]]
```
Once instantiated, you can modify `criterion.M` to suit your needs or impose other kind of penalties. All this and more (e.g. how to use this tool to model a-priori inter-observer disagreement knowledge you may have - a confusion matrix for annotators) is explained in the `CS_loss.ipynb` notebook inside this repo.

### Cost-Sensitive Regularization
In our experiments (and elsewhere) we found out that simply using a CS loss leads to lots of troubles in terms of CNNs staying at local minima where they will predict a trivial configuration (all the time the same category), which seems to be very satisfying for this kind of losses. For this reason, we recommend using this as a regularizer for other standard classification losses. 

In our implementation we provide a wrapper for doing this, where you specify a `base_loss` and the regularization parameter `lambd`:

```
from utils.losses import CostSensitiveRegularizedLoss

n_classes = 3 
base_loss = 'ce'
lambd = 10
cs_regularized_criterion = CostSensitiveRegularizedLoss(n_classes=n_classes, base_loss=base_loss, lambd=lambd)
```

We provide other base losses in our implementation (focal loss, cross-entropy with label smoothing, cross-entropy with gaussian label smoothing). Please see the notebook for more details.

### Cost-Senstive Regularization for Diabetic retinopathy Classification
You first need to download the [data](https://www.kaggle.com/c/diabetic-retinopathy-detection/data), unzip and and store all images in the `data/images` folder, see the csv files for our train/val split. We also pre-processed images by cropping to the field-of-view and resizing, which provides faster training than having to resize on-the-fly for each iteration. You can probably recycle part of the code we used for this, it's in `prepare_training_data.py` and `prepare_test_data.py`.

When everything is ready, you can for example run the following:
```
python train.py --csv_train train.csv --model_name resnext50 --base_loss gls --labmd 10 --exp 2 --save_path gls_reg_1e2_resnext50_exp2
```
Which trains a resnext50 model on the eyepacs dataset with a Gaussian Label Smoothing+Cross-Entropy loss, that is regularized by a CS term with a weight of 10, and where the cost matrix is L2 (because of --exp 2). 
After training, weights are stored at `experiments/gls_reg_1e2_resnext50_exp2/`. If you now want to generate predictions on the Eyepacs test set, you would run:
```
test_dihedral_tta.py --csv_test data/test_eyepacs.csv --model_name resnext50 --load_path experiments/gls_reg_1e2_resnext50_exp2 --csv_out results/results.csv
```
Results of this model on the test set of Eyepacs and on Messidor-2 are already present in `results/gls_reg_1e2_resnext50_exp2_Mtask_eyepacs.csv` and `results/gls_reg_1e2_resnext50_exp2_Mtask_messidor2.csv`.



