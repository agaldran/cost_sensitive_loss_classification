# Cost-Sensitive loss for multi-class classification
This is a preliminary repository containing our implementation of cost-sensitive loss functions for classification tasks in pytorch, as presented in:

```
Cost-Sensitive Regularization for Diabetic Retinopathy Grading from Eye Fundus Images
Adrian Galdran, José Dolz, Hadi Chakor, Hervé Lombaert, Ismail Ben Ayed
Medical Image Computing and Computer Assisted Intervention, 2020 (accepted)
```

The proposed idea is quite simple. If you want to penalize different kinds of errors while training your model to perform multi-class classification, you first neeed to encode those penalties into a penalty (or confusion) matrix. In a silly example, imagine you have a problem with `n=3` classes, and you are very worried that your model mis-classifies instances of class `2` as class `0`, but you don't care at all about any of the other possible types of errors. You would build a matrix like the following:

<img src="http://www.sciweavers.org/tex2img.php?eq=%0A%5Cdisplaystyle%0AM%20%3D%20%5Cbegin%7Bbmatrix%7D%0A0%20%26%20%200%20%26%200%5C%5C%0A0%20%26%20%200%20%26%200%5C%5C%0A10%20%26%20%200%20%26%200%5Cend%7Bbmatrix%7D%0A&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0" align="center" border="0" alt="\displaystyleM = \begin{bmatrix}0 &  0 & 0\\0 &  0 & 0\\10 &  0 & 0\end{bmatrix}" width="137" height="65" />


Given an example with a label `y=k` and a one-hot encoded prediction `U(x)=[x_1, x_2, x_3]` (being U your neural network or whatever model you are using), a cost-sensitive loss would be computed simply taking the scalar product of the `k`-th row of `M` and `x`. As you can see, if your example has `y=2` and you have a (correct) prediction `U(x)=[0,0,1]`, then the loss is `l=<[10, 0, 0], [0,0,1]> = 0`. However, if you have an incorrect prediction `U(x)=[1,0,0]`, you get `l=10`. And the funny part, if you have a prediction `U(x)=[0,1,0]`, you still get `l=0`, since you did not penalize this kind of error within `M`.

Enough talking, if you want to use this loss function, you just need to import it and instantiate it as follows:
```
from utils.losses import CostSensitiveLoss

n_classes = 3
criterion = CostSensitiveLoss(n_classes)
```

By default, `criterion` implements a cost matrix that penalizes faraway predictions more than closeby predictions, which is a useful thing to have in image grading/ordinal classification problems:

```
print(criterion.M)
[[0.0000, 0.5000, 1.0000],
[0.5000, 0.0000, 0.5000],
[1.0000, 0.5000, 0.0000]]
```
You can modify that to suit your needs or impose other kind of penalties. All this and more (e.g. how to use this tool to model a-priori inter-observer disagreement knowledge you may have - a confusion matrix for annotators) is explained in the `CS_loss.ipynb` notebook inside this repo.

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

If you find the code here useful in your research, we appreciate if you can cite our work. Thanks!

