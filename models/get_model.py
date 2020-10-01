import sys
from torchvision.models import resnet as resnet_imagenet
import torch


# Remember to Normalize!
# For models trained on imagenet: transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
# For models trained on cifar10: transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]
def get_arch(model_name, n_classes=3, pretrained=False):
    '''
    Classification options are 'resnet18',  'resnet50',  'resnext50', 'resnext101'; pretrained=False/True
    '''
    mean, std = None, None  # these will only not be None when pretrained==True

    if model_name == 'resnet18':
        model = resnet_imagenet.resnet18(pretrained=pretrained)
        if pretrained: mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else: mean, std = [0.4310, 0.3012, 0.2162], [0.2748, 0.2021, 0.1691]
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

    elif model_name == 'resnet50':
        model = resnet_imagenet.resnet50(pretrained=pretrained)
        if pretrained: mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else: mean, std = [0.4310, 0.3012, 0.2162], [0.2748, 0.2021, 0.1691]
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

    elif model_name == 'resnet50_sws':
        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
        if pretrained: mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else: mean, std = [0.4310, 0.3012, 0.2162], [0.2748, 0.2021, 0.1691]
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

    elif model_name == 'resnext50':
        model = resnet_imagenet.resnext50_32x4d(pretrained=pretrained)
        if pretrained: mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else: mean, std = [0.4310, 0.3012, 0.2162], [0.2748, 0.2021, 0.1691]
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

    elif model_name == 'resnext50_sws':
        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
        if pretrained: mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else: mean, std = [0.4310, 0.3012, 0.2162], [0.2748, 0.2021, 0.1691]
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)

    elif model_name == 'resnext101':
        model = resnet_imagenet.resnext101_32x8d(pretrained=pretrained)
        if pretrained: mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else: mean, std = [0.4310, 0.3012, 0.2162], [0.2748, 0.2021, 0.1691]
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
    else: sys.exit('not a valid model_name, check models.get_model.py')

    return model, mean, std


