from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.modules.classifier import Classifier as ClassifierBase
from ..modules.grl import GradientReverseLayer


class UnknownClassBinaryCrossEntropy(nn.Module):
    r"""
    Binary cross entropy loss to make a boundary for unknown samples, proposed by
    `Open Set Domain Adaptation by Backpropagation (ECCV 2018) <https://arxiv.org/abs/1804.10427>`_.

    Given a sample on target domain :math:`x_t` and its classifcation outputs :math:`y`, the binary cross entropy
    loss is defined as

    .. math::
        L_{adv}(x_t) = -t log(p(y=C+1|x_t)) - (1-t)log(1-p(y=C+1|x_t))

    where t is a hyper-parameter and C is the number of known classes.

    Args:
        t (float): Predefined hyper-parameter. Default: 0.5

    Inputs:
        - y (tensor): classification outputs (before softmax).

    Shape:
        - y: :math:`(minibatch, C+1)`  where C is the number of known classes.
        - Outputs: scalar

    """
    def __init__(self, t: Optional[float]=0.5):
        super(UnknownClassBinaryCrossEntropy, self).__init__()
        self.t = t

    def forward(self, y):
        # y : N x (C+1)
        softmax_output = F.softmax(y, dim=1)
        unknown_class_prob = softmax_output[:, -1].contiguous().view(-1, 1)
        known_class_prob = 1. - unknown_class_prob

        unknown_target = torch.ones((y.size(0), 1)).to(y.device) * self.t
        known_target = 1. - unknown_target
        return - torch.mean(unknown_target * torch.log(unknown_class_prob + 1e-6)) \
               - torch.mean(known_target * torch.log(known_class_prob + 1e-6))


def H(x):
    return x * torch.log2(x + 1e-10) + (1 - x) * torch.log2(1 - x + 1e-10)


class UnknownClassBinaryCrossEntropyWeight(nn.Module):
    def __init__(self, t: Optional[float]=0.5):
        super(UnknownClassBinaryCrossEntropyWeight, self).__init__()
        self.t = t
    
    def forward(self, y):
        # y : N x (C+1)
        softmax_output = F.softmax(y, dim=1)
        unknown_class_prob = softmax_output[:, -1].contiguous().view(-1, 1)
        known_class_prob = 1. - unknown_class_prob

        unknown_target = torch.ones((y.size(0), 1)).to(y.device) * self.t
        known_target = 1. - unknown_target

        m = H(unknown_class_prob) + 1

        return - torch.mean(m * unknown_target * torch.log(unknown_class_prob + 1e-6)) \
               - torch.mean(m * known_target * torch.log(known_class_prob + 1e-6))


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
        self.grl = GradientReverseLayer()

    def forward(self, x: torch.Tensor, grad_reverse: Optional[bool] = False):
        features = self.backbone(x)
        features = self.bottleneck(features)
        if grad_reverse:
            features = self.grl(features)
        outputs = self.head(features)
        return outputs, features


class LeakySoftmax(nn.Module):
    def __init__(self, coeff=1.0, dim=-1):
        super(LeakySoftmax, self).__init__()
        self.softmax = torch.nn.Softmax(dim=dim)
        self.coeff = coeff
        self.dim = dim

    def forward(self, x):
        shape = list(x.size())
        shape[self.dim] =1
        # print("coeff: ", self.coeff)
        leaky = (torch.ones(*shape, dtype=x.dtype) * np.log(self.coeff))
        concat = torch.cat([x, leaky], dim=self.dim)
        y = self.softmax(concat)
        prob_slicing = [slice(None, None, 1) for i in shape]
        prob_slicing[self.dim] = slice(None, -1, 1)
        prob = y[tuple(prob_slicing)]
        prob_slicing[self.dim] = slice(-1, None, 1)
        total_prob = 1.0 - y[tuple(prob_slicing)]
        return prob, total_prob


class Aug_Cla(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottlenect_dim: Optional[int] = 256, **kwargs):
        super(Aug_Cla, self).__init__(backbone, num_classes)
        classifier_output_dim = num_classes - 1
        self.classifier_auxiliary = nn.Sequential(
            nn.Linear(bottlenect_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, classifier_output_dim),
            LeakySoftmax(classifier_output_dim)
        )

    def forward(self, x, k):
        # print(x)
        y_aug, d_aug = self.classifier_auxiliary(x)
        d_aug = d_aug * k
        return y_aug, d_aug

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [
            {"params": self.classifier_auxiliary.parameters(), "lr": 1.0 * base_lr}
        ]
        return params
