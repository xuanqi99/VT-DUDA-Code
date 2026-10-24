from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.modules.classifier import Classifier as ClassifierBase
from ..modules.entropy import entropy


__all__ = ['MinimumClassConfusionLoss', 'ImageClassifier']


class MinimumClassConfusionLoss(nn.Module):
    r"""
    Minimum Class Confusion loss minimizes the class confusion in the target predictions.

    You can see more details in `Minimum Class Confusion for Versatile Domain Adaptation (ECCV 2020) <https://arxiv.org/abs/1912.03699>`_

    Args:
        temperature (float) : The temperature for rescaling, the prediction will shrink to vanilla softmax if
          temperature is 1.0.

    .. note::
        Make sure that temperature is larger than 0.

    Inputs: g_t
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`

    Shape:
        - g_t: :math:`(minibatch, C)` where C means the number of classes.
        - Output: scalar.

    Examples::
        >>> temperature = 2.0
        >>> loss = MinimumClassConfusionLoss(temperature)
        >>> # logits output from target domain
        >>> g_t = torch.randn(batch_size, num_classes)
        >>> output = loss(g_t)

    MCC can also serve as a regularizer for existing methods.
    Examples::
        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> temperature = 2.0
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim, hidden_size=1024)
        >>> cdan_loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> mcc_loss = MinimumClassConfusionLoss(temperature)
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> total_loss = cdan_loss(g_s, f_s, g_t, f_t) + mcc_loss(g_t)
    """

    def __init__(self, temperature: float):
        super(MinimumClassConfusionLoss, self).__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        if batch_size == 0:
            return logits.sum() * 0.0                           

        predictions = F.softmax(logits / self.temperature, dim=1)          

                                                             
        with torch.no_grad():
            entropy_weight = entropy(predictions)
            entropy_weight = 1.0 + torch.exp(-entropy_weight)
            entropy_weight = (batch_size * entropy_weight / (torch.sum(entropy_weight) + 1e-12)).unsqueeze(1)         

                                 
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions)

                                                                                          
        class_confusion_matrix = class_confusion_matrix / (class_confusion_matrix.sum(dim=1, keepdim=True) + 1e-12)

        mcc_loss = (class_confusion_matrix.sum() - torch.trace(class_confusion_matrix)) / float(num_classes)
        return mcc_loss


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
                                                       
                           
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
