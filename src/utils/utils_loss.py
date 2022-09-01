import torch
from torch import nn
import torch.nn.functional as F


def get_loss_fn(loss_name="CEL", gamma=1, alpha=1, lb_smooth=0.1, weight=None, reduction="mean"):
    if loss_name in ["cross-entropy", "BCEL", "BCE", "CEL", "CE"]:
        loss_fn = nn.BCEWithLogitsLoss(weight, reduction=reduction)
    elif loss_name in ["focal", "FL"]:
        loss_fn = FocalLoss(gamma, weight, reduction=reduction)
    elif loss_name in ["FLS", "FLwS"]:
        loss_fn = FocalLossWithSmoothing(gamma, lb_smooth, weight, reduction=reduction)
    elif loss_name in ["dice", "DL"]:
        loss_fn = SelfAdjDiceLoss(alpha, gamma, reduction)
    else:
        raise ValueError("loss must be cross-entropy, focal, or dice.")
    return loss_fn


class FocalLoss(nn.Module):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma:int, weight:torch.tensor=None, reduction:str="mean"):
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(weight=weight, reduction="none")

    def forward(self, input_, target):
        cross_entropy = self.bce(input_, target)
        # input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        difficulty_level = self._estimate_difficulty_level(input_, target)
        # print(cross_entropy.shape, difficulty_level.shape)
        loss = difficulty_level * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum' else loss
    
    def _estimate_difficulty_level(self, input_, target):
        """
        :param logits:
        :param label:
        :return:
        """
        prob = F.sigmoid(input_)
        input_prob = target*prob + (1-target)*(1-prob)
        difficulty_level = torch.pow(1 - input_prob, self.gamma)
        return difficulty_level
    

    
class FocalLossWithSmoothing(FocalLoss):
    def __init__(self, gamma:int=1, lb_smooth:float=0.1, weight:torch.Tensor=None, reduction:str="mean"):
        """
        :param gamma:
        :param lb_smooth:
        """
        super(FocalLossWithSmoothing, self).__init__(gamma, weight, reduction)
        self.lb_smooth = lb_smooth


    def forward(self, input_, target):
        """
        :param logits: (batch_size,)
        :param label:
        :return:
        """
        difficulty_level = self._estimate_difficulty_level(input_, target)

        with torch.no_grad():
            target = target.clone().detach()
            lb_target = target*(target-self.lb_smooth) + (1-target)*self.lb_smooth
            
        cross_entropy = self.bce(input_, lb_target)
        loss = difficulty_level * cross_entropy
        return torch.mean(loss) if self._reduction == 'mean' else torch.sum(loss) if self._reduction == 'sum' else loss
    
    
    
class SelfAdjDiceLoss(torch.nn.Module):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)
    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, alpha:float=1.0, gamma:float=1.0, reduction:str="mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.gamma) / (probs_with_factor + 1 + self.gamma)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")