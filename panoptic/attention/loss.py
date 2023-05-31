import torch
import torch.nn as nn


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=2., gamma=2., reduction='none'):
        super().__init__()
        self.alpha = 1 / (1 + pos_weight)
        self.gamma = gamma
        self.eps = 1e-6
        self.reduction = reduction

    def forward(self, output, target):
        prob = torch.sigmoid(output).clip(self.eps, 1. - self.eps)

        # not perfect, consider reimplementing with log_softmax for numerical stability
        back_ce = -(1 - target) * torch.pow(prob, self.gamma) * torch.log(1 - prob)
        fore_ce = -target * torch.pow(1 - prob, self.gamma) * torch.log(prob)

        out = (1 - self.alpha) * back_ce + self.alpha * fore_ce
        if self.reduction == 'none':
            return out
        return out.mean()


class ConsensusLoss(nn.Module):
    def __init__(self, eta=.4, lambd=1.1, beta=None, pos_weight=None,
                 seg_criterion=FocalWithLogitsLoss(reduction='none')):
        super().__init__()
        self.eta = eta
        self.beta = beta
        self.lambd = lambd
        self.seg_criterion = seg_criterion

    def forward(self, output, target):
        weight = self.loss_weight(target)
        target = torch.where(target >= self.eta, 1., 0.)
        return (self.seg_criterion(output, target) * weight).mean()

    def loss_weight(self, target):
        p = torch.sum(target >= self.eta)
        n = torch.sum(target == 0)
        weight = torch.zeros_like(target)
        weight[target >= self.eta] = n / (p + n)
        weight[target == 0] = self.lambd * p / (p + n)
        return weight


class ConsensusLossMC(ConsensusLoss):
    def __init__(self, eta=.49, **kwargs):
        super().__init__(eta=eta, **kwargs)

    def loss_weight(self, target):
        weight = torch.tensor([1], device=target.device)[None, :, None, None]
        weight = weight.expand_as(target) * torch.logical_or(target == 0., target > 0.49)

        return weight


class GuidedAuxLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss()):
        super().__init__()
        self.criterion = criterion

    def forward(self, aux_outputs):
        guided_losses, reconstruction_losses = zip(*[self.aux_guided_loss(aux) for aux in aux_outputs])
        return .25 * sum(guided_losses) + .1 * sum(reconstruction_losses)

    def aux_guided_loss(self, aux):
        return (
            self.criterion(aux['in_semantic_vectors'], aux['out_semantic_vectors']),
            self.criterion(aux['in_attention_encodings'], aux['out_attention_encodings'])
        )


class DAFLoss(nn.Module):
    def __init__(self,
                 seg_criterion=nn.BCEWithLogitsLoss(),
                 aux_loss=GuidedAuxLoss(),
                 pos_weight=None):
        super().__init__()
        self.seg_criterion = seg_criterion
        if pos_weight is not None:
            self.seg_criterion.pos_weight = pos_weight
        self.aux_loss = aux_loss

    def forward(self, output, target):
        if not (type(output[0]) == tuple or type(output[0]) == list):
            return self.seg_criterion(output, target)

        segmentations, aux_outputs = output

        seg_losses = [
            self.seg_criterion(seg, target)
            for seg in segmentations
        ]

        if all(o is None for o in aux_outputs):
            return sum(seg_losses)

        aux_loss = self.aux_loss(aux_outputs)
        return sum(seg_losses) + aux_loss


class DAFConsensusLoss(DAFLoss):
    """Loss class for DAF network on multiclass multilabel

    Args:
        seg_criterion (nn._loss, optional): loss fn used for segmentation
            labels.
        aux_loss (nn._loss, optional): loss fn used for attention
            vectors in self-guided protocol.
    """
    def __init__(self,
                 consensus_criterion=ConsensusLossMC(eta=.49, lambd=1.1, beta=None),
                 aux_loss=GuidedAuxLoss(),
                 pos_weight=None):
        super().__init__(
            seg_criterion=consensus_criterion,
            aux_loss=aux_loss,
            pos_weight=pos_weight
        )
