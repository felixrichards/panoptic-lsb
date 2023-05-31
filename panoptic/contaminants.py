import torch
import torch.nn as nn
import torch.nn.functional as F

from panoptic.attention.models import AttentionMS, MSAttMaskGenerator
from panoptic.attention.loss import GuidedAuxLoss, FocalWithLogitsLoss


class ContaminantClassifier(nn.Module):
    def __init__(self, in_channels, contaminant_idx, output_size=7, score_thresh=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.contaminant_idx = contaminant_idx
        self.downsize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.output_size = output_size

        self.cls_logits = nn.Linear(in_channels, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.score_thresh = score_thresh

    def forward(self, features, targets):
        # get logit for each feature and average
        logits = []
        features = list(features.values())
        for feature in features:
            t = self.downsize(feature)
            t = F.interpolate(t, (self.output_size, self.output_size)).mean(dim=(2, 3))
            logits.append(self.cls_logits(t))
        logits = (sum(logits) / len(logits))

        losses = {}
        if self.training:
            # convert targets to same format as logits
            contaminant_targets = torch.tensor([[torch.any(t['labels'] == self.contaminant_idx)] for t in targets])
            contaminant_targets = contaminant_targets.to(logits.dtype).to(logits.device)

            loss = self.loss(logits, contaminant_targets)
            losses.update(loss)
        else:
            contaminant_targets = None

        proposals = self.prepare_proposals(logits, contaminant_targets)

        return proposals, losses

    def loss(self, outputs, targets):
        # compute bcewithlogitsloss
        loss = self.bce_loss(outputs, targets)
        return {
            'loss_con_classifier': loss
        }

    def prepare_proposals(self, objectness, targets):
        # convert output to same as proposals
        objectness = torch.sigmoid(objectness.clone().detach())

        if targets is None:
            return objectness.squeeze(1)

        # return only true positive proposals
        return torch.logical_and(objectness > self.score_thresh, targets).squeeze(1)


class ContaminantSegmenter(nn.Module):
    def __init__(self, in_channels, contaminant_idx, base_channels=64, scales=[0, 1, 2, 3, 4], score_thresh=0.2525, attention_head=None):
        super().__init__()
        self.contaminant_idx = contaminant_idx
        self.attention_net = AttentionMS(
            in_channels=[256, 256, 256, 256, 256],
            base_channels=base_channels,
            scales=scales,
            attention_head=attention_head
        )
        self.mask_generator = MSAttMaskGenerator(base_channels, 1, scales, pad_to_remove=0)
        self.scales = scales

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.aux_loss = GuidedAuxLoss()
        self.score_thresh = score_thresh

    def forward(self, images, features, targets):
        segs, refined_segs, aux_outs = self.attention_net(features)
        segs, refined_segs = self.mask_generator(images, segs, refined_segs)

        losses = {}
        if self.training:
            segs = segs + refined_segs
            mask_loss = self.mask_loss(segs, targets)
            aux_loss = self.aux_loss(aux_outs)
            losses.update({'loss_con_mask': mask_loss, 'aux_loss': aux_loss})

        return torch.sigmoid(sum(segs) / len(segs)), losses

    def mask_loss(self, outputs, targets):
        targets = self.organise_targets(outputs, targets)
        targets = targets.to(outputs[0].dtype).to(outputs[0].device)

        mask_loss = sum([self.loss_fn(seg, targets) for seg in outputs])
        return mask_loss

    def organise_targets(self, _, targets):
        # find for each target, which idx stores the contaminant mask
        return torch.stack([t['masks'][t['labels'] == self.contaminant_idx] for t in targets])


class ContaminantAloneSegmenter(ContaminantSegmenter):
    def __init__(self, in_channels, contaminant_idx, base_channels=64, scales=[0, 1, 2, 3, 4], attention_head=None):
        super().__init__(in_channels, contaminant_idx, base_channels, scales, attention_head=attention_head)
        self.loss_fn = FocalWithLogitsLoss(reduction='mean')

    def organise_targets(self, outputs, targets):
        # find for each target, which idx stores the contaminant mask
        # if no contaminant, make an empty mask
        return torch.stack([
            t['masks'][t['labels'] == self.contaminant_idx]
            if self.contaminant_idx in t['labels'] else torch.zeros_like(o[0])
            for t, o in zip(targets, outputs)
        ])
