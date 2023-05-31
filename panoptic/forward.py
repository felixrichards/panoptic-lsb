import warnings

from collections import OrderedDict
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn.functional as F

from torch import Tensor


def panoptic_forward(self, images, targets=None):
    # type: (None, List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).

    """
    if self.training and targets is None:
        raise ValueError("In training mode, targets should be passed")
    if self.training:
        assert targets is not None
        for target in targets:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                    raise ValueError("Expected target boxes to be a tensor"
                                     "of shape [N, 4], got {:}.".format(
                                            boxes.shape))
            else:
                raise ValueError("Expected target boxes to be of type "
                                 "Tensor, got {:}.".format(type(boxes)))

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = self.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError("All bounding boxes should have positive height and width."
                                 " Found invalid box {} for target at index {}."
                                 .format(degen_bb, target_idx))

    features = self.backbone(images.tensors)

    if 'aux_outs' in features:
        aux_outs = features.pop('aux_outs')
    else:
        aux_outs = None

    if isinstance(features, torch.Tensor):
        features = OrderedDict([('0', features)])

    losses = {}
    if hasattr(self, 'contaminant_segmenter'):
        if hasattr(self, 'contaminant_classifier'):
            contaminant_proposals, con_classifier_losses = self.contaminant_classifier(features, targets)
            # filter out samples with no contaminants in targets also
            to_mask = contaminant_proposals > self.contaminant_classifier.score_thresh
            contaminant_masks = None
            if torch.any(to_mask):
                contaminant_masks, con_mask_losses = self.contaminant_segmenter(
                    images.tensors[to_mask],
                    [f[to_mask] for f in list(features.values())],
                    [t for i, t in enumerate(targets) if to_mask[i]] if targets is not None else None,
                )
                losses.update(con_mask_losses)
            losses.update(con_classifier_losses)
            con_bboxes = None
        else:
            # do alone contaminant segmenter stuff
            contaminant_masks, con_mask_losses = self.contaminant_segmenter(
                images.tensors,
                list(features.values()),
                targets,
            )
            contaminant_proposals, con_bboxes = fake_proposals(contaminant_masks)
            losses.update(con_mask_losses)

        # remove contaminant labels/masks
        if self.training:
            for t in targets:
                to_keep = t['labels'] != self.contaminant_segmenter.contaminant_idx
                t['labels'] = t['labels'][to_keep]
                t['masks'] = t['masks'][to_keep]
                t['boxes'] = t['boxes'][to_keep]
                t['area'] = t['area'][to_keep]
                t['iscrowd'] = t['iscrowd'][to_keep]

    proposals, proposal_losses = self.rpn(images, features, targets)

    detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
    detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

    losses.update(detector_losses)
    losses.update(proposal_losses)

    if aux_outs is not None:
        aux_loss = self.aux_loss(aux_outs)
        losses.update({'aux_loss': aux_loss})

    if hasattr(self, 'contaminant_segmenter') and not self.training:
        detections = postprocess_contaminants(
            detections,
            contaminant_proposals,
            contaminant_masks,
            original_image_sizes,
            self.contaminant_segmenter.score_thresh,
            self.contaminant_segmenter.contaminant_idx,
            bboxes=con_bboxes
        )

    if torch.jit.is_scripting():
        if not self._has_warned:
            warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
            self._has_warned = True
        return losses, detections
    else:
        return self.eager_outputs(losses, detections)


def fake_proposals(masks):
    # create proposals with same format as outputs from contaminant classifier, that match cirrus masks
    masks = masks.detach()
    bboxes = [bound_object(mask[0] > 0.5) for mask in masks]
    objectness = torch.stack([
        torch.mean(mask[0]) if bbox is not None else torch.tensor(0.).to(mask.device)
        for bbox, mask in zip(bboxes, masks)
    ])
    return objectness, bboxes


def bound_object(mask):
    if not torch.any(mask):
        return None
    pos = torch.where(mask)
    xmin = torch.min(pos[1])
    xmax = torch.max(pos[1])
    ymin = torch.min(pos[0])
    ymax = torch.max(pos[0])
    return [xmin, ymin, xmax, ymax]


def postprocess_contaminants(detections, proposals, masks, sizes, score_thresh, cont_idx, bboxes=None):
    # how to handle false positive proposals? should a mask object be included
    for prop_i in (proposals > score_thresh).nonzero():
        if bboxes is None:
            bbox = [0, 0, *sizes[prop_i]]
        else:
            bbox = bboxes[prop_i]
        device = detections[prop_i]['boxes'].device
        detections[prop_i]['boxes'] = torch.cat((
            detections[prop_i]['boxes'],
            torch.tensor([bbox]).to(device)
        ), dim=0)
        detections[prop_i]['labels'] = torch.cat((
            detections[prop_i]['labels'],
            torch.tensor([cont_idx]).to(device)
        ), dim=0)
        detections[prop_i]['scores'] = torch.cat((
            detections[prop_i]['scores'],
            torch.tensor([proposals[prop_i]]).to(device)
        ), dim=0)
        detections[prop_i]['masks'] = torch.cat((
            detections[prop_i]['masks'],
            F.interpolate(masks[prop_i], sizes[prop_i], mode='bilinear') * 4
        ), dim=0)
    return detections
