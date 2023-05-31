import torch


MASK_RCNN_OPTIM_PARAMS = {'lr': 0.01, 'weight_decay': 5e-4, 'momentum': 0.9}
ATTENTION_OPTIM_PARAMS = {'lr': 1e-3, 'weight_decay': 5e-7}

MASK_RCNN_LR_SCHEDULER_PARAMS = {'step_size': 25, 'gamma': 0.5}
ATTENTION_LR_SCHEDULER_PARAMS = {'lr_decay': 0.98}


class PanopticOptimiser(torch.optim.Optimizer):
    def __init__(self, model):
        # create list of dicts separating mask rcnn weights and attention weights
        attention_weights = [p for p in model.contaminant_segmenter.parameters() if p.requires_grad]
        maskrcnn_weights = [p for p in model.parameters() if p.requires_grad]
        maskrcnn_weights = [p for p in maskrcnn_weights if not any(p is q for q in attention_weights)]

        self.mask_rcnn_optimizer = torch.optim.SGD(maskrcnn_weights, **MASK_RCNN_OPTIM_PARAMS)
        self.attention_optimizer = torch.optim.Adam(maskrcnn_weights, **ATTENTION_OPTIM_PARAMS)

    @torch.no_grad()
    def step(self):
        self.mask_rcnn_optimizer.step()
        self.attention_optimizer.step()


class PanopticLRScheduler():
    def __init__(self, optimizer: PanopticOptimiser):
        self.mask_rcnn_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer.mask_rcnn_optimizer,
            **MASK_RCNN_LR_SCHEDULER_PARAMS
        )
        self.attention_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer.attention_optimizer,
            **ATTENTION_LR_SCHEDULER_PARAMS
        )

    def step(self):
        self.mask_rcnn_lr_scheduler.step()
        self.attention_lr_scheduler.step()


def create_optim(model, panoptic=False):
    # construct an optimizer and a learning rate scheduler
    if panoptic:
        optimiser = PanopticOptimiser(model)
        return optimiser, PanopticLRScheduler(optimiser)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=25,
                                                   gamma=0.5)
    return optimizer, lr_scheduler
