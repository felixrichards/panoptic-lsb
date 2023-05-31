import os
import glob
import shutil

import numpy as np
import torch
import matplotlib.pyplot as plt

import data.training_utils
from data.dataset import LSBInstanceDataset
from data.training_utils import construct_dataset


def update_dataset(args, model, ver_dir, dataset_num):
    def iou(a, b):
        inter = torch.logical_and(a, b).sum()
        union = torch.logical_or(a, b).sum()
        return inter / union

    # copy dataset
    dataset_temp = construct_dataset(class_map=args.class_map, transform={'resize': [1024, 1024]})
    if dataset_num == 1:
        old_dir = data.training_utils.datasets['instance']['annotations']
    else:
        old_dir = create_new_dir(args.model_key, ver_dir, dataset_num - 1)
    old_dir = exact_dir(old_dir, args.class_map)

    new_dir = create_new_dir(args.model_key, ver_dir, dataset_num)
    new_dir = exact_dir(new_dir, args.class_map)

    copy_dataset(old_dir, new_dir)

    # loop through gals
    for (image, target), gal in zip(dataset_temp, dataset_temp.galaxies):
        # generate predictions on unaugmented training set
        model.eval()
        preds = model(image.unsqueeze(0).cuda())[0]
        target['masks'] = target['masks'].cuda()
        target['labels'] = target['labels'].cuda()
        target['boxes'] = target['boxes'].cuda()
        # loop through preds
        for i in range(preds['masks'].shape[0]):
            # check if mask does not have high overlap with any target mask
            if torch.any(target['labels'] == preds['labels'][i]) and preds['labels'][i] != 2:
                ious = [iou(target['masks'][j], preds['masks'][i]) for j in range(target['masks'].shape[0]) if target['labels'][j] == preds['labels'][i]]
                ious = torch.stack(ious)
                if not torch.any(ious > .1):
                    # plot it over image
                    plot_stuff(image, target, preds, i, gal, dataset_temp.classes)
                    # wait for user input
                    if grab_user_input():
                        # save into dataset
                        save_pred(new_dir, preds, target, i, dataset_temp.classes, gal)


def save_pred(new_dir, preds, target, i, classes, galaxy):
    target_masks = target['masks'][preds['labels'][i] == target['labels']]
    pred_mask = preds['masks'][i]
    new_mask = torch.cat((target_masks, pred_mask), dim=0).cpu().numpy()
    new_mask = (new_mask > .5).astype(bool)
    np.savez(
        os.path.join(new_dir, f"name={galaxy}-class={classes[preds['labels'][i]]}"),
        shape=new_mask.shape,
        centre=None,
        mask=np.packbits(new_mask)
    )


def plot_stuff(image, target, preds, i, galaxy, classes):
    preds = {
        'masks': preds['masks'][i],
        'boxes': preds['boxes'][i].unsqueeze(0),
        'labels': preds['labels'][i].unsqueeze(0)
    }
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(18, 9))
    fig.suptitle(f'{galaxy}. Annotations vs. predictions {i}')

    for k, (ax_k, stuff) in enumerate(zip(ax[0], (target, preds))):
        ax_k.imshow(image[1].cpu().numpy())
        labels, indices = torch.sort(stuff['labels'], dim=0)
        masks = stuff['masks'][indices]
        boxes = stuff['boxes'][indices]
        LSBInstanceDataset.plot_instance_labels(
            ax_k,
            masks.cpu(),
            labels.cpu(),
            boxes.cpu(),
            classes
        )
    plt.show()


def grab_user_input():
    def valid(x):
        if x.lower() in ('y', 'yes', 'true'):
            return True
        if x.lower() in ('n', 'no', 'false'):
            return False
        return None

    loop = True
    while loop:
        answer = valid(input("Keep mask? (y/n) "))
        loop = answer is None
    return answer


def copy_dataset(old_dir, new_dir):
    os.makedirs(new_dir, exist_ok=True)
    from_files = glob.glob(os.path.join(old_dir, '*.*'))
    to_files = [os.path.join(new_dir, os.path.split(path)[-1]) for path in from_files]
    for ff, tf in zip(from_files, to_files):
        shutil.copy(ff, tf)
    return new_dir


def create_new_dir(model_key, ver_dir, num):
    new_dir = f'E:/MATLAS Data/annotations/active_sets/{model_key}/{ver_dir}/{num}/annotations/'
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def exact_dir(data_dir, class_map):
    return os.path.join(data_dir, f"double/{class_map}")
