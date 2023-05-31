import argparse
import glob
import os
import torch
import matplotlib.pyplot as plt

from data.dataset import LSBInstanceDataset
from data.training_utils import lsb_datasets, construct_dataset


def main():
    parser = argparse.ArgumentParser(description='Plots active datasets.')

    # parser.add_argument('--base_dir',
    #                     default='', type=str,
    #                     help='Path to base dataset directory. (default: %(default)s)')
    parser.add_argument('--active_dir',
                        default='', type=str,
                        help='Path to active datasets directory. (default: %(default)s)')
    parser.add_argument('--class_map',
                        default='basicnocontaminantsnocompanions',
                        choices=[None, 'coco', *LSBInstanceDataset.class_maps.keys()],
                        help='Which class map to use. (default: %(default)s)')

    args = parser.parse_args()

    base_set = construct_dataset(class_map=args.class_map, transform={'resize': [1024, 1024]})

    active_sets = glob.glob(os.path.join(args.active_dir, '*/'))
    active_sets = [construct_dataset(
        class_map=args.class_map,
        transform={'resize': [1024, 1024]},
        ann_path=os.path.join(set_path, 'annotations')) for set_path in active_sets]

    all_sets = [base_set] + active_sets

    set_labels = ['Base dataset'] + [f'Updated dataset {num + 1}' for num in range(len(active_sets))]
    pred_labels = [f'Correct false positives after {epochs} epochs' for epochs in range(25, 200, 5)]

    fig_dir = os.path.join(args.active_dir, '../figs/')
    os.makedirs(fig_dir, exist_ok=True)

    for gal in base_set.galaxies:
        # create Nx2 plot
        fig, ax = plt.subplots(len(active_sets), 2, squeeze=False, figsize=(7, 11))
        fig.suptitle(f'{gal}. Active learning datasets: class_map={base_set.class_map_key}')
        all_idxs = []
        for i, (ax_i, before_set, after_set) in enumerate(zip(ax, all_sets[:-1], all_sets[1:])):
            ax_i[0].set_title(set_labels[i])
            ax_i[1].set_title(pred_labels[i])
            image, before_target = before_set.get_galaxy(gal)
            _, after_target = after_set.get_galaxy(gal)

            ax_i[0].imshow(image[1].cpu().numpy())
            ax_i[1].imshow(image[1].cpu().numpy())
            plot_stuff(before_set, before_target, ax_i[0])

            idxs = identify_new_labels(before_target, after_target)
            print(idxs)
            all_idxs.append(idxs)

            plot_stuff(after_set, after_target, ax_i[0], idxs=idxs)

        if any(all_idxs):
            fig.savefig(
                os.path.join(fig_dir, f'{gal}-r-active'),
                bbox_inches='tight',
            )


def identify_new_labels(before_target, after_target):
    def iou(a, b):
        inter = torch.logical_and(a, b).sum()
        union = torch.logical_or(a, b).sum()
        return inter / union

    idxs = []
    for i, mask in enumerate(after_target['masks']):
        ious = [iou(before_target['masks'][j], mask) for j in range(before_target['masks'].shape[0]) if before_target['labels'][j] == after_target['labels'][i]]
        ious = torch.stack(ious)
        if not torch.any(ious > .1):
            idxs.append(i)

    return idxs


def plot_stuff(dataset, stuff, ax, idxs=None):
    if idxs is not None:
        stuff['labels'] = stuff['labels'][idxs]
        stuff['masks'] = stuff['masks'][idxs]
        stuff['boxes'] = stuff['boxes'][idxs]

    labels, indices = torch.sort(stuff['labels'], dim=0)
    masks = stuff['masks'][indices]
    boxes = stuff['boxes'][indices]
    dataset.plot_instance_labels(
        ax,
        masks,
        labels,
        boxes,
        dataset.classes,
    )

if __name__ == '__main__':
    main()
