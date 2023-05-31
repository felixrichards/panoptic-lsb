import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

import data.training_utils
from active import update_dataset, create_new_dir
from figplot import plot_preds
from data.training_utils import lsb_datasets, construct_dataset
from data import class_maps
from panoptic.optim import create_optim
from panoptic.models import get_model
from panoptic.mrcnnhelper.engine import train_one_epoch, evaluate
from panoptic.mrcnnhelper.utils import collate_fn


def from_checkpoint(path, model, opt):
    checkpoint = torch.load(os.path.join(path, 'checkpoint.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'] + 1


def save_checkpoint(path, model, opt, epoch, model_key, validation=False):
    if validation:
        save_path = os.path.join(path, 'val', f'checkpoint_{epoch=}.pt')
    else:
        save_path = os.path.join(path, 'checkpoint.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'model_key': model_key
    }, save_path)


def get_model_key(args):
    if args.checkpoint_dir:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pt'))
        print(checkpoint.keys())
        if 'model_key' in checkpoint:
            model_key = checkpoint['model_key']
        else:
            model_key = args.model_key
    else:
        model_key = args.model_key
    return model_key


def get_datasets(class_map):
    # verify that class_map can be non-optional
    if class_map in class_maps.class_maps:
        return lsb_datasets(class_map)
    else:
        pass
        # make sure classes is set
        # return coco


def main():
    parser = argparse.ArgumentParser(description='Runs MaskRCNN on lsb annotations.')

    parser.add_argument('--checkpoint_dir',
                        default='', type=str,
                        help='Path to checkpoint directory. (default: %(default)s)')
    parser.add_argument('--survey_dir',
                        default=None, type=str,
                        help='Path to survey images. (default: %(default)s)')
    parser.add_argument('--mask_dir',
                        default=None, type=str,
                        help='Path to annotation masks. (default: %(default)s)')
    parser.add_argument('--model_key',
                        default='2311Attention', type=str,
                        help='Model key. (default: %(default)s)')
    parser.add_argument('--class_map',
                        default='basicnocontaminants',
                        choices=[None, 'coco', *class_maps.class_maps.keys()],
                        help='Which class map to use. (default: %(default)s)')
    parser.add_argument('--active',
                        action='store_true',
                        help='Whether to attempt student teacher framework.')
    parser.add_argument('--active_step',
                        default=25, type=int,
                        help='Number of epochs run before updating dataset. (default: %(default)s)')
    parser.add_argument('--active_start',
                        default=25, type=int,
                        help='Number of epochs run before updating dataset. (default: %(default)s)')
    parser.add_argument('--batch_size',
                        default=1, type=int,
                        help='Training batch size. (default: %(default)s)')

    args = parser.parse_args()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    class_map = args.class_map

    if args.mask_dir is not None or args.survey_dir is not None:
        update_data_dir(args.mask_dir, survey_dir=args.survey_dir)

    # get num_classes
    num_classes = len(class_maps.class_maps[class_map]['classes'])

    print('getting model')
    # get the model using our helper function
    model_key = get_model_key(args)
    print("Loading model", model_key)
    model = get_model(model_key, num_classes)
    is_panoptic = hasattr(model, 'contaminant_segmenter')

    # move model to the right device
    model.to(device)

    # get optim and lr sched
    optimizer, lr_scheduler = create_optim(model, panoptic=is_panoptic)

    # Check if checkpoint is provided
    if args.checkpoint_dir:
        # Load checkpoint
        model_folder = args.checkpoint_dir
        ver_dir = os.path.split(model_folder)[-1]
        start_epoch = from_checkpoint(model_folder, model, optimizer)
        if args.active and start_epoch > args.active_start:
            mask_dir = create_new_dir(
                args.model_key,
                ver_dir,
                (start_epoch - args.active_start) // args.active_step + 1)
            update_data_dir(mask_dir)
    else:
        # Initialise new checkpoint
        model_folder = os.path.join('./models', args.model_key)
        os.makedirs(model_folder, exist_ok=True)
        ver_dir = 'ver' + str(len(glob.glob(os.path.join(model_folder, '*/'))))
        model_folder = os.path.join(model_folder, ver_dir)
        os.makedirs(os.path.join(model_folder, 'val'), exist_ok=True)
        start_epoch = 0

    # dataset_val exists because i'm lazy
    dataset_train, dataset_val, dataset_test = get_datasets(class_map)

    data_loader_train, _, data_loader_test = get_data_loaders(
        dataset_train,
        dataset_val,
        dataset_test,
        batch_size=args.batch_size
    )

    num_epochs = 1
    for epoch in range(start_epoch, num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        save_checkpoint(model_folder, model, optimizer, epoch, args.model_key)
        # update the learning rate
        lr_scheduler.step()

        # check for active learning
        if args.active:
            if epoch % args.active_step == 0 and epoch > args.active_start:
                dataset_num = (epoch - args.active_start) // args.active_step + 1
                # add good false positives to training set
                with torch.no_grad():
                    update_dataset(args, model, ver_dir, dataset_num)
                new_dir = create_new_dir(args.model_key, ver_dir, dataset_num)
                update_data_dir(new_dir)

                # get new dataloaders
                dataset_train, dataset_val, dataset_test = get_datasets(class_map)
                data_loader_train, _, data_loader_test = get_data_loaders(
                    dataset_train,
                    dataset_val,
                    dataset_test,
                    batch_size=args.batch_size
                )
                optimizer, lr_scheduler = create_optim(model)

                plot_new_dataset(new_dir, class_map)

    # Plot predictions
    os.makedirs(os.path.join('./figs', args.model_key), exist_ok=True)
    plot_preds(model, data_loader_test, dataset_test, device, ver_dir=ver_dir, model_key=args.model_key)

    # Evaluate predictions and plot PR-curves
    for lbl in range(1, len(dataset_test.classes)):
        evaluator = evaluate(model, data_loader_test, device=device, classes=[lbl])
        ap_plot(evaluator, dataset_test.classes[lbl], os.path.join('./figs', args.model_key, ver_dir))

    classes = torch.arange(1, len(dataset_test.classes))
    if 'cirrus' in class_map:
        classes = classes[:-1]
        cirrus_iou = calc_iou(model, data_loader_test, device, idx=len(classes) + 1)
        print(f'Final {cirrus_iou=}')

    evaluator = evaluate(model, data_loader_test, device=device, classes=classes)
    ap_plot(evaluator, 'all', os.path.join('./figs', args.model_key, ver_dir))


def calc_iou(model, data_loader_test, device, idx=4):
    def iou(a, b):
        inter = torch.logical_and(a, b).sum()
        union = torch.logical_or(a, b).sum()
        return inter / union

    def get_mask(result):
        cirrus_idxs = result['labels'] == idx
        if not torch.any(cirrus_idxs):
            mask = torch.zeros_like(result['masks'][0])
        else:
            mask = result['masks'][cirrus_idxs][0]
        return mask

    ious = []
    with torch.no_grad():
        model.eval()
        for image, target in data_loader_test:
            image = list(img.to(device) for img in image)
            outputs = model(image)
            pred_mask = get_mask(outputs[0])[0].to(device)
            tar_mask = get_mask(target[0]).to(device)
            ious.append(iou(pred_mask, tar_mask))

    return sum(ious) / len(ious)


def plot_new_dataset(new_dir, class_map):
    dataset_base_dir = '/'.join(new_dir.split(os.sep)[:-1])
    fig_dir = os.path.join(dataset_base_dir, 'figs')
    dataset = construct_dataset(class_map=class_map, transform={'resize': [1024, 1024]})
    for gal in dataset.galaxies:
        dataset.plot_galaxy(gal, save_fig=fig_dir)


def ap_plot(evaluator, class_lbl, fig_dir):
    def create_axes():
        fig, ax = plt.subplots()
        fig.tight_layout()
        # ax.set_xlabel('Recall')
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 6))
        # ax.set_ylabel('Precision')
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(True)
        return fig, ax
    cmap = cm.get_cmap('winter')

    for iou_type, coco_eval in evaluator.coco_eval.items():
        fig, ax = create_axes()

        aind = [coco_eval.params.areaRngLbl.index('all')]
        mind = [coco_eval.params.maxDets.index(100)]
        precisions = coco_eval.eval['precision']
        precisions = precisions[:, :, :, aind, mind].mean(axis=2)
        APs = precisions.mean(axis=1)
        r = np.linspace(0, 1, 101)
        thresholds = np.arange(.5, 1., .05)
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
        colours = [cmap(x) for x in np.linspace(0, 1, 10)]
        for t, p, AP, l, c in zip(thresholds, precisions, APs, linestyles, colours):
            ax.plot(r, p, label=f'AP@{t:.2f} = {AP[0]:.3f}', linestyle=l, color=c)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        legend = ax.legend(loc='center', bbox_to_anchor=(.9, 0.5), fontsize=13)
        legend.set_title(class_lbl, prop={'size': 13})

        plt.savefig(os.path.join(fig_dir, f'AP-{class_lbl}-{iou_type}.pdf'), bbox_inches="tight")


def update_data_dir(mask_dir, survey_dir=None):
    if mask_dir is not None:
        data.training_utils.datasets['instance']['annotations'] = mask_dir
    if survey_dir is not None:
        data.training_utils.datasets['instance']['images'] = survey_dir


def get_data_loaders(dataset_train, dataset_val, dataset_test, batch_size=2):
    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    return data_loader_train, data_loader_val, data_loader_test


if __name__ == '__main__':
    main()
