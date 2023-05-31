import os
import matplotlib.pyplot as plt
import torch
import gc


def prepare_stuff(outputs, target):
    outputs = outputs[0]
    outputs = {k: v.to('cpu') for k, v in outputs.items()}
    out_masks = outputs['masks'].squeeze(1)
    out_labels = outputs['labels']
    out_boxes = outputs['boxes']
    out_scores = outputs['scores']
    out_N = out_masks.shape[0]
    target = target[0]
    tar_masks = target['masks']
    tar_labels = target['labels']
    tar_boxes = target['boxes']
    tar_N = tar_masks.shape[0]

    tar_labels, indices = torch.sort(tar_labels, dim=0)
    tar_masks = tar_masks[indices]
    tar_boxes = tar_boxes[indices]
    out_labels, indices = torch.sort(out_labels, dim=0)
    out_masks = out_masks[indices]
    out_boxes = out_boxes[indices]

    return [tar_masks, out_masks], [tar_labels, out_labels], [tar_boxes, out_boxes], [tar_N, out_N], out_scores


def plot_preds(model, data_loader_test, dataset_test, device, galaxies=None, ver_dir=None, model_key=''):
    def remove_ticks(*axs):
        for ax in axs:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    classes = dataset_test.classes
    with torch.no_grad():
        model.eval()
        for image, target in data_loader_test:
            idx = target[0]['image_id']
            galaxy = dataset_test.galaxies[idx]
            if galaxies is not None:
                if galaxy not in galaxies:
                    continue

            image = list(img.to(device) for img in image)
            outputs = model(image)

            a_masks, a_labels, a_boxes, a_N, out_scores = prepare_stuff(outputs, target)
            del outputs
            del target

            fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(18, 9))
            fig.suptitle(f'{galaxy}. Annotations vs. predictions: class_map={dataset_test.class_map_key}')
            mask_types = ['Annotation', 'Predicted']
            bands = ['g', 'r']
            band_i = 1
            remove_ticks(*ax[0])
            for k, (ax_k, masks, labels, boxes, N) in enumerate(zip(ax[0], a_masks, a_labels, a_boxes, a_N)):
                img = ((image[0][band_i] - image[0][band_i].min()) / (image[0][band_i].max() - image[0][band_i].min()))
                ax_k.imshow(img.cpu().numpy(), cmap='gray')
                del img
                # ax_k.set_title(f'{mask_types[k]} overlay on {bands[band_i]}-band')
                dataset_test.plot_instance_labels(
                    ax_k,
                    masks,
                    labels,
                    boxes,
                    classes,
                    preds=None if k == 0 else out_scores
                )

            if ver_dir is not None:
                fig_dir = os.path.join('./figs', model_key, ver_dir)
                os.makedirs(fig_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(fig_dir, f'{galaxy}-{bands[band_i]}-preds'),
                    bbox_inches='tight',
                )
                print('saving', os.path.join(fig_dir, f'{galaxy}-{bands[band_i]}-preds'))
                plt.cla()
                fig.clf()
                plt.close('all')
            else:
                plt.show()
            del fig
            gc.collect()
