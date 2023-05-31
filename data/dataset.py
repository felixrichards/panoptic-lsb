import ast
import glob
import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import PIL.Image as Image
import torch
import matplotlib.patheffects as PathEffects

from torchvision import transforms
from torch.utils.data.dataset import Dataset

from data import class_maps


class LSBInstanceDataset(Dataset):
    """
    Dataset class for LSB data w/ instance masks.

    Args:
        survey_dir (str or list of str): Path to survey directory.
        mask_dir (str): Path to mask directory.
        user_weights (str): The user weights used to create consensus masks. Choices are typically 'double' and
            'uniform'. Defaults to 'double'.
        indices (array-like, optional): Indices of total dataset to use.
            Defaults to None.
        num_classes (int, optional): Number of classes. Defaults to 2.
        transform (Trasform, optional): Transform(s) to
            be applied to the data. Defaults to None.
        aug_mult (int, optional): Number of augmentated samples per original sample. Defaults to 1.
        bands (str or list of str, optional): Input spectral bands. Defaults to 'g'.
        padding (int, optional): Amount of padding to apply around each input image. Defaults to 0.
        class_map (str, optional): Which class configuration to use - see data.class_maps for options.
    """
    means = {
        'g': .265,
        'r': .313,
    }
    stds = {
        'g': .753,
        'r': .918,
    }

    def __init__(self, survey_dir: str, mask_dir: str, user_weights: str = 'double', indices=None, num_classes=None,
                 transform=None, aug_mult=1, bands='g', padding=0, class_map=None):
        if user_weights is not None:
            mask_dir = os.path.join(mask_dir, user_weights)
        if class_map is not None:
            mask_dir = os.path.join(mask_dir, class_map)

        self.galaxies, self.img_paths, self.mask_paths = self.load_data(
            survey_dir,
            mask_dir
        )

        if indices is not None:
            self.galaxies = [self.galaxies[i] for i in indices]
            self.img_paths = [self.img_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]

        if type(bands) is str:
            bands = [bands]

        self.bands = bands
        self.num_channels = len(bands)
        self.transform = transform
        self.aug_mult = aug_mult

        self.padding = padding
        self.set_class_map(class_map)
        if num_classes is not None:
            self.num_classes = num_classes  # This gets set by set_class_map so need to set manually if it is passed

    def set_class_map(self, class_map):
        if type(class_map) is str:
            self.classes = class_maps.class_maps[class_map]['classes']
            self.num_classes = len(self.classes) - 1
            self.class_map = class_maps.class_maps[class_map]['idxs']
            self.class_map_key = class_map
            self.class_balances = class_maps.class_maps[class_map]['class_balances']
            self.segment_classes = class_maps.class_maps[class_map]['segment']
            self.detect_classes = class_maps.class_maps[class_map]['detect']
        elif type(class_map) is dict:
            self.class_map = class_map['class_map']
            self.num_classes = max(class_map['class_map'])
            if 'classes' in class_map:
                self.classes = class_map['classes']
            else:
                self.classes = None
            if 'class_balances' in class_map:
                self.class_balances = class_map['class_balances']
            else:
                self.class_balances = [1 for _ in range(self.num_classes)]
            if 'class_map_key' in class_map:
                self.class_map_key = class_map['class_map_key']
            else:
                self.class_map_key = 'custom'
            if 'segment' in class_map:
                self.segment_classes = class_map['segment']
            else:
                self.segment_classes = [True] * self.num_classes
            if 'detect' in class_map:
                self.detect_classes = class_map['detect']
            else:
                self.detect_classes = [True] * self.num_classes
            if 'user_weights' in class_map:
                self.user_weights = class_map['user_weights']
        else:
            self.classes = None
            self.class_map = class_map
            self.class_map_key = 'custom'
            self.class_balances = [None] * self.num_classes
            self.segment_classes = [None] * self.num_classes
            self.detect_classes = [None] * self.num_classes
            self.num_classes = None

    @classmethod
    def bound_object(cls, mask):
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    def bounding_boxes(self, masks, labels):
        num_objs, H, W = masks.shape
        boxes = []
        for i in range(num_objs):
            if self.detect_classes[labels[i] - 1]:
                boxes.append(self.bound_object(masks[i]))
            else:
                boxes.append([0, 0, H, W])

        return torch.tensor(boxes, dtype=torch.float32)

    def handle_transforms(self, image, mask):
        if self.transform is not None:
            t = self.transform(image=image, mask=mask)
            image = t['image']
            mask = t['mask']
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        # albumentations workaround
        if self.padding > 0:
            mask = remove_padding(mask, self.padding)
        image = image.to(torch.float32)
        return image, mask

    @classmethod
    def to_albu(cls, t):
        t = t.transpose((1, 2, 0))
        return t.astype('float32')

    def __getitem__(self, i):
        i = i // self.aug_mult
        img = np.load(self.img_paths[i])

        masks = []
        labels = []
        for class_key, mask_path in self.mask_paths[i].items():
            mask = self.decode_np_mask(np.load(mask_path, allow_pickle=True))[0]
            mask = mask[np.count_nonzero(mask, axis=(1, 2)) > 0]
            masks.append(mask)
            labels += [self.classes.index(class_key)] * mask.shape[0]
        masks = np.concatenate(masks, axis=0)
        labels = torch.tensor(labels)

        img = self.to_albu(img)
        masks = self.to_albu(masks)

        img, masks = self.handle_transforms(img, masks)
        masks = masks.to(torch.uint8)

        # hopefully masks.shape = [N,H,W] - confirm
        boxes = self.bounding_boxes(masks, labels)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # remove bad labels
        masks, boxes, labels, area = masks[area != 0], boxes[area != 0], labels[area != 0], area[area != 0]
        num_objs = labels.shape[0]

        # this is just to stop things breaking - has no significance
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        out = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'area': area,
            'image_id': torch.tensor([i]),
            'iscrowd': iscrowd,
        }

        return (
            img,
            out
        )

    @classmethod
    def decode_filename(cls, path):
        def check_list(item):
            if item[0] == '[' and item[-1] == ']':
                return [i.strip() for i in ast.literal_eval(item)]
            return item
        filename = os.path.split(path)[-1]
        filename = filename[:filename.rfind('.')]
        pairs = [pair.split("=") for pair in filename.split("-")]
        args = {key: check_list(val) for key, val in pairs}
        return args

    @classmethod
    def decode_np_mask(cls, array):
        shape, mask, centre = array['shape'], array['mask'], array['centre']
        mask = np.unpackbits(mask)
        mask = mask[:np.prod(shape)]
        return mask.reshape(shape), centre

    @classmethod
    def load_data(cls, survey_dir, mask_dir):
        all_mask_paths = sorted([
            array for array in glob.glob(os.path.join(mask_dir, '*.npz'))
        ])
        galaxies = []
        img_paths = []
        mask_paths = {}
        for i, mask_path in enumerate(all_mask_paths):
            mask_args = cls.decode_filename(mask_path)
            galaxy = mask_args['name']
            gal_path = os.path.join(
                survey_dir,
                f'name={galaxy}.npy'
            )
            if galaxy in galaxies:
                mask_paths[galaxy][mask_args['class']] = mask_path
            else:
                valid_gal_path = os.path.exists(gal_path)
                if valid_gal_path:
                    galaxies.append(galaxy)
                    img_paths.append(gal_path)
                    mask_paths[galaxy] = {}
                    mask_paths[galaxy][mask_args['class']] = mask_path
        mask_paths = [mask_paths[galaxy] for galaxy in galaxies]

        return galaxies, img_paths, mask_paths

    def get_galaxy(self, galaxy):
        try:
            index = self.galaxies.index(galaxy)
        except ValueError:
            print(f'Galaxy {galaxy} not stored in this dataset.')
            return None
        return self[index * self.aug_mult]

    def plot_galaxy(self, galaxy, include_mask=True, save_fig=None, fits_path=None):
        item = self.get_galaxy(galaxy)
        if fits_path is None:
            gal = self._gal(item)[0]
        else:
            gal = self.get_colour_image(galaxy, fits_path)
        target = item[1]
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(f'class_map={self.class_map_key}')
        ax.imshow(gal)
        ax.set_title(galaxy)

        if include_mask:
            target['labels'], indices = torch.sort(target['labels'], dim=0)
            target['masks'] = target['masks'][indices]
            target['boxes'] = target['boxes'][indices]
            self.plot_instance_labels(
                ax,
                target['masks'],
                target['labels'],
                target['boxes'],
                self.classes,
            )
        if save_fig:
            os.makedirs(save_fig, exist_ok=True)
            plt.savefig(os.path.join(save_fig, f'{galaxy}_instance.png'))
        else:
            plt.show()
        plt.close('all')

    @classmethod
    def plot_instance_labels(cls, ax, masks, labels, boxes, classes, preds=None):
        alpha = .5
        colors = [
            None,
            (.835, .169, 0., alpha),
            (0., .447, .698, alpha),
            (0., .620, .451, alpha),
            (.337, .706, .914, alpha),
            (.902, .624, 0., alpha)
        ]

        for i, class_ in enumerate(labels):
            mask = masks[i].numpy().astype('float32')
            mask[mask < 0.5] = np.nan
            mask[mask >= 0.5] = 1.
            contour = (mask >= 1).astype('uint8') * 255
            contour, _ = cv2.findContours(contour, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            mask_to_plot = mask[:, :, None] * np.full((*mask.shape, 4), colors[class_])
            contour_colour = np.array(list(colors[class_])) * 255
            contour_colour[3] = .9
            cv2.drawContours(mask_to_plot, contour, contourIdx=-1, color=contour_colour, thickness=3)
            del mask
            ax.imshow(mask_to_plot, vmin=0, vmax=1)
            del mask_to_plot
            x, y, x1, y1 = boxes[i]
            box = patches.Rectangle(
                (x, y),
                (x1 - x),
                (y1 - y),
                linewidth=2,
                edgecolor=colors[class_][:3],
                facecolor='none'
            )
            if 'Cirrus' not in classes[class_]:
                ax.add_patch(box)
            if preds is not None:
                text_x = x + 5 + (2 if preds[i].item() < .995 else 0)
                text = f'{preds[i].item():.2f}'
            else:
                text_x = x + 5
                text = f'{classes[class_]}'
            txt = ax.text(
                text_x,
                y1 + 17,
                text,
                color='black',
                weight='roman',
                fontsize=12,
                bbox={
                    'facecolor': colors[class_], 'edgecolor': colors[class_]
                }
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
        ax.invert_yaxis()

    @classmethod
    def _gal(cls, item):
        return item[0]

    @classmethod
    def _mask(cls, item):
        return item[1]['masks']

    def __len__(self):
        return len(self.img_paths) * self.aug_mult


class SynthCirrusDataset(Dataset):
    """Loads cirrus dataset from file.

    Args:
        img_dir (str): Path to dataset directory.
        transform (Trasform, optional): Transform(s) to
            be applied to the data.
        target_transform (Trasform, optional): Transform(s) to
            be applied to the targets.
    """
    def __init__(self, img_dir, indices=None, denoise=False, angle=False,
                 transform=None, target_transform=None, padding=0):
        self.cirrus_paths = [
            img for img in glob.glob(os.path.join(img_dir, 'input/*.png'))
        ]
        if denoise:
            self.mask_paths = [
                img for img in glob.glob(os.path.join(img_dir, 'clean/*.png'))
            ]
        else:
            self.mask_paths = [
                img for img in glob.glob(os.path.join(img_dir, 'cirrus_mask/*.png'))
            ]
        if angle:
            self.angles = torch.tensor(np.load(os.path.join(img_dir, 'angles.npy'))).unsqueeze(1)

        self.num_classes = 2
        self.transform = transform
        self.target_transform = target_transform
        self.angle = angle

        if indices is not None:
            self.cirrus_paths = [self.cirrus_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]

        self.padding = padding

    def __getitem__(self, i):
        cirrus = np.array(Image.open(self.cirrus_paths[i]))
        mask = np.array(Image.open(self.mask_paths[i]))
        if self.transform is not None:
            t = self.transform(image=cirrus, mask=mask)
            cirrus = t['image']
            mask = t['mask']
        cirrus = transforms.ToTensor()(cirrus)
        mask = transforms.ToTensor()(mask)
        # albumentations workaround
        if self.padding > 0:
            mask = remove_padding(mask, self.padding)
        if self.angle:
            return cirrus, mask, self.angles[i]
        return cirrus, mask

    def __len__(self):
        return len(self.cirrus_paths)


def remove_padding(t, p):
    return t[..., p//2:-p//2, p//2:-p//2]
