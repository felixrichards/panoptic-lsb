## Overview

This repo provides source code for the paper "Panoptic Segmentation of Galactic Structures in LSB Images", implementing instance & panoptic segmentation on MATLAS data, and a HITL training procedure.

## Installation

Install [poetry](https://python-poetry.org/docs/). Run
```
poetry install
```

If you have a GPU,
```
poetry run pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Otherwise,
```
poetry run pip install torch==1.10.0+cpu torchvision==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```


## Class maps

Class maps describe how raw LSB annotations, from the annotation tool, are converted into training labels. All different maps can be found in data/class_maps.py:
- the keys of the dict relate to the valid values of class_map that can be passed into run.py
- the 'classes' key of each class map describes the annotation classes present

## Example commands

Segment galaxies, diffuse halos, tidal structures, ghosted halos with MaskRCNN
```
python ./run.py --model_key=2311Pretrained --class_map=basichalosnocompanions
```

Segment galaxies, diffuse halos, tidal structures, ghosted halos and cirrus with panoptic network (MaskRCNN + Gridded Gabor Attention)
```
python ./run.py --model_key=0504PreContaminantGabor --class_map=basichaloscirrusnocompanions
```
