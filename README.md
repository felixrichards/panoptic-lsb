## Overview

This repo implements instance & panoptic segmentation on MATLAS data, and a HITL trianing procedure.

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
