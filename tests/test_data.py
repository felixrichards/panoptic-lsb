import torch

from data.dataset import LSBInstanceDataset


def test_lsbinstance_load(setup_func):
    class_map = 'basichalosnocompanions'
    dataset = LSBInstanceDataset(setup_func['lsbinstance_survey_dir'], setup_func['lsbinstance_mask_dir'], bands=['g', 'r'], class_map=class_map)
    galaxies = ['NGC2592', 'NGC2594']
    for galaxy in galaxies:
        item = dataset.get_galaxy(galaxy)
    assert item[1].shape[0] == 3
    assert torch.any(item[1][1] > 0) and torch.any(item[1][0] > 0) and torch.any(item[1][2] > 0)
