import pytest


@pytest.fixture(scope='session')
def setup_func(request):
    survey_dir = "E:/Matlas Data/FITS/matlas"
    mask_dir = "E:/MATLAS Data/annotations/drawtest3009"

    lsb_survey_dir = "E:/Matlas Data/np"
    lsb_mask_dir = "E:/MATLAS Data/annotations/consensus"

    lsbinstance_mask_dir = "E:/MATLAS Data/annotations/instance_combined"
    lsbinstance_survey_dir = "E:/Matlas Data/np_surveys/05_1024"

    return locals()
