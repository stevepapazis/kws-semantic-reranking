from utils.gw_dataset import GWDataset
from utils.iam_dataset import IAMDataset



def get_dataset(dataset_path, dataset_name, subset="all", test_fold=None,  segmentation_level='form'):
    if dataset_name == 'iam':
        dataset_constructor = IAMDataset
    elif dataset_name == 'gw':
        dataset_constructor = lambda *_args, **_kwargs: GWDataset(*_args, **_kwargs, fold=test_fold)
    else:
        raise NotImplementedError
    return dataset_constructor(dataset_path, subset=subset, segmentation_level='form', fixed_size=None, transforms=None)
