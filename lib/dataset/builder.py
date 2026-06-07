import os
import re
import torch
import torchvision.datasets as datasets

from .dataset import ImageNetDataset, TinyImageNetDataset
from .dataloader import fast_collate, DataPrefetcher,fast_collate_timeseries, fast_collate_timeseries_swappedAxes
from .mixup import Mixup
from . import transform

#by me
import os
from pathlib import Path
from typing import cast, Any, Dict, List, Tuple, Optional
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Subset


def _check_torch_version(target='1.7.0'):
    if torch.__version__ == 'parrots':
        return False
    version = re.match('([\d.])*', torch.__version__).group()
    target = re.match('([\d.])*', target).group()
    major, minor, patch = [int(x) for x in version.split('.')[:3]]
    t_major, t_minor, t_patch = [int(x) for x in target.split('.')[:3]]
    if major > t_major:
        return True
    elif major == t_major:
        if minor > t_minor:
            return True
        elif minor == t_minor:
            if patch >= t_patch:
                return True
    return False


# for pytorch>=1.7.0, we add persistent_workers=True in 
# dataloader params
if _check_torch_version('1.7.0'):
    _LOADER_PARAMS = dict(persistent_workers=True)
else:
    _LOADER_PARAMS = dict()


class IndexedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = self.base[idx]
        # common case: (x, y)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            x, y = out
            return x, y, idx
        # if base returns more fields, just append idx
        return (*out, idx)
    
def build_dataloader(args):
    # pre-configuration for the dataset
    if args.dataset == 'imagenet':
        args.data_path = 'data/imagenet' if args.data_path == '' else args.data_path
        args.num_classes = 1000
        args.input_shape = (3, 224, 224)
    elif args.dataset == 'tinyimagenet':
        args.data_path = 'data/tiny-imagenet-200' if args.data_path == '' else args.data_path
        args.num_classes = 200
        args.input_shape = (3, 64, 64)
    elif args.dataset == 'cifar10':
        args.data_path = 'data/cifar' if args.data_path == '' else args.data_path
        args.num_classes = 10
        args.input_shape = (3, 32, 32)
    elif args.dataset == 'cifar100':
        args.data_path = 'data/cifar' if args.data_path == '' else args.data_path
        args.num_classes = 100
        args.input_shape = (3, 32, 32)
    elif args.dataset == 'stl10':
        args.data_path = 'data/stl10' if args.data_path == '' else args.data_path
        args.num_classes = 10
        args.input_shape = (3, 96, 96)

    # train
    if args.dataset == 'imagenet':
        train_transforms_l, train_transforms_r = transform.build_train_transforms(
            args.aa, args.color_jitter, args.reprob, args.remode, args.interpolation, args.image_mean, args.image_std)
        
        imagenet_root = args.data_path

        train_img_root = os.path.join(imagenet_root, 'Data/CLS-LOC/train')
  

        train_meta = '/data/gpfs/projects/punim1910/datasets/imagenet_meta/train_cls.txt'

        train_dataset = ImageNetDataset(
            train_img_root,
            train_meta,
            transform=train_transforms_l
        )
    # if args.dataset == 'imagenet':
    #     train_transforms_l, train_transforms_r = transform.build_train_transforms(
    #         args.aa, args.color_jitter, args.reprob, args.remode, args.interpolation, args.image_mean, args.image_std)
    #     train_dataset = ImageNetDataset(
    #         os.path.join(args.data_path, 'train'), os.path.join(args.data_path, 'meta/train.txt'), transform=train_transforms_l)
    elif args.dataset == 'tinyimagenet':
        train_transforms_l, train_transforms_r = transform.build_train_transforms_tinyimagenet(
            args.aa, args.color_jitter, args.reprob, args.remode, args.interpolation, args.image_mean, args.image_std)
        train_dataset = TinyImageNetDataset(
            root=args.data_path, split='train', transform=train_transforms_l )
    elif args.dataset == 'cifar10':
        train_transforms_l, train_transforms_r = transform.build_train_transforms_cifar10(
            args.cutout_length, args.image_mean, args.image_std)
        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, download=True, transform=train_transforms_l)
    elif args.dataset == 'cifar100':
        train_transforms_l, train_transforms_r = transform.build_train_transforms_cifar10(
            args.cutout_length, args.image_mean, args.image_std)
        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, download=True, transform=train_transforms_l)
                 # apply stratified subsampling if data_ratio < 1.0
        if args.data_ratio < 1.0:
            train_dataset = stratified_subsample(
                train_dataset,
                ratio=args.data_ratio,
                num_classes=args.num_classes,
                seed=args.seed
            )
        print("Train dataset size:-----------", len(train_dataset))
    elif args.dataset == 'stl10':
        train_transforms_l, train_transforms_r = transform.build_train_transforms_stl10(
            aa_config_str=args.aa,
            color_jitter=args.color_jitter,
            reprob=args.reprob,
            remode=args.remode,
            interpolation=args.interpolation,
            mean=args.image_mean,
            std=args.image_std,
        )
        train_dataset = datasets.STL10(
            root=args.data_path, split='train', download=True, transform=train_transforms_l)

    # mixup
    mixup_active = args.mixup > 0. or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_transform = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob,
                                switch_prob=args.mixup_switch_prob, mode=args.mixup_mode, label_smoothing=args.smoothing, num_classes=args.num_classes)
    else:
        mixup_transform = None

    if args.is_indexed:
         train_dataset=IndexedDataset(train_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, 
        pin_memory=False, sampler=train_sampler, collate_fn=fast_collate, drop_last=True, **_LOADER_PARAMS)
    train_loader = DataPrefetcher(train_loader, train_transforms_r, mixup_transform)

    # val
    if args.dataset == 'imagenet':
        val_transforms_l, val_transforms_r = transform.build_val_transforms(
            args.interpolation, args.image_mean, args.image_std
        )

        imagenet_root = args.data_path

        val_meta   = '/data/gpfs/projects/punim1910/datasets/imagenet_meta/val.txt'
        val_img_root   = os.path.join(imagenet_root, 'Data/CLS-LOC/val')
        val_dataset = ImageNetDataset(
            val_img_root,
            val_meta,
            transform=val_transforms_l
        )
    # if args.dataset == 'imagenet':
    #     val_transforms_l, val_transforms_r = transform.build_val_transforms(args.interpolation, args.image_mean, args.image_std)
    #     val_dataset = ImageNetDataset(os.path.join(args.data_path, 'val'), os.path.join(args.data_path, 'meta/val.txt'), transform=val_transforms_l)
    elif args.dataset == 'tinyimagenet':
        val_transforms_l, val_transforms_r = transform.build_val_transforms_tinyimagenet(args.interpolation, args.image_mean, args.image_std)
        val_dataset = TinyImageNetDataset(root=args.data_path, split='val', transform=val_transforms_l )
    elif args.dataset == 'cifar10':
        val_transforms_l, val_transforms_r = transform.build_val_transforms_cifar10(args.image_mean, args.image_std)
        val_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=val_transforms_l)
    elif args.dataset == 'cifar100':
        val_transforms_l, val_transforms_r = transform.build_val_transforms_cifar10(args.image_mean, args.image_std)
        val_dataset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=val_transforms_l)
    elif args.dataset == 'stl10':
        val_transforms_l, val_transforms_r = transform.build_val_transforms_stl10(
            interpolation=args.interpolation,
            mean=args.image_mean,
            std=args.image_std,
        )
        val_dataset = datasets.STL10(
            root=args.data_path, split='test', download=True, transform=val_transforms_l)

    if args.is_indexed:
         val_dataset=IndexedDataset(val_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(args.batch_size * args.val_batch_size_multiplier), 
        shuffle=False, num_workers=args.workers, pin_memory=False, 
        sampler=val_sampler, collate_fn=fast_collate, **_LOADER_PARAMS)
    val_loader = DataPrefetcher(val_loader, val_transforms_r)

    return train_dataset, val_dataset, train_loader, val_loader



def stratified_subsample(dataset, ratio, num_classes, seed=0):
    np.random.seed(seed)

    # get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise ValueError("Dataset has no targets/labels attribute")

    indices = []

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        n_select = int(len(class_idx) * ratio)

        selected = np.random.choice(class_idx, n_select, replace=False)
        indices.extend(selected)

    return Subset(dataset, indices)

#####################
@dataclass
class InputData:
    x: torch.Tensor   # shape: (N, C, T) or (N, T, C)
    y: torch.Tensor   # shape: (N,) with integer class labels

    def split(self, split_size: float, seed: int) -> "Tuple[InputData, InputData]":
        train_x, val_x, train_y, val_y = train_test_split(
            self.x.numpy(),
            self.y.numpy(),
            test_size=split_size,
            stratify=self.y.numpy(),
            random_state=seed)
        
        return (InputData(x=torch.from_numpy(train_x), y=torch.from_numpy(train_y).long()),
                InputData(x=torch.from_numpy(val_x), y=torch.from_numpy(val_y).long()))


UCR_DATASETS = [
    'Haptics', 'Worms', 'Computers', 'UWaveGestureLibraryAll',
    'Strawberry', 'Car', 'BeetleFly', 'wafer', 'CBF', 'Adiac',
    'Lighting2', 'ItalyPowerDemand', 'yoga', 'Trace', 'ShapesAll',
    'Beef', 'MALLAT', 'MiddlePhalanxTW', 'Meat', 'Herring',
    'MiddlePhalanxOutlineCorrect', 'FordA', 'SwedishLeaf',
    'SonyAIBORobotSurface', 'InlineSkate', 'WormsTwoClass', 'OSULeaf',
    'Ham', 'uWaveGestureLibrary_Z', 'NonInvasiveFatalECG_Thorax1',
    'ToeSegmentation1', 'ScreenType', 'SmallKitchenAppliances',
    'WordsSynonyms', 'MoteStrain', 'synthetic_control', 'Cricket_X',
    'ECGFiveDays', 'Wine', 'Cricket_Y', 'TwoLeadECG', 'Two_Patterns',
    'Phoneme', 'MiddlePhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
    'DistalPhalanxTW', 'FacesUCR', 'ECG5000', '50words', 'HandOutlines',
    'Coffee', 'Gun_Point', 'FordB', 'InsectWingbeatSound', 'MedicalImages',
    'Symbols', 'ArrowHead', 'ProximalPhalanxOutlineAgeGroup',
    'SonyAIBORobotSurfaceII', 'ChlorineConcentration', 'Plane', 'Lighting7',
    'PhalangesOutlinesCorrect', 'ShapeletSim', 'DistalPhalanxOutlineAgeGroup',
    'uWaveGestureLibrary_X', 'FaceFour', 'RefrigerationDevices', 'ECG200',
    'ToeSegmentation2', 'CinC_ECG_torso', 'BirdChicken', 'OliveOil',
    'LargeKitchenAppliances', 'uWaveGestureLibrary_Y',
    'NonInvasiveFatalECG_Thorax2', 'FISH', 'ProximalPhalanxOutlineCorrect',
    'Cricket_Z', 'FaceAll', 'StarLightCurves', 'ElectricDevices', 'Earthquakes',
    'DiatomSizeReduction', 'ProximalPhalanxTW'
]


def check_nan_values(data: np.ndarray, dataset_name: str) -> bool:
    has_nan = np.isnan(data).any()
    if has_nan:
        print(f"NaN values found in {dataset_name}.")
    else:
        print(f"No NaN values found in {dataset_name}.")
    return has_nan


def load_ucr_data(args) -> Tuple[InputData, InputData]:
    """
    Loads UCR dataset from:
        args.data_path / {dataset_name}_TRAIN
        args.data_path / {dataset_name}_TEST

    Expected raw format:
        first column  = class label
        remaining cols = time-series values
    """
    dataset_name = args.data_path.name

    train = np.loadtxt(args.data_path / f"{dataset_name}_TRAIN", delimiter=",")
    test = np.loadtxt(args.data_path / f"{dataset_name}_TEST", delimiter=",")

    check_nan_values(train, f"{dataset_name} train")
    check_nan_values(test, f"{dataset_name} test")

    # Raw labels from first column
    y_train_raw = train[:, 0]
    y_test_raw = test[:, 0]

    # Build consistent label mapping using train labels
    unique_classes = np.unique(y_train_raw)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}

    y_train = np.array([class_to_idx[c] for c in y_train_raw], dtype=np.int64)
    y_test = np.array([class_to_idx[c] for c in y_test_raw], dtype=np.int64)

    # UCR is usually univariate: make shape (N, 1, T)
    train_x = torch.from_numpy(train[:, 1:]).float().unsqueeze(1)
    test_x = torch.from_numpy(test[:, 1:]).float().unsqueeze(1)

    train_size, n_chans, seq_len = train_x.shape
    test_size, _, _ = test_x.shape

    class_distribution_train = dict(Counter(y_train.tolist()))
    class_distribution_test = dict(Counter(y_test.tolist()))

    print(
        f"dataset={dataset_name}, seq_len={seq_len}, n_chan={n_chans},"
        f"train_size={train_size}, test_size={test_size}, "
        f"train_class_dist={class_distribution_train}, "
        f"test_class_dist={class_distribution_test}"
    )

    # Update args
    args.num_classes = len(unique_classes)
    args.n_chans = n_chans
    args.seq_len = seq_len

    # Downsample if needed
    if seq_len > args.downsample_size:
        train_x = F.interpolate(
            train_x, size=args.downsample_size, mode="linear", align_corners=True)
        test_x = F.interpolate(
            test_x, size=args.downsample_size, mode="linear", align_corners=True)
        args.seq_len = args.downsample_size
        print(f"Downsampled from {seq_len} to {args.downsample_size}")

    # Optional channel-last format: (N, T, C)
    if args.chan_last:
        train_x = train_x.transpose(1, 2)
        test_x = test_x.transpose(1, 2)

    # Update input shape
    args.input_shape = tuple(train_x.shape[1:])
    print("input_shape:", args.input_shape)

    train_input = InputData(
        x=train_x.contiguous(),
        y=torch.from_numpy(y_train).long())
    test_input = InputData(
        x=test_x.contiguous(),
        y=torch.from_numpy(y_test).long())
 
    return train_input, test_input


def _load_data(args) -> Tuple[InputData, InputData]:
    assert args.dataset in UCR_DATASETS, (
        f"{args.dataset} must be one of the UCR datasets: "
        f"https://www.cs.ucr.edu/~eamonn/time_series_data/"
    )

    args.data_path = Path(args.data_base_folder) / "UCR_TS_Archive_2015" / args.dataset
    print("args.data_path: ",args.data_path)
    train_data, test_data = load_ucr_data(args)
    return train_data, test_data


def build_dataloader_ts(args):
    """
    Returns:
        if args.val_size is not None:
            train_data, val_data, test_data, train_loader, val_loader, test_loader
        else:
            train_data, test_data, train_loader, test_loader
    """
    train_data, test_data = _load_data(args)

    if args.val_size is not None:
        train_data, val_data = train_data.split(args.val_size, args.seed)

    train_dataset = TensorDataset(train_data.x, train_data.y)
    test_dataset = TensorDataset(test_data.x, test_data.y)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        sampler=train_sampler,
        drop_last=False,
        **_LOADER_PARAMS)
    train_loader = DataPrefetcher(train_loader)

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.batch_size * args.val_batch_size_multiplier),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        sampler=test_sampler,
        drop_last=False,
        **_LOADER_PARAMS)
    test_loader = DataPrefetcher(test_loader)

    if args.val_size is not None:
        val_dataset = TensorDataset(val_data.x, val_data.y)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False)

        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size * args.val_batch_size_multiplier),
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            drop_last=False,
            **_LOADER_PARAMS)
        val_loader = DataPrefetcher(val_loader)

        return train_data, val_data, test_data, train_loader, val_loader, test_loader

    return train_data, test_data, train_loader, test_loader

# UCR_DATASETS = ['Haptics', 'Worms', 'Computers', 'UWaveGestureLibraryAll',
#                 'Strawberry', 'Car', 'BeetleFly', 'wafer', 'CBF', 'Adiac',
#                 'Lighting2', 'ItalyPowerDemand', 'yoga', 'Trace', 'ShapesAll',
#                 'Beef', 'MALLAT', 'MiddlePhalanxTW', 'Meat', 'Herring',
#                 'MiddlePhalanxOutlineCorrect', 'FordA', 'SwedishLeaf',
#                 'SonyAIBORobotSurface', 'InlineSkate', 'WormsTwoClass', 'OSULeaf',
#                 'Ham', 'uWaveGestureLibrary_Z', 'NonInvasiveFatalECG_Thorax1',
#                 'ToeSegmentation1', 'ScreenType', 'SmallKitchenAppliances',
#                 'WordsSynonyms', 'MoteStrain', 'synthetic_control', 'Cricket_X',
#                 'ECGFiveDays', 'Wine', 'Cricket_Y', 'TwoLeadECG', 'Two_Patterns',
#                 'Phoneme', 'MiddlePhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
#                 'DistalPhalanxTW', 'FacesUCR', 'ECG5000', '50words', 'HandOutlines',
#                 'Coffee', 'Gun_Point', 'FordB', 'InsectWingbeatSound', 'MedicalImages',
#                 'Symbols', 'ArrowHead', 'ProximalPhalanxOutlineAgeGroup',
#                 'SonyAIBORobotSurfaceII', 'ChlorineConcentration', 'Plane', 'Lighting7',
#                 'PhalangesOutlinesCorrect', 'ShapeletSim', 'DistalPhalanxOutlineAgeGroup',
#                 'uWaveGestureLibrary_X', 'FaceFour', 'RefrigerationDevices', 'ECG200',
#                 'ToeSegmentation2', 'CinC_ECG_torso', 'BirdChicken', 'OliveOil',
#                 'LargeKitchenAppliances', 'uWaveGestureLibrary_Y',
#                 'NonInvasiveFatalECG_Thorax2', 'FISH', 'ProximalPhalanxOutlineCorrect',
#                 'Cricket_Z', 'FaceAll', 'StarLightCurves', 'ElectricDevices', 'Earthquakes',
#                 'DiatomSizeReduction', 'ProximalPhalanxTW']
# @dataclass
# class InputData:
#     x: torch.Tensor
#     y: torch.Tensor

#     def split(self, split_size: float) -> "Tuple[InputData, InputData]":
#         train_x, val_x, train_y, val_y = train_test_split(
#             self.x.numpy(), self.y.numpy(), test_size=split_size, stratify=self.y
#         )
#         return (InputData(x=torch.from_numpy(train_x), y=torch.from_numpy(train_y)),
#                 InputData(x=torch.from_numpy(val_x), y=torch.from_numpy(val_y)))
#    
# def load_ucr_data_bk(args ) -> Tuple[InputData, InputData]:
#     experiment = args.data_path.parts[-1]

#     train = np.loadtxt(args.data_path / f'{experiment}_TRAIN', delimiter=',')
#     test = np.loadtxt(args.data_path / f'{experiment}_TEST', delimiter=',')
    
#     if args.encoder is None:
#         args.encoder = OneHotEncoder(categories='auto', sparse_output=False)
#         y_train = args.encoder.fit_transform(np.expand_dims(train[:, 0], axis=-1))
#     else:
#         y_train = args.encoder.transform(np.expand_dims(train[:, 0], axis=-1))
#     y_test = args.encoder.transform(np.expand_dims(test[:, 0], axis=-1))

#     #check for Nan Values:
#     check_nan_values(train, args.dataset)
#     check_nan_values(test, args.dataset)
    
#     # UCR data is univariate, so an additional dimension is added at index 1 to make it of shape (N, Channels, Seqeunce Length) as the model expects
#     train_x = torch.from_numpy(train[:, 1:]).unsqueeze(1).float()
#     test_x = torch.from_numpy(test[:, 1:]).unsqueeze(1).float()

#     [train_size, n_chan, seq_len] = list(train_x.size())
#     [test_size, n_chan, seq_len] = list(test_x.size())
    
#     #update output classes from the current datatset
#     class_distribution = dict(Counter(train[:, 0]))
#     class_distribution_test = dict(Counter(test[:, 0]))
#     print("database name---",args.dataset, "seq_len:", seq_len,"train_size:", train_size,"test_size:", test_size, "class distribution----:", class_distribution, "class distribution_test----:", class_distribution_test)
    
#     #update args
#     args.num_classes= len(class_distribution)
#     args.n_chans=n_chan
#     args.seq_len=seq_len
        
#     #downsmaple the signals
#     if (seq_len > args.downsample_size):
#         train_x = F.interpolate(train_x, size=args.downsample_size, mode='linear',align_corners=True)
#         test_x = F.interpolate(test_x, size=args.downsample_size, mode='linear',align_corners=True)
#         args.seq_len= args.downsample_size
#         print("downsmaple from size:", seq_len, ", to a new size:", args.downsample_size)
        
#     #if chan_last is True : input tesnsor should be in shape : (batch, seq_len, channels)
#     if args.chan_last: 
#         train_x = torch.swapaxes(train_x, 1,2)
#         test_x = torch.swapaxes(test_x, 1,2)
    
#     #update args
#     [_train_size, dim1, dim2] = list(train_x.size())
#     args.input_shape =  (dim1, dim2)

#     train_input = InputData(x=train_x, y=torch.from_numpy(y_train))
#     test_input = InputData(x=test_x, y=torch.from_numpy(y_test))
#     return train_input, test_input


# #############################
# def _load_data_bk(args) -> Tuple[InputData, InputData]:
#     assert args.dataset in UCR_DATASETS, \
#         f'{args.dataset} must be one of the UCR datasets: ' \
#         f'https://www.cs.ucr.edu/~eamonn/time_series_data/'
#     args.data_path = Path(args.data_base_folder +'/UCR_TS_Archive_2015/' + args.dataset)
#     train, test = load_ucr_data(args)
#     return train, test
    

# ############################
# def build_dataloader_ts_bk(args) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
#     """
#     Arguments
#     ----------
#     args: arguments from parser 

#     Returns
#     ----------
#     Tuple of (train_data, val_data, test_data, train_loader, val_loader, test_loader) if val_size is not None
#     Tuple of (train_data, test_data, train_loader, test_loader) else
#     """
#     train_data, test_data = _load_data(args)

#     if args.val_size is not None: 
#         train_data, val_data = train_data.split(args.val_size)

#     #determine the collate function accordingly
#     if args.chan_last:
#         collate=fast_collate_timeseries_swappedAxes
#     else:
#         collate=fast_collate_timeseries
        
#     train_sampler = torch.utils.data.distributed.DistributedSampler(TensorDataset(train_data.x, train_data.y), shuffle=True)
#     train_loader = torch.utils.data.DataLoader(
#     TensorDataset(train_data.x, train_data.y), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, 
#         pin_memory=False, sampler=train_sampler, collate_fn=collate, drop_last=False, **_LOADER_PARAMS)
#     train_loader = DataPrefetcher(train_loader)

#     test_sampler = torch.utils.data.distributed.DistributedSampler(TensorDataset(test_data.x, test_data.y), shuffle=False)
#     test_loader = torch.utils.data.DataLoader(
#     TensorDataset(test_data.x, test_data.y), batch_size=int(args.batch_size * args.val_batch_size_multiplier), shuffle=False, num_workers=args.workers, 
#         pin_memory=False, sampler=test_sampler, collate_fn=collate, drop_last=False, **_LOADER_PARAMS)
#     test_loader = DataPrefetcher(test_loader)

#     if args.val_size is not None: 
#         val_sampler = torch.utils.data.distributed.DistributedSampler(TensorDataset(val_data.x, val_data.y), shuffle=False)
#         val_loader = torch.utils.data.DataLoader(
#             TensorDataset(val_data.x, val_data.y), batch_size=int(args.batch_size * args.val_batch_size_multiplier), 
#             shuffle=False, num_workers=args.workers, pin_memory=False, 
#             sampler=val_sampler, collate_fn=collate, **_LOADER_PARAMS)
#         val_loader = DataPrefetcher(val_loader)

#         return train_data, val_data, test_data, train_loader, val_loader, test_loader 

#     return train_data, test_data, train_loader, test_loader
        
