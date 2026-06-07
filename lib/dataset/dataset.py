import io
import torch
import warnings
from PIL import Image
from torch.utils.data import Dataset
import os
import torch.nn.functional as F



try:
    import mc
    from .file_io import PetrelMCBackend
    _has_mc = True
except ModuleNotFoundError:
    warnings.warn('mc module not found, using original '
                  'Image.open to read images')
    _has_mc = False

import glob

class ImageNetDataset(Dataset):
    def __init__(self, root, meta_file, transform=None, num_classes=1000, one_hot=True):
        self.root = root
        self.transform = transform
        self.num_classes = num_classes
        self.one_hot = one_hot

        if _has_mc:
            with open('./data/mc_prefix.txt', 'r') as f:
                prefix = f.readline().strip()
            self.root = prefix + '/' + ('train' if 'train' in self.root else 'val')

        with open(meta_file) as f:
            meta_list = f.readlines()

        raw_metas = []
        class_names = []

        for line in meta_list:
            parts = line.strip().split()
            path = parts[0]

            # Case 1: train path like n01440764/n01440764_10026
            if "/" in path:
                class_name = path.split("/")[0]
                rel_path = path

            # Case 2: val path like ILSVRC2012_val_00000001
            else:
                matches = glob.glob(os.path.join(self.root, "*", path + "*"))

                if len(matches) == 0:
                    raise FileNotFoundError(f"Cannot find image for meta path: {path}")

                full_path = matches[0]
                class_name = os.path.basename(os.path.dirname(full_path))
                rel_path = os.path.relpath(full_path, self.root)

            raw_metas.append((rel_path, class_name))
            class_names.append(class_name)

        unique_classes = sorted(set(class_names))
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}

        self.metas = [
            (rel_path, class_to_idx[class_name])
            for rel_path, class_name in raw_metas
        ]

        self.num = len(self.metas)

        labels_after = [cls for _, cls in self.metas]
        print(
            "ImageNet labels after mapping:",
            min(labels_after),
            max(labels_after),
            len(set(labels_after))
        )

        assert min(labels_after) >= 0
        assert max(labels_after) < self.num_classes

        self._mc_initialized = False

    def __len__(self):
        return self.num

    def _init_memcached(self):
        if not self._mc_initialized:
            self.backend = PetrelMCBackend()
            self._mc_initialized = True

    def __getitem__(self, index):
        rel_path, cls = self.metas[index]
        filename = os.path.join(self.root, rel_path)

        if not os.path.exists(filename):
            if os.path.exists(filename + ".JPEG"):
                filename = filename + ".JPEG"
            elif os.path.exists(filename + ".jpg"):
                filename = filename + ".jpg"
            elif os.path.exists(filename + ".jpeg"):
                filename = filename + ".jpeg"
            else:
                raise FileNotFoundError(f"Image file not found: {filename}")

        if _has_mc:
            self._init_memcached()
            buff = self.backend.get(filename)
            with Image.open(buff) as img:
                img = img.convert("RGB")
        else:
            img = Image.open(filename).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        cls = torch.tensor(cls, dtype=torch.long)

        if self.one_hot:
            cls = F.one_hot(cls, num_classes=self.num_classes).float()

        return img, cls

class ImageNetDatasetbk(Dataset):
    r"""
    Dataset using memcached to read data.

    Arguments
        * root (string): Root directory of the Dataset.
        * meta_file (string): The meta file of the Dataset. Each line has a image path
          and a label. Eg: ``nm091234/image_56.jpg 18``.
        * transform (callable, optional): A function that transforms the given PIL image
          and returns a transformed image.
    """
    def __init__(self, root, meta_file, transform=None):
        self.root = root
        if _has_mc:
            with open('./data/mc_prefix.txt', 'r') as f:
                prefix = f.readline().strip()
            self.root = prefix + '/' + \
                ('train' if 'train' in self.root else 'val')
        self.transform = transform
        with open(meta_file) as f:
            meta_list = f.readlines()
        self.num = len(meta_list)
        self.metas = []
        # for line in meta_list:
        #     path, cls = line.strip().split()
        #     self.metas.append((path, int(cls)))
        labels = []

        for line in meta_list:
            path, cls = line.strip().split()
            cls = int(cls)
            labels.append(cls)
            self.metas.append((path, cls))

        # remap labels → 0...999
        unique_labels = sorted(set(labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}

        self.metas = [(path, label_map[cls]) for path, cls in self.metas]
        self._mc_initialized = False

    def __len__(self):
        return self.num

    def _init_memcached(self):
        if not self._mc_initialized:
            '''
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file)
            self._mc_initialized = True
            '''
            self.backend = PetrelMCBackend()

    # def __getitem__(self, index):
    #     filename = self.root + '/' + self.metas[index][0]
    #     cls = self.metas[index][1]

    #     if _has_mc:
    #         # memcached
    #         self._init_memcached()
    #         '''
    #         value = mc.pyvector()
    #         self.mclient.Get(filename, value)
    #         value_buf = mc.ConvertBuffer(value)
    #         buff = io.BytesIO(value_buf)
    #         '''
    #         buff = self.backend.get(filename)
    #         with Image.open(buff) as img:
    #             img = img.convert('RGB')
    #     else:
    #         img = Image.open(filename).convert('RGB')

    #     # transform
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     return img, cls

    def __getitem__(self, index):
        rel_path = self.metas[index][0]
        filename = os.path.join(self.root, rel_path)
        cls = self.metas[index][1]

        if not os.path.exists(filename):
            if os.path.exists(filename + ".JPEG"):
                filename = filename + ".JPEG"
            elif os.path.exists(filename + ".jpg"):
                filename = filename + ".jpg"
            elif os.path.exists(filename + ".jpeg"):
                filename = filename + ".jpeg"
            else:
                raise FileNotFoundError(f"Image file not found: {filename}")

        if _has_mc:
            self._init_memcached()
            buff = self.backend.get(filename)
            with Image.open(buff) as img:
                img = img.convert("RGB")
        else:
            img = Image.open(filename).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # one-hot label for ImageNet-1K
        cls = torch.tensor(cls, dtype=torch.long)
        cls = F.one_hot(cls, num_classes=1000).float()

        return img, cls

class TinyImageNetDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        root: path to tiny-imagenet-200
        split: 'train' or 'val'
        """
        self.root = root
        self.split = split
        self.transform = transform

        wnids_file = os.path.join(root, 'wnids.txt')
        with open(wnids_file, 'r') as f:
            self.classes = [line.strip() for line in f if line.strip()]

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        if split == 'train':
            train_dir = os.path.join(root, 'train')
            for cls_name in self.classes:
                img_dir = os.path.join(train_dir, cls_name, 'images')
                if not os.path.isdir(img_dir):
                    continue
                for fname in os.listdir(img_dir):
                    if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                        path = os.path.join(img_dir, fname)
                        target = self.class_to_idx[cls_name]
                        self.samples.append((path, target))

        elif split == 'val':
            val_dir = os.path.join(root, 'val')
            img_dir = os.path.join(val_dir, 'images')
            anno_file = os.path.join(val_dir, 'val_annotations.txt')

            img_to_cls = {}
            with open(anno_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    # format: image_name, class_id, x0, y0, x1, y1
                    img_name, cls_name = parts[0], parts[1]
                    img_to_cls[img_name] = cls_name

            for fname, cls_name in img_to_cls.items():
                path = os.path.join(img_dir, fname)
                if os.path.isfile(path):
                    target = self.class_to_idx[cls_name]
                    self.samples.append((path, target))
        else:
            raise ValueError(f"Unsupported split: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target