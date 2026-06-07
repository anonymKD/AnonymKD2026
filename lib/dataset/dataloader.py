import torch
import numpy as np

def fast_collate(batch, memory_format=torch.contiguous_format):
    # batch element is either (img, y) or (img, y, idx)
    first = batch[0]
    has_idx = isinstance(first, (tuple, list)) and len(first) >= 3

    imgs = [b[0] for b in batch]
    targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)

    if has_idx:
        indices = torch.tensor([b[2] for b in batch], dtype=torch.int64)
    else:
        indices = None

    # imgs are numpy arrays; your original code assumes CHW
    w = imgs[0].shape[2]
    h = imgs[0].shape[1]

    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)

    for i, nump_array in enumerate(imgs):
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)

        tensor[i] += torch.from_numpy(nump_array)

    if indices is None:
        return tensor, targets
    return tensor, targets, indices

class DataPrefetcher():
    def __init__(self, loader, transforms=None, mixup_transform=None):
        self.loader = loader
        self.loader_iter = iter(loader)
        self.transforms = transforms
        self.mixup_transform = mixup_transform
        self.stream = torch.cuda.Stream()

        # will be set in preload()
        self.next_idx = None

    def preload(self):
        try:
            batch = next(self.loader_iter)  # can be (input,target) or (input,target,idx)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_idx = None
            return

        if isinstance(batch, (tuple, list)) and len(batch) >= 3:
            self.next_input, self.next_target, self.next_idx = batch[0], batch[1], batch[2]
        else:
            self.next_input, self.next_target = batch
            self.next_idx = None

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

            if self.transforms is not None:
                self.next_input = self.transforms(self.next_input.float())

            if self.mixup_transform is not None:
                # Mixup changes targets; idx stays aligned with the mixed samples.
                self.next_input, self.next_target = self.mixup_transform(self.next_input, self.next_target)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        idx = self.next_idx

        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())

        self.preload()

        # return 2-tuple or 3-tuple depending on whether idx exists
        if idx is None:
            return input, target
        return input, target, idx

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self.preload()
        return self

    def __next__(self):
        batch = self.next()
        # batch can be (input,target) or (input,target,idx)
        input = batch[0]
        if input is None:
            raise StopIteration
        return batch

    def __len__(self):
        return len(self.loader)
    
def fast_collate_bk(batch, memory_format=torch.contiguous_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].shape[2]
    h = imgs[0].shape[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, nump_array in enumerate(imgs):
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        #nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets

def fast_collate_timeseries(batch, memory_format=torch.contiguous_format):
    # Extract time series and labels
    series = [item[0] for item in batch]  # List of NumPy arrays
    targets = torch.tensor([item[1].argmax(dim=-1) for item in batch], dtype=torch.int64)  # Convert labels to tensor(remove one-hot encoding)

    # Get time series length (assume all have the same shape: num_channels * T)
    T = series[0].shape[-1]  
    if series[0].ndim == 1:
        C=1
    else:
        C=series[0].shape[0]

    # Create an empty tensor for batch storage (batch_size, C, T)
    tensor = torch.zeros((len(series), C, T), dtype=torch.float32).contiguous(memory_format=memory_format)

    # Fill the tensor with time series data
    for i, ts in enumerate(series):
        ts = np.expand_dims(ts, axis=0) if ts.ndim == 1 else ts  # Ensure shape (1, T) when no channel data
        tensor[i] += ts

    return tensor, targets

def fast_collate_timeseries_swappedAxes(batch, memory_format=torch.contiguous_format):
    # Extract time series and labels
    series = [item[0] for item in batch]  # List of NumPy arrays
    targets = torch.tensor([item[1].argmax(dim=-1) for item in batch], dtype=torch.int64)  # Convert labels to tensor(remove one-hot encoding)

    # Get time series length (assume all have the same shape: T * num_channels)
    T = series[0].shape[0]  
    if series[0].ndim == 1:
        C=1
    else:
        C=series[0].shape[-1]

    # Create an empty tensor for batch storage (batch_size, T, C)
    tensor = torch.zeros((len(series), T, C), dtype=torch.float32).contiguous(memory_format=memory_format)

    # Fill the tensor with time series data
    for i, ts in enumerate(series):
        ts = np.expand_dims(ts, axis=-1) if ts.ndim == 1 else ts  # Ensure shape ( T, C) when no channel data
        tensor[i] += ts

    return tensor, targets


class DataPrefetcher_bk():
    def __init__(self, loader, transforms=None, mixup_transform=None):
        self.loader = loader
        self.loader_iter = iter(loader)
        self.transforms = transforms
        self.mixup_transform = mixup_transform
        self.stream = torch.cuda.Stream()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader_iter)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            if self.transforms is not None:
                self.next_input = self.transforms(self.next_input.float())
            if self.mixup_transform is not None:
                self.next_input, self.next_target = \
                    self.mixup_transform(self.next_input, self.next_target)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def __iter__(self):
        self.loader_iter = iter(self.loader)   # re-generate an iter for each epoch
        self.preload()
        return self

    def __next__(self):
        input, target = self.next()
        if input is None:
            raise StopIteration
        return input, target

    def __len__(self):
        return len(self.loader)


