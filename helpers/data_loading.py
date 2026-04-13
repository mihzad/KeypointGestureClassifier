from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision.transforms import v2
from torchvision.transforms import functional as F
from PIL import Image
from torchvision import tv_tensors
from scripts.weighted_sampling_distributor import analyze_weaknesses_produce_weights

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable

class KeypointsDataset(Dataset):
    """
    Dataset for pre-extracted MediaPipe keypoints stored in .npz file.

    Each sample:
        X: (T=16, 42)
        y: scalar label

    Options:
        - additional keypoint per-sample transform.
    """

    def __init__(self, npz_path: Path, keypoint_transform: None | Callable = None):
        data = np.load(npz_path)

        self.X = data["X"]  # (N, T, 42)
        self.y = data["y"]  # (N,)

        self.targets = self.y
        self.transform = keypoint_transform
        # Convert once (faster than per-sample)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        video = self.X[idx]   # (T, 42)
        label = self.y[idx]
        if self.transform is not None:
            video = self.transform(video)
        return video, label

class VideoFramesFolderDataset(Dataset):
    '''
        per-frame video loader. the folder should have structure:
        folder/class/v1 (each video in its own subfolder)/[collection of frames].
        Returns list of PIL images for each video, and corresponding label.
        :parameter:
        :param root_dir: self-explanatory.
    '''
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

        # Map class names to indices
        self.classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Build list of (video_dir_path, label_index)
        self.samples = []
        for cls_name in self.classes:
            class_dir = root_dir / cls_name

            for vid_path in sorted(class_dir.iterdir()):
                if vid_path.is_dir():
                    self.samples.append((vid_path, self.class_to_idx[cls_name]))

        self.targets = np.array([s[1] for s in self.samples])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        frame_paths = sorted([f for f in video_dir.iterdir() if f.is_file()])


        frames = []

        for path in frame_paths:
            img = Image.open(path).convert('RGB')
            frames.append(img)


        return frames, label



def create_dataloaders(training_set: KeypointsDataset,
                        validation_set: KeypointsDataset,
                        batch_size, num_workers,
                        additional_scaler_statistics_file: None | Path = None,
                        additional_scaler_eps=0.25,
                        train_only=False):
    
        #======== creating balanced-batch dataloaders ========
    t_class_counts = np.bincount(training_set.targets)
    t_class_weights = 1.0 / t_class_counts #weights based on dataset imbalance

    # extra modifiers based on current situation and model`s weaknesses
    if additional_scaler_statistics_file is not None:
        t_class_modifiers = analyze_weaknesses_produce_weights(additional_scaler_statistics_file,
                                                               eps=additional_scaler_eps)
        t_class_weights *= t_class_modifiers

    t_sample_weights = [t_class_weights[t] for t in training_set.targets]
    train_sampler = WeightedRandomSampler(t_sample_weights, num_samples=len(t_sample_weights), replacement=True)
    train_loader = DataLoader(training_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

    if train_only:
        return train_loader, None
    
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader




