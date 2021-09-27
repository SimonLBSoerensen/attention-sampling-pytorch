import glob
import os
from PIL import Image
import torch
import sys
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

def make_weights_for_balanced_classes(classes, num_classes):
    count = [0] * num_classes
    for c in classes:
        count[c] += 1
    weight_per_class = [0.] * num_classes

    N = float(sum(count))
    for i in range(num_classes):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(classes)

    for idx, c in enumerate(classes):
        weight[idx] = weight_per_class[c]
    return weight


class SpeedLimits(Dataset):
    """Provide a torch.utils.data.Dataset for the XX dataset.

    Arguments
    ---------
        directory: str, The directory of the dataset
        split: str, What dataset split to use
    """

    def __init__(self, directory, split, scale_factor=0.3):
        # Define data augmentation
        if split == "training":
            self.image_transform = transforms.Compose([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                                                       transforms.RandomAffine(degrees=0,
                                                                               translate=(100 / 1280, 100 / 960)),
                                                       transforms.ToTensor()
                                                       ])
        elif split == "validation":
            self.image_transform = transforms.Compose([transforms.ToTensor()
                                                       ])
        else:
            self.image_transform = transforms.Compose([transforms.ToTensor()
                                                       ])

        # Get list of image paths and classes
        self._image_paths, self._image_classes = self._get_image_paths_and_classes(directory, split)
        self._uclasses = np.unique(self._image_classes)

        # Calculate the sampling weights for the images depending on the class the image belongs to
        weights = make_weights_for_balanced_classes(self._image_classes, len(self._uclasses))
        weights = torch.DoubleTensor(weights)

        # Make the sampler there will sample images depending on the weights they have been given
        self.sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        self.x_low_size, self.x_high_size = None, None
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, i):
        # Get the image path and class of image
        image, category = self._image_paths[i], self._image_classes[i]

        # Load the high resolution image
        x_high = Image.open(image)
        # Apply data augmentation
        x_high = self.image_transform(x_high)

        # Down scale the high resolution image to the low resolution there is used for the attention map
        x_low = F.interpolate(x_high[None, ...], scale_factor=self.scale_factor, mode='bilinear',
                              align_corners=False, recompute_scale_factor=True)[0]

        # Return the low resolution, high resolution and class of the given index
        return x_low, x_high, category

    def _get_image_paths_and_classes(self, directory, split):
        # Get list of image paths and classes
        image_paths = None
        image_classes = None
        return image_paths, image_classes

    @property
    def image_size(self):
        if self.x_low_size is None or self.x_high_size is None:
            x_low, x_high, _ = self.__getitem__(0)
            self.x_low_size = x_low.shape[1:]
            self.x_high_size = x_high.shape[1:]
        return self.x_low_size, self.x_high_size

    @property
    def number_of_classes(self):
        return len(self._uclasses)

    @property
    def unique_classes(self):
        return self._uclasses

    @property
    def class_frequencies(self):
        """Compute and return the class specific frequencies."""
        freqs = np.zeros(len(self._uclasses), dtype=np.float32)
        for category in self._image_classes:
            freqs[category] += 1
        return freqs / len(self._uclasses)

    def _get_indexs_for_class(self, c):
        return np.where(self._image_classes == c)[0]

    def strided(self, N):
        """Extract N images almost in equal proportions from each category."""
        idxs = []

        per_class = np.floor(N / len(self._uclasses))
        leftover = N % len(self._uclasses)

        if per_class:
            for c in self._uclasses:
                class_indexs = self._get_indexs_for_class(c)
                replace = len(class_indexs) >= per_class
                idxs += list(np.random.choice(class_indexs, per_class, replace=replace))

        if leftover:
            classes = np.copy(self._uclasses)
            np.random.shuffle(classes)
            for c in classes[:leftover]:
                class_indexs = self._get_indexs_for_class(c)
                idxs += list(np.random.choice(class_indexs, 1))

        return idxs



