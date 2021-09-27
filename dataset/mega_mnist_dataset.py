import json
from os import path
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from torchvision import transforms


class MNIST(Dataset):
    """Load a Megapixel MNIST dataset. See make_mnist.py."""

    CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(self, dataset_dir, train=True):
        with open(path.join(dataset_dir, "parameters.json")) as f:
            self.parameters = json.load(f)

        filename = "train.npy" if train else "test.npy"
        N = self.parameters["n_train" if train else "n_test"]
        W = self.parameters["width"]
        H = self.parameters["height"]
        self.scale = self.parameters["scale"]

        self._high_shape = (H, W, 1)
        self._low_shape = (int(self.scale*H), int(self.scale*W), 1)
        self._data = np.load(path.join(dataset_dir, filename), allow_pickle=True)
        self.image_transform = transforms.Normalize([0.5], [0.5])

        self.labels = np.argmax(np.vstack(self._data[:,2]), axis=1)

        self._uclasses = np.unique(self.labels)
        self._class_indexs = {c: np.where(self.labels == c)[0] for c in self._uclasses}

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError()

        # Placeholders
        x_high = np.zeros(self._high_shape, dtype=np.float32).ravel()

        # Fill the sparse representations
        data = self._data[i]
        x_high[data[1][0]] = data[1][1]

        # Reshape to their final shape
        x_high = x_high.reshape(self._high_shape)

        x_high = torch.from_numpy(x_high)
        x_high = x_high.permute(2, 0, 1)
        x_high = self.image_transform(x_high)
        x_low = F.interpolate(x_high[None, ...], scale_factor=self.scale)[0]

        label = np.argmax(data[2])

        return x_low, x_high, label

    def strided(self, N):
        """Extract N images almost in equal proportions from each category."""

        idxs = []

        # Get class indexs
        per_class = np.floor(N/len(self._uclasses))
        left_over = N % len(self._uclasses)

        if per_class:
            for c in self._uclasses:
                replace = len(self._class_indexs[c]) <= per_class
                idxs += list(np.random.choice(self._class_indexs[c], per_class, replace=replace))

        if left_over:
            classes = np.copy(self._uclasses)
            np.random.shuffle(classes)
            for c in classes[:left_over]:
                idxs += list(np.random.choice(self._class_indexs[c], 1))

        return idxs


if __name__ == '__main__':
    mnist_dataset = MNIST('mega_mnist', train=True)
    mnist_dataset[0]