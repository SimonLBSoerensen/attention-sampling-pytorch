import glob
from collections import namedtuple
from functools import partial
import hashlib
import os
from PIL import Image
import torch
import urllib.request
from os import path
import sys
import zipfile
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


def check_file(filepath, md5sum):
    """Check a file against an md5 hash value.
    Returns
    -------
        True if the file exists and has the given md5 sum False otherwise
    """
    try:
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(partial(f.read, 4096), b""):
                md5.update(chunk)
        print(filepath, md5.hexdigest())
        return md5.hexdigest() == md5sum
    except FileNotFoundError:
        return False

def files_are_extracted(directory):
    return len(glob.glob(os.path.join(directory, "*"))) > 1

def ensure_dataset_exists(directory, tries=1, progress_file=sys.stderr):
    """Ensure that the dataset is downloaded and is correct.
    Correctness is checked only against the annotations files.
    """
    set1_url = "http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set1/Set1Part0.zip"
    set1_annotations_url = "http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set1/annotations.txt"
    set1_annotations_md5 = "abc2432cb4433e314b7db116f0a80324"
    set2_url = "http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set2/Set2Part0.zip"
    set2_annotations_url = "http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set2/annotations.txt"
    set2_annotations_md5 = "5c575d53bc06750126932e80217970f9"

    zip_1_file = path.join(directory, "Set1Part0.zip")
    if not os.path.exists(zip_1_file):
        print("Downloading Set1", file=progress_file)
        download_file(set1_url, zip_1_file,
                      progress_file=progress_file)
    else:
        print("Set1 zip ex")

    set1_extract = path.join(directory, "Set1")
    if not(files_are_extracted(set1_extract)):
        print("Extracting...", file=progress_file)
        with zipfile.ZipFile(path.join(directory, "Set1Part0.zip")) as archive:
            archive.extractall(set1_extract)


    zip_1_anno_file = path.join(directory, "Set1", "annotations.txt")
    if not os.path.exists(zip_1_anno_file):
        print("Getting annotation file", file=progress_file)
        download_file(
            set1_annotations_url,
            zip_1_anno_file,
            progress_file=progress_file
        )

    zip_2_file = path.join(directory, "Set2Part0.zip")
    if not os.path.exists(zip_2_file):
        print("Downloading Set2", file=progress_file)
        download_file(set2_url, zip_2_file,
                      progress_file=progress_file)

    set2_extract = path.join(directory, "Set2")
    if not (files_are_extracted(set2_extract)):
        print("Extracting...", file=progress_file)
        with zipfile.ZipFile(path.join(directory, "Set2Part0.zip")) as archive:
            archive.extractall(set2_extract)

    zip_2_anno_file = path.join(directory, "Set2", "annotations.txt")
    if not os.path.exists(zip_2_anno_file):
        print("Getting annotation file", file=progress_file)
        download_file(
            set2_annotations_url,
            zip_2_anno_file,
            progress_file=progress_file
        )

    integrity = (
            check_file(
                path.join(directory, "Set1", "annotations.txt"),
                set1_annotations_md5
            ) and check_file(
                        path.join(directory, "Set2", "annotations.txt"),
                        set2_annotations_md5
                    )
    )

    if integrity:
        return
    else:
        raise RuntimeError(("Cannot download dataset or dataset download "
                            "is corrupted"))


def download_file(url, destination, progress_file=sys.stderr):
    """Download a file with progress."""
    response = urllib.request.urlopen(url)
    n_bytes = response.headers.get("Content-Length")
    if n_bytes == "":
        n_bytes = 0
    else:
        n_bytes = int(n_bytes)

    message = "\rReceived {} / {}"
    cnt = 0
    with open(destination, "wb") as dst:
        while True:
            print(message.format(cnt, n_bytes), file=progress_file,
                  end="", flush=True)
            data = response.read(65535)
            if len(data) == 0:
                break
            dst.write(data)
            cnt += len(data)
    print(file=progress_file)


class Sign(namedtuple("Sign", ["visibility", "bbox", "type", "name"])):
    """A sign object. Useful for making ground truth images as well as making
    the dataset."""

    @property
    def x_min(self):
        return self.bbox[2]

    @property
    def x_max(self):
        return self.bbox[0]

    @property
    def y_min(self):
        return self.bbox[3]

    @property
    def y_max(self):
        return self.bbox[1]

    @property
    def area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def center(self):
        return [
            (self.y_max - self.y_min) / 2 + self.y_min,
            (self.x_max - self.x_min) / 2 + self.x_min
        ]

    @property
    def visibility_index(self):
        visibilities = ["VISIBLE", "BLURRED", "SIDE_ROAD", "OCCLUDED"]
        return visibilities.index(self.visibility)

    def pixels(self, scale, size):
        return zip(*(
            (i, j)
            for i in range(round(self.y_min * scale), round(self.y_max * scale) + 1)
            for j in range(round(self.x_min * scale), round(self.x_max * scale) + 1)
            if i < round(size[0] * scale) and j < round(size[1] * scale)
        ))

    def __lt__(self, other):
        if not isinstance(other, Sign):
            raise ValueError("Signs can only be compared to signs")

        if self.visibility_index != other.visibility_index:
            return self.visibility_index < other.visibility_index

        return self.area > other.area


class STS:
    """The STS class reads the annotations and creates the corresponding
    Sign objects."""

    def __init__(self, directory, train=True, seed=0):
        cwd = os.getcwd()
        directory = path.join(cwd, directory)
        ensure_dataset_exists(directory)

        self._directory = directory
        self._inner = "Set{}".format(1 + ((seed + 1 + int(train)) % 2))
        self._data = self._load_signs(self._directory, self._inner)

    def _load_files(self, directory, inner):
        files = set()
        with open(path.join(directory, inner, "annotations.txt")) as f:
            for l in f:
                files.add(l.split(":", 1)[0])
        return sorted(files)

    def _read_bbox(self, parts):
        def _float(x):
            try:
                return float(x)
            except ValueError:
                if len(x) > 0:
                    return _float(x[:-1])
                raise

        return [_float(x) for x in parts]

    def _load_signs(self, directory, inner):
        with open(path.join(directory, inner, "annotations.txt")) as f:
            lines = [l.strip() for l in f]
        keys, values = zip(*(l.split(":", 1) for l in lines))
        all_signs = []
        for v in values:
            signs = []
            for sign in v.split(";"):
                if sign == [""] or sign == "":
                    continue
                parts = [s.strip() for s in sign.split(",")]
                if parts[0] == "MISC_SIGNS":
                    continue
                signs.append(Sign(
                    visibility=parts[0],
                    bbox=self._read_bbox(parts[1:5]),
                    type=parts[5],
                    name=parts[6]
                ))
            all_signs.append(signs)
        images = [path.join(directory, inner, f) for f in keys]

        return list(zip(images, all_signs))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]


class SpeedLimits(Dataset):
    """Provide a Keras Sequence for the SpeedLimits dataset which is basically
    a filtered version of the STS dataset.
    Arguments
    ---------
        directory: str, The directory that the dataset already is or is going
                   to be downloaded in
        train: bool, Select the training or testing sets
        seed: int, The prng seed for the dataset
    """
    LIMITS = ["50_SIGN", "70_SIGN", "80_SIGN"]
    CLASSES = ["EMPTY", *LIMITS]

    def __init__(self, directory, train=True, seed=0):
        # Create list with tuples of (image path, image class)
        self._data = self._filter(STS(directory, train, seed))

        # Define data augmentation
        if train:
            self.image_transform = transforms.Compose([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                                                       transforms.RandomAffine(degrees=0,
                                                                               translate=(100 / 1280, 100 / 960)),
                                                       transforms.ToTensor()
                                                       # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                       ])
        else:
            self.image_transform = transforms.Compose([transforms.ToTensor()
                                                       # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                       ])
        # Calculate the sampling weights for the images depending on the class the image belongs to
        weights = make_weights_for_balanced_classes(self._data, len(self.CLASSES))
        weights = torch.DoubleTensor(weights)

        # Make the sampler there will sample images depending on the weights they have been given
        self.sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    def _filter(self, data):
        filtered = []
        for image, signs in data:
            # Check if there is a speed limits sign and if it is visible then is will be acceptable if there is
            # only other signs or the speed limits is not visible then not acceptable
            # empty (no signs at all) is also acceptable
            signs, acceptable = self._acceptable(signs)
            if acceptable:
                if not signs:
                    # If the image is empty the it belongs to class 0
                    filtered.append((image, 0))
                else:
                    # If there is a speed limit sign then the class is the index of the sign in self.CLASSES
                    filtered.append((image, self.CLASSES.index(signs[0].name)))
        # Return a list of (image path, image class) pers
        # where the classes are of the empty class or one of the speed limits
        return filtered

    def _acceptable(self, signs):
        # Keep it as empty
        if not signs:
            return signs, True

        # Filter just the speed limits and sort them wrt visibility
        signs = sorted(s for s in signs if s.name in self.LIMITS)

        # No speed limit but many other signs
        if not signs:
            return None, False

        # Not visible sign so skip
        if signs[0].visibility != "VISIBLE":
            return None, False

        return signs, True

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        # Get the image path and class of image
        image, category = self._data[i]

        # Load the high resolution image and perform the image transformation (data augmentation)
        x_high = Image.open(image)
        x_high = self.image_transform(x_high)

        # Down scale the high resolution image to the low resolution there is used for the attention map
        x_low = F.interpolate(x_high[None, ...], scale_factor=0.3, mode='bilinear',
                              align_corners=False, recompute_scale_factor=True)[0]

        # Return the low resolution, high resolution and class of the given index
        return x_low, x_high, category

    @property
    def image_size(self):
        return self[0][0].shape[1:]

    @property
    def class_frequencies(self):
        """Compute and return the class specific frequencies."""
        freqs = np.zeros(len(self.CLASSES), dtype=np.float32)
        for image, category in self._data:
            freqs[category] += 1
        return freqs / len(self._data)

    def strided(self, N):
        """Extract N images almost in equal proportions from each category."""
        order = np.arange(len(self._data))
        np.random.shuffle(order)
        idxs = []
        cat = 0
        while len(idxs) < N:
            for i in order:
                image, category = self._data[i]
                if cat == category:
                    idxs.append(i)
                    cat = (cat + 1) % len(self.CLASSES)
                if len(idxs) >= N:
                    break
        return idxs


def make_weights_for_balanced_classes(images, num_classes):
    count = [0] * num_classes
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * num_classes
    N = float(sum(count))
    for i in range(num_classes):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def reverse_transform(inp):
    """ Do a reverse transformation. inp should be of shape [3, H, W] """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


if __name__ == '__main__':
    speedlimit_dataset = SpeedLimits('traffic_data')

    speedlimit_dataloader = DataLoader(speedlimit_dataset, shuffle=True, batch_size=4)
