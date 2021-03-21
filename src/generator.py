import os
from glob import glob
import re

import numpy as np
from PIL import Image
import pandas as pd
from tensorflow import keras

from src.config import c as gc

default_images_augmentation_params = {
    "hshear_p": 0.25,
    "hshear_range": [-0.5, 0.5],
    "vshear_p": 0.25,
    "vshear_range": [-0.5, 0.5],
    "rotate_p": 0.25,
    "rotate_angle_range": [-45, 45],
    "vflip_p": 0.25,
    "hflip_p": 0.25,
    "zoom_p": 0.25,
    "zoom_range": [0.5, 2],
    "brightness_p": 0.25,
    "brightness_range": [0.5, 2],
    "saturation_p": 0.25,
    "saturation_range": [-0.5, 2],
}


class Generator(keras.utils.Sequence):
    def __init__(
        self,
        csv_file,
        images_dir,
        batch_size=32,
        images_mean=128,
        images_std=255,
        images_target_size=(896, 896),
        cache_dir=gc["DATA_DIR"] + "/images_cache",
        images_augmentation=default_images_augmentation_params,
    ):
        self.batch_size = batch_size
        self.csv_file = csv_file
        self.images_dir = images_dir
        self.images_mean = images_mean
        self.images_std = images_std
        self.images_target_size = images_target_size
        self.cache_dir = cache_dir
        self.images_augmentation_params = images_augmentation

        # read inpuit files list
        self.src_files = glob(f"{self.images_dir}/*.jpg")

        # create label index map
        if csv_file:
            self.labels = self._read_labels()

        # create cache dir for images
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.n_samples = len(self.src_files)
        self.n_batches = self.n_samples // self.batch_size

    def _read_labels(self):
        """
        Convert labels to one-hot representation
        Returns dict mapping to 1-hot label
        """

        df = pd.read_csv(self.csv_file, index_col=0)

        # label indexes in 1-hot representation
        label_ixs = sorted(list(set(" ".join(set(df.labels)).split(" "))))
        label_ixs = dict([(x, label_ixs.index(x)) for x in label_ixs])
        self.label_ixs = label_ixs

        labels = {}  # id: 1h

        for k, v in df.to_dict()["labels"].items():
            y = np.zeros((len(label_ixs)), dtype=np.float32)
            y[list(map(lambda x: label_ixs[x], v.split(" ")))] = 1.0
            labels[k[:-4]] = y

        return labels

    def __len__(self):
        """
        Length in batches
        """
        return self.n_batches

    def __getitem__(self, b_ix):
        """
        Produce batch, by batch index
        """

        b_X = 0
        b_Y = 0
        return (b_X, b_Y)

    def get_one(self, one_ix):
        """
        Get single item by absolute index
        """

        src_file = self.src_files[one_ix]
        sample_id = re.findall("/(\\w+)\\.jpg", src_file)[0]
        cache_file = f"{self.cache_dir}/{sample_id}.npy"

        # read and cache file
        if not os.path.isfile(cache_file):
            img = Image.open(src_file)
            img = img.resize(self.images_target_size, resample=Image.BICUBIC)
            img = np.array(img).astype(np.float16)
        else:
            img = np.load(cache_file)

        # verify that cached data has the corect dimensions
        assert img.shape == self.images_target_size + (3,)

        x, y = 0, 0

        return (x, y)