import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import (
    random_brightness,
    random_rotation,
    random_shear,
    random_shift,
    random_zoom,
)

from src.config import c as gc
from utils import list_indexes

default_image_augmenation_options = {
    "rotation_max_degrees": 45,
    "zoom_range": (0.75, 1.25),
    "shift_max_fraction": {"w": 0.25, "h": 0.25},
    "shear_max_degrees": 45,
    "brightness_range": (0.5, 1.5),
}


class Generator(keras.utils.Sequence):
    def __init__(
        self,
        csv_file,
        images_src_dir,
        batch_size=32,
        target_image_size=(224, 224),
        cache_dir=gc["DATA_DIR"] + "/images_cache",
        image_augmentation_options=default_image_augmenation_options,
        shuffle=False,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.csv_file = csv_file
        self.images_dir = images_src_dir
        self.target_image_size = target_image_size
        self.cache_dir = cache_dir
        self.image_augmentation_options = image_augmentation_options

        # create label index map
        if csv_file:
            self.labels = self._read_labels()
            self.ids = list(self.labels.keys())  # list of ids

        # create cache dir for images
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        # shuffle data, also repeated after each epoch if needed
        if self.shuffle:
            np.random.shuffle(self.ids)

        self.n_samples = len(self.ids)
        self.n_batches = self.n_samples // self.batch_size

    def _read_labels(self):
        """
        Convert labels to one-hot representation
        Returns dict mapping to 1-hot label
        """

        df = pd.read_csv(self.csv_file, index_col=0)

        # label indexes in 1-hot representation
        label_ixs = sorted(list(set(" ".join(set(df.labels)).split(" "))))
        label_ixs = list_indexes(label_ixs)
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

    def get_one(
        self, one_ix, use_cached=True, write_cache=True, normalize=True, augment=True
    ):
        """
        Get single item by absolute index
        """

        id = self.ids[one_ix]
        src_file = f"{self.images_dir}/{id}.jpg"

        sample_id = re.findall("/(\\w+)\\.jpg", src_file)[0]
        cache_file = (
            f"{self.cache_dir}/{sample_id}_"
            + f"{self.target_image_size[0]}x"
            + f"{self.target_image_size[1]}.npy"
        )

        # read and cache file
        if not use_cached or not os.path.isfile(cache_file):
            x = Image.open(src_file)
            x = x.resize(self.target_image_size, resample=Image.BICUBIC)
            x = np.array(x)
            if write_cache:
                np.save(cache_file[:-4], x)
        else:
            x = np.load(cache_file)

        # verify that cached data has the corect dimensions
        # np array has HxWXC layout, unlike PIL Image's WxHxC
        assert x.shape == (self.target_image_size[1], self.target_image_size[0], 3)

        # if no csv file is provided, return no label
        y = self.labels[sample_id] if self.csv_file else None

        # augment
        if augment and self.image_augmentation_options is not None:
            x = self._augment_image(x)

        # normalize (sample-wise)
        if normalize:
            x = x.astype(np.float64)
            x = x - np.mean(x, axis=(0, 1))
            x = x / np.std(x, axis=(0, 1))

        return x.astype(np.float16), y

    def _augment_image(self, x):
        """
        Randomply augment image
        """

        assert x.dtype == np.uint8

        # common options
        co = {
            "row_axis": 0,
            "col_axis": 1,
            "channel_axis": 2,
            # can be 'constant', 'nearest', 'reflect', 'wrap'
            "fill_mode": "nearest",
            "cval": 0.0,
        }

        # default_image_augmenation_options = {
        #     "rotation_max_degrees": 45,
        #     "zoom_range": (0.75, 1.25),
        #     "shift_max_fraction": {"w": 0.25, "h": 0.25},
        #     "shear_max_degrees": 45,
        #     "brightness_range": (0.5, 1.5),
        # }

        o = self.image_augmentation_options

        x = random_rotation(x, o["rotation_max_degrees"], **co)
        x = random_shear(x, o["shear_max_degrees"], **co)
        x = random_shift(
            x, o["shift_max_fraction"]["w"], o["shift_max_fraction"]["h"], **co
        )
        x = random_zoom(x, o["zoom_range"], **co)
        x = random_brightness(x, o["brightness_range"])

        return x

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.ids)
