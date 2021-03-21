import unittest

import numpy as np
from src.config import c as gc
from src.generator import Generator, default_images_augmentation_params


class Test_Generator(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32

        self.generator = Generator(
            csv_file=f"{gc['DATA_DIR']}/src/train.csv",
            images_dir=f"{gc['DATA_DIR']}/src/train_images",
            batch_size=self.batch_size,
            images_mean=128,
            images_std=255,
            images_target_size=(224, 225),
            images_augmentation=default_images_augmentation_params,
            cache_dir=gc["DATA_DIR"] + "/images_cache",
        )

    def test_len(self):
        # length in batches
        self.assertEqual(self.generator.__len__(), 582)

    def test_get_one_1(self):
        r = self.generator.get_one(0)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
