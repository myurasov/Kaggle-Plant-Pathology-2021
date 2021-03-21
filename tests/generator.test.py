import unittest

import numpy as np
from src.config import c as gc
from src.generator import Generator, default_image_augmenation_options


class Test_Generator(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32

        self.generator = Generator(
            csv_file=f"{gc['DATA_DIR']}/src/train.csv",
            images_src_dir=f"{gc['DATA_DIR']}/src/train_images",
            batch_size=self.batch_size,
            images_mean=128,
            images_std=255,
            target_image_size=(224, 224),
            image_augmentation_options=default_image_augmenation_options,
            cache_dir=gc["DATA_DIR"] + "/images_cache",
        )

    def test_len(self):
        # length in batches
        self.assertEqual(self.generator.__len__(), 582)

    def test_get_one_1(self):
        r = self.generator.get_one(
            0, use_cached=False, write_cache=False, normalize=True, augment=True
        )


if __name__ == "__main__":
    unittest.main(warnings="ignore")
