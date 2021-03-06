#!/usr/bin/python

import argparse
from multiprocessing import Pool, cpu_count
from pprint import pformat

import numpy as np
from tqdm import tqdm

from src.config import c as gc
from src.generator import Generator

# region: read arguments
parser = argparse.ArgumentParser(
    description="Precache decoded and resized images",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--set",
    type=str,
    default="train",
    help="Set name",
)

parser.add_argument(
    "--size",
    type=int,
    default=[224, 224],
    nargs="+",
    help="Target image size (WxH)",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

g = Generator(
    csv_file=f"{gc['DATA_DIR']}/src/{args.set}.csv",
    images_src_dir=f"{gc['DATA_DIR']}/src/{args.set}_images",
    target_image_size=tuple(args.size),
    image_augmentation_options=None,
    cache_dir=gc["DATA_DIR"] + "/images_cache",
)


def _mapping(x):
    x, y = g.get_one(
        x, use_cached=False, write_cache=True, normalize=False, augment=False
    )


with Pool(cpu_count()) as pool:
    list(
        tqdm(
            pool.imap(
                _mapping,
                range(g.n_samples),
            ),
            total=g.n_samples,
        )
    )
