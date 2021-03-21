import os
import random
import shutil
from collections import namedtuple

import numpy as np


def create_dir(dir, remove=True):
    if remove:
        shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir, exist_ok=True)


def rename_dir(dir1, dir2, remove_dir2=True):
    if remove_dir2:
        shutil.rmtree(dir2, ignore_errors=True)
    os.rename(dir1, dir2)


def dict_to_struct(d):
    return namedtuple("Struct", d.keys())(*d.values())


def fix_random_seed(seed=123):
    random.seed(123)
    np.random.seed(123)


def list_indexes(list, cols=None):
    """
    Creates a dictionary mapping values to indexes
    """
    if cols is None:
        cols = list
    return dict([(x, list.index(x)) for x in cols])
