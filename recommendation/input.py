from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from imgaug import augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator

from rsna.data import ImageDataset

SEED = 42

PIL_INTERPOLATION_METHODS = dict(
    nearest=Image.NEAREST,
    bilinear=Image.BILINEAR,
    bicubic=Image.BICUBIC,
    hamming=Image.HAMMING,
    box=Image.BOX,
    lanczos=Image.LANCZOS,
)


def create_dataset(mode: str, data_frame: pd.DataFrame, images_path: str, target_size: Tuple[int, int],
                   interpolation: str, augmenter: iaa.Augmenter = None,
                   seed: int = SEED) -> ImageDataset:
    return ImageDataset(mode=mode, data_frame=data_frame, images_path=images_path, target_size=target_size,
                        interpolation=interpolation, augmenter=augmenter, seed=seed)
