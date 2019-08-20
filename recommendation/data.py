import os
from ast import literal_eval
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras_preprocessing.image import Iterator
import imgaug as ia
from imgaug import augmenters as iaa
from keras_retinanet.models.retinanet import AnchorParameters
from keras_retinanet.preprocessing.generator import Generator
from keras_retinanet.utils.anchors import anchors_for_shape
from keras_retinanet.utils.image import preprocess_image
from torch.utils.data.dataset import Dataset

ORIG_WIDTH = 1024
ORIG_HEIGHT = 1024


class ImageDataset(Dataset):
    def __init__(self, mode: str, data_frame: pd.DataFrame, images_path: str, target_size: Tuple[int, int],
                 interpolation: str, augmenter: iaa.Augmenter = None, seed: int = None):
        self.mode = mode
        self.target_size = target_size
        self._interpolation = interpolation
        self._augmenter = augmenter

        self.x_scale = target_size[0] / ORIG_WIDTH
        self.y_scale = target_size[1] / ORIG_HEIGHT

        self.im_list = np.array([os.path.join(images_path, image) for image in data_frame["FileName"]])
        self.labels: pd.Series = data_frame["BoundingBox"].apply(literal_eval)

        self.labels: np.ndarray = self.labels.apply(self._transform_label_for_classification
                                                    if self.mode == "classification"
                                                    else self._transform_label_for_localization).values

    def _transform_label_for_classification(self, bounding_boxes: List[dict]):
        if len(bounding_boxes) > 0:
            return np.array(bounding_boxes[0]["target"])
        else:
            return np.array(0)

    def _transform_label_for_localization(self, bounding_boxes: List[dict]):
        if len(bounding_boxes) == 0 or bounding_boxes[0]["target"] == 0:
            return np.zeros((0, 5))
        else:
            boxes = np.zeros((len(bounding_boxes), 5))
            for idx, bounding_box in enumerate(bounding_boxes):
                boxes[idx, 0] = bounding_box["x"] * self.x_scale
                boxes[idx, 1] = bounding_box["y"] * self.y_scale
                boxes[idx, 2] = (bounding_box["x"] + bounding_box["w"]) * self.x_scale
                boxes[idx, 3] = (bounding_box["y"] + bounding_box["h"]) * self.y_scale
                boxes[idx, 4] = 0
            return boxes

    def _load_image(self, im: str) -> np.ndarray:
        image = img_to_array(load_img(im, target_size=self.target_size, interpolation=self._interpolation))

        if self._augmenter:
            image = self._augment_image(image)

        return image

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        image_aug = self._augmenter.augment_image(image)
        return image_aug.astype('float32')

    def get_aspect_ratio(self, index: int) -> float:
        image = Image.open(self.im_list[index])
        return float(image.width) / float(image.height)

    def get_label(self, index: int) -> np.ndarray:
        return self.labels[index]

    def load_image(self, index: int):
        return self._load_image(self.im_list[index])

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.load_image(index), self.get_label(index)

    def __len__(self):
        return self.labels.shape[0]

class ImageIterator(Iterator):
    """
    Keras Sequence object to train a model on larger-than-memory data.
    """

    def __init__(self, dataset: ImageDataset, batch_size: int, preprocess_image_mode: str, shuffle: bool = True, seed: int = None):
        super().__init__(len(dataset), batch_size, shuffle, seed)
        self._dataset = dataset
        self._preprocess_image=_preprocess_image_function(preprocess_image_mode)

    def _get_batches_of_transformed_samples(self, index_array: np.ndarray):
        batch: List[Tuple[np.ndarray, np.ndarray]] = [self._dataset[index] for index in index_array]

        batch_x = np.array([self._preprocess_image(item[0]) for item in batch])
        batch_y = np.array([item[1] for item in batch])
        return batch_x, batch_y


class LocalizationImageGenerator(Generator):
    def __init__(self, dataset: ImageDataset, batch_size: int, preprocess_image_mode: str, augmenter: iaa.Augmenter = None,
                 anchor_parameters: AnchorParameters= AnchorParameters.default, shuffle: bool = True):
        assert dataset.mode == "localization"
        self._dataset = dataset
        super().__init__(batch_size=batch_size, image_min_side=dataset.target_size[0],
                         image_max_side=dataset.target_size[0], group_method="random", shuffle_groups=shuffle,
                         preprocess_image=_preprocess_image_function(preprocess_image_mode))
        self._augmenter = augmenter
        self._anchor_parameters = anchor_parameters

    def size(self):
        return len(self._dataset)

    def __len__(self):
        return (self.size() + self.batch_size - 1) // self.batch_size

    def num_classes(self):
        return 1

    def image_aspect_ratio(self, image_index):
        return self._dataset.get_aspect_ratio(image_index)

    def label_to_name(self, label):
        return "Pneumonia"

    def load_image(self, image_index):
        return self._dataset.load_image(image_index)

    def load_annotations(self, image_index):
        return self._dataset.get_label(image_index)

    def random_transform_group_entry(self, image: np.ndarray, annotations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._augmenter is None:
            return image, annotations
        seq_det = self._augmenter.to_deterministic()

        image_aug = seq_det.augment_images([image])[0].astype('float32')

        if annotations.shape[0] > 0:
            bbs = ia.BoundingBoxesOnImage([
                ia.BoundingBox(x1=a[0], y1=a[1], x2=a[2], y2=a[3]) for a in annotations
            ], shape=image.shape)
            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

            annotations_aug = np.array([np.array([bb.x1, bb.y1, bb.x2, bb.y2, an[4]])
                                        for (bb, an) in zip(bbs_aug.bounding_boxes, annotations)])
            return image_aug, annotations_aug
        else:
            return image_aug, annotations

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        image = self.preprocess_image(image)

        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations[:, :4] *= image_scale

        return image, annotations

    def generate_anchors(self, image_shape):
        return anchors_for_shape(image_shape, shapes_callback=self.compute_shapes,
                                 ratios=self._anchor_parameters.ratios, scales=self._anchor_parameters.scales,
                                 strides=self._anchor_parameters.strides, sizes=self._anchor_parameters.sizes)

def _preprocess_image_function(mode: str):
    return dict(rescale=_rescale_preprocess_image, tf=_tf_preprocess_image, caffe=_caffe_preprocess_image,
                chexnet=_chexnet_preprocess_image)[mode]

def _rescale_preprocess_image(image: np.ndarray):
    return image / 255.

def _tf_preprocess_image(image: np.ndarray):
    return preprocess_image(image, mode="tf")

def _caffe_preprocess_image(image: np.ndarray):
    return preprocess_image(image, mode="caffe")

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def _chexnet_preprocess_image(image: np.ndarray):
    image = image / 255.
    return (image - _IMAGENET_MEAN) / _IMAGENET_STD

