import sys
sys.path.insert(0, 'augraphy')

import augraphy
import torchvision.transforms as transforms
import random
import torch
import numpy as np
import logging
import cv2

from albumentations import augmentations
from PIL import Image, ImageFilter

from augmixations.blots import HandWrittenBlot
from warp_mls import WarpMLS

logger = logging.getLogger(__name__)


class Paperize(object):
    def __init__(self, process_datasets=None, p=0.5):
        self.process_datasets = process_datasets or []

        paper_phase = [
            augraphy.PaperFactory(texture_path='augraphy/paper_textures/', p=1.),
            augraphy.BrightnessTexturize(range=(0.8, 1.), deviation=0.05, p=0.5),
        ]
        post_phase = [
            augraphy.BookBinding(radius_range=(1, 10), curve_intensity_range=(0, 20), p=0.25),
            augraphy.Brightness(range=(0.5, 1.), p=0.25),
            augraphy.Gamma(range=(0.3, 1.8), p=0.25),
            augraphy.LightingGradient(p=0.25),
        ]
        self.pipeline = augraphy.AugraphyPipeline(ink_phase=[], paper_phase=paper_phase, post_phase=post_phase)
        self.p = p

    def __call__(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            return inputs

        image, dataset = inputs
        if dataset not in self.process_datasets or random.random() < self.p:
            return image

        np_image = np.array(image)
        np_image = self.mask_background(np_image)
        if np_image.shape[0] >= 30 and np_image.shape[1] >= 30:
            try:
                np_image = self.pipeline.augment(np_image)['output']
            except Exception as e:
                logger.info(e)

        image = Image.fromarray(np_image)
        return image

    @staticmethod
    def mask_background(image):
        original_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image = cv2.bitwise_not(image)

        kernel = np.ones((15, 15), np.uint8)
        image = cv2.dilate(image, kernel, iterations=2)

        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        image = gray_image & image
        original_image[np.where(image == 0)] = 0
        return original_image


class NumpyAugmentation(object):
    def __call__(self, image):
        np_image = np.array(image)
        np_image = self.forward(np_image)
        return Image.fromarray(np_image)

    def forward(self, np_image):
        raise NotImplementedError


class ResizePad(NumpyAugmentation):
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.ratio = int(width / height)

    def forward(self, img):
        h, w, _ = img.shape

        ratio = w / h
        if ratio < self.ratio:
            padding = np.zeros((h, self.ratio * h - w, 3), dtype=np.uint8)
            img = cv2.hconcat([img, padding])
        elif ratio > self.ratio:
            padding = np.zeros((w // self.ratio - h, w, 3), dtype=np.uint8)
            img = cv2.vconcat([img, padding])
        img = cv2.resize(img, (self.width, self.height))

        return img.astype(np.uint8)


class WeightedRandomChoice:

    def __init__(self, trans, weights=None):
        self.trans = trans
        if not weights:
            self.weights = [1] * len(trans)
        else:
            assert len(trans) == len(weights)
            self.weights = weights

    def __call__(self, img):
        t = random.choices(self.trans, weights=self.weights, k=1)[0]
        try:
            tfm_img = t(img)
        except Exception as e:
            logger.warning('Error during data_aug:'+str(e))
            return img

        return tfm_img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Dilation(torch.nn.Module):

    def __init__(self, kernel=3):
        super().__init__()
        self.kernel=kernel

    def forward(self, img):
        return img.filter(ImageFilter.MaxFilter(self.kernel))

    def __repr__(self):
        return self.__class__.__name__ + '(kernel={})'.format(self.kernel)


class Erosion(torch.nn.Module):

    def __init__(self, kernel=3):
        super().__init__()
        self.kernel=kernel

    def forward(self, img):
        return img.filter(ImageFilter.MinFilter(self.kernel))

    def __repr__(self):
        return self.__class__.__name__ + '(kernel={})'.format(self.kernel)


class Underline(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, img):
        img_np = np.array(img.convert('L'))
        black_pixels = np.where(img_np < 50)
        try:
            y1 = max(black_pixels[0])
            x0 = min(black_pixels[1])
            x1 = max(black_pixels[1])
        except:
            return img
        for x in range(x0, x1):
            for y in range(y1, y1-3, -1):
                try:
                    img.putpixel((x, y), (0, 0, 0))
                except:
                    continue
        return img


class KeepOriginal(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img


class ToGray(NumpyAugmentation):
    def __init__(self):
        self.transform = augmentations.transforms.ToGray(always_apply=True)

    def forward(self, image):
        augmented = self.transform(image=image)
        return augmented['image']


class Distort(NumpyAugmentation):
    def __init__(self, segment=3):
        self.segment = segment

    def forward(self, src):
        img_h, img_w = src.shape[:2]

        cut = img_w // self.segment
        thresh = cut // 3

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
        dst_pts.append([img_w - np.random.randint(thresh), np.random.randint(thresh)])
        dst_pts.append([img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
        dst_pts.append([np.random.randint(thresh), img_h - np.random.randint(thresh)])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, self.segment, 1):
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                            np.random.randint(thresh) - half_thresh])
            dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                            img_h + np.random.randint(thresh) - half_thresh])

        trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
        dst = trans.generate()

        return dst


class Stretch(NumpyAugmentation):
    def __init__(self, segment=4):
        self.segment = segment

    def forward(self, src):
        img_h, img_w = src.shape[:2]

        cut = img_w // self.segment
        thresh = cut * 4 // 5

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, 0])
        dst_pts.append([img_w, 0])
        dst_pts.append([img_w, img_h])
        dst_pts.append([0, img_h])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, self.segment, 1):
            move = np.random.randint(thresh) - half_thresh
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + move, 0])
            dst_pts.append([cut * cut_idx + move, img_h])

        trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
        dst = trans.generate()

        return dst


class Perspective(NumpyAugmentation):
    def forward(self, src):
        img_h, img_w = src.shape[:2]

        thresh = img_h // 2

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, np.random.randint(thresh)])
        dst_pts.append([img_w, np.random.randint(thresh)])
        dst_pts.append([img_w, img_h - np.random.randint(thresh)])
        dst_pts.append([0, img_h - np.random.randint(thresh)])

        trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
        dst = trans.generate()

        return dst


class Blot(NumpyAugmentation):
    def __init__(self, max_count=2):
        def get_params(count):
            return {
                'incline': (-10, 10),
                'intensivity': (0.5, 0.9),
                'transparency': (0.05, 0.3),
                'count': count,
            }

        self.blots = [HandWrittenBlot(params=get_params(count=i+1)) for i in range(max_count)]

    def forward(self, image):
        blot = self.blots[random.randint(0, len(self.blots) - 1)]
        return blot(image)


class PaperColor(NumpyAugmentation):
    def __init__(self):
        post_phase = [
            augraphy.BookBinding(radius_range=(1, 10), curve_intensity_range=(0, 20), p=0.25),
            augraphy.Brightness(range=(0.5, 1.), p=0.25),
            augraphy.Gamma(range=(0.3, 1.8), p=0.25),
            augraphy.LightingGradient(p=0.25),
        ]
        self.pipeline = augraphy.AugraphyPipeline(ink_phase=[], paper_phase=[], post_phase=post_phase)

    def forward(self, np_image):
        if np_image.shape[0] >= 30 and np_image.shape[1] >= 30:
            try:
                np_image = self.pipeline.augment(np_image)['output']
            except Exception as e:
                logger.info(e)

        return np_image


# 0: InterpolationMode.NEAREST,
# 2: InterpolationMode.BILINEAR,
# 3: InterpolationMode.BICUBIC,
# 4: InterpolationMode.BOX,
# 5: InterpolationMode.HAMMING,
# 1: InterpolationMode.LANCZOS,
def build_data_aug(size, mode, preprocess_datasets, resnet=False, resizepad=True,
                   use_additional_augs=False):
    if resnet:
        norm_tfm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        norm_tfm = transforms.Normalize(0.5, 0.5)
    if resizepad:
        resize_tfm = ResizePad(size[0], size[1])
    else:
        resize_tfm = transforms.Resize(size, interpolation=3)

    if mode == 'train':
        augmentations = [
            # transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation(degrees=(-10, 10), expand=True, fill=0),
            transforms.GaussianBlur(3),
            Dilation(3),
            Erosion(3),
            Underline(),
            KeepOriginal(),
        ]
        if use_additional_augs:
            augmentations.extend([
                Distort(),
                Stretch(),
                Perspective(),
                Blot(),
                PaperColor(),
            ])

        return transforms.Compose([
            Paperize(preprocess_datasets),
            ToGray(),
            WeightedRandomChoice(augmentations),
            resize_tfm,
            transforms.ToTensor(),
            norm_tfm
        ])
    else:
        return transforms.Compose([
            Paperize(),
            ToGray(),
            resize_tfm,
            transforms.ToTensor(),
            norm_tfm
        ])


if __name__ == '__main__':
    tfm = ResizePad()
    img = Image.open('temp.jpg')
    tfm(img).save('temp2.jpg')
