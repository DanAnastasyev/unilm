import glob
import logging
import os
import random

import torch
from collections import Counter
from fairseq.data import FairseqDataset, data_utils
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


_LANG_TO_ID = {
    'en': torch.tensor([250001]),
    'ru': torch.tensor([250002]),
}


def default_collater(target_dict, samples, dataset=None):
    if not samples:
        return None
    if any([sample is None for sample in samples]):
        if not dataset:
            return None
        len_batch = len(samples)
        while True:
            samples.append(dataset[random.choice(range(len(dataset)))])
            samples =list(filter (lambda x:x is not None, samples))
            if len(samples) == len_batch:
                break
    indices = []

    imgs = [] # bs, c, h , w
    target_samples = []
    target_ntokens = 0

    for sample in samples:
        index = sample['id']
        indices.append(index)

        imgs.append(sample['tfm_img'])

        target_samples.append(sample['label_ids'].long())
        target_ntokens += len(sample['label_ids'])

    num_sentences = len(samples)

    target_batch = data_utils.collate_tokens(target_samples,
                                             pad_idx=target_dict.pad(),
                                             eos_idx=target_dict.eos(),
                                             move_eos_to_beginning=False)
    rotate_batch = data_utils.collate_tokens(target_samples,
                                             pad_idx=target_dict.pad(),
                                             eos_idx=target_dict.bos(),
                                             move_eos_to_beginning=True)

    indices = torch.tensor(indices, dtype=torch.long)
    imgs = torch.stack(imgs, dim=0)

    return {
        'id': indices,
        'net_input': {
            'imgs': imgs,
            'prev_output_tokens': rotate_batch
        },
        'ntokens': target_ntokens,
        'nsentences': num_sentences,
        'target': target_batch
    }

def read_txt_and_tokenize(txt_path: str, bpe, target_dict):
    annotations = []
    with open(txt_path, 'r', encoding='utf8') as fp:
        for line in fp.readlines():
            line = line.rstrip()
            if not line:
                continue
            line_split = line.split(',', maxsplit=8)
            quadrangle = list(map(int, line_split[:8]))
            content = line_split[-1]

            if bpe:
                encoded_str = bpe.encode(content)
            else:
                encoded_str = content

            xs = [quadrangle[i] for i in range(0, 8, 2)]
            ys = [quadrangle[i] for i in range(1, 8, 2)]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            annotations.append({'bbox': bbox, 'encoded_str': encoded_str, 'category_id': 0, 'segmentation': [quadrangle]})  # 0 for text, 1 for background

    return annotations

def SROIETask2(root_dir: str, bpe, target_dict, crop_img_output_dir=None):
    data = []
    img_id = -1

    crop_data = []
    crop_img_id = -1

    image_paths = natsorted(list(glob.glob(os.path.join(root_dir, '*.jpg'))))
    for jpg_path in tqdm(image_paths):
        im = Image.open(jpg_path).convert('RGB')

        img_w, img_h = im.size
        img_id += 1

        txt_path = jpg_path.replace('.jpg', '.txt')
        annotations = read_txt_and_tokenize(txt_path, bpe, target_dict)
        img_dict = {'file_name': jpg_path, 'width': img_w, 'height': img_h, 'image_id':img_id, 'annotations':annotations}
        data.append(img_dict)

        for ann in annotations:
            crop_w = ann['bbox'][2] - ann['bbox'][0]
            crop_h = ann['bbox'][3] - ann['bbox'][1]

            if not (crop_w > 0 and crop_h > 0):
                logger.warning('Error occurs during image cropping: {} has a zero area bbox.'.format(os.path.basename(jpg_path)))
                continue
            crop_img_id += 1
            crop_im = im.crop(ann['bbox'])
            if crop_img_output_dir:
                crop_im.save(os.path.join(crop_img_output_dir, '{:d}.jpg'.format(crop_img_id)))
            crop_img_dict = {'img':crop_im, 'file_name': jpg_path, 'width': crop_w, 'height': crop_h, 'image_id':crop_img_id, 'encoded_str':ann['encoded_str']}
            crop_data.append(crop_img_dict)

    return data, crop_data

class SROIETextRecognitionDataset(FairseqDataset):
    def __init__(self, root_dir, tfm, bpe_parser, target_dict, crop_img_output_dir=None):
        self.root_dir = root_dir
        self.tfm = tfm
        self.target_dict = target_dict
        # self.bpe_parser = bpe_parser
        self.ori_data, self.data = SROIETask2(root_dir, bpe_parser, target_dict, crop_img_output_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dict = self.data[idx]

        image = img_dict['img']
        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)

        tfm_img = self.tfm(image)   # h, w, c
        return {'id': idx, 'tfm_img': tfm_img, 'label_ids': input_ids}

    def size(self, idx):
        img_dict = self.data[idx]

        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)
        return len(input_ids)

    def num_tokens(self, idx):
        return self.size(idx)


    def collater(self, samples):
        return default_collater(self.target_dict, samples)

def STR(gt_path, bpe_parser, langs, datasets):
    root_dir = os.path.dirname(gt_path)
    data = []
    img_id = 0
    with open(gt_path, 'r') as fp:
        for line in tqdm(list(fp.readlines()), desc='Loading STR:'):
            line = line.rstrip()
            fields = line.split('\t')
            if len(fields) == 2:
                img_file, text, lang, dataset = fields[0], fields[1], 'en', None
            elif len(fields) == 3:
                img_file, text, lang, dataset = fields[0], fields[1], fields[2], None
            elif len(fields) == 4:
                img_file, text, lang, dataset = fields

            img_path = os.path.join(root_dir, 'image', img_file)
            if not bpe_parser:
                encoded_str = text
            else:
                encoded_str = bpe_parser.encode(text)

            if langs is not None and lang not in langs:
                continue

            if datasets is not None and dataset not in datasets:
                continue

            data.append({
                'img_path': img_path,
                'image_id': img_id,
                'text': text,
                'encoded_str': encoded_str,
                'lang': lang,
                'dataset': dataset or lang,
            })
            img_id += 1

    return data


class SyntheticTextRecognitionDataset(FairseqDataset):
    def __init__(self, gt_path, tfm, bpe_parser, target_dict, langs, datasets):
        self.gt_path = gt_path
        self.tfm = tfm
        self.target_dict = target_dict
        self.data = STR(gt_path, bpe_parser, langs, datasets)

        datasets_counter = Counter()
        for sample in self.data:
            datasets_counter[(sample['lang'], sample['dataset'])] += 1

        dataset_stat = '\n'.join(f'- lang = {lang}, dataset = {dataset}: {count}'
                                 for (lang, dataset), count in datasets_counter.most_common())
        logger.info('Dataset stat: %s\n%s', gt_path, dataset_stat)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dict = self.data[idx]

        image = Image.open(img_dict['img_path']).convert('RGB')
        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)
        input_ids = torch.cat((_LANG_TO_ID[img_dict['lang']], input_ids))

        tfm_img = self.tfm((image, img_dict['dataset']))  # h, w, c
        return {'id': idx, 'tfm_img': tfm_img, 'label_ids': input_ids}

    def size(self, idx):
        img_dict = self.data[idx]

        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)
        return len(input_ids)

    def num_tokens(self, idx):
        return self.size(idx)

    def collater(self, samples):
        return default_collater(self.target_dict, samples)
