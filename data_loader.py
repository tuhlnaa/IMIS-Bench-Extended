import math
import os
import numpy as np
import torch
from monai import data, transforms
import itertools
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import os
import ast
from scipy import sparse
import random
from scipy.ndimage import binary_opening, binary_closing
from scipy.ndimage import label as label_structure
from scipy.ndimage import sum as sum_structure
import json
import torch.distributed as dist
from PIL import Image
from dataloaders.data_utils import (
    Resize, 
    PermuteTransform, 
    LongestSidePadding, 
    Normalization, 
    get_points_from_mask, 
    get_bboxes_from_mask
    )
import cv2


class UniversalDataset(Dataset):
    def __init__(self, args, datalist, classes_list, transform):
        self.data_dir = args.data_dir
        self.datalist = datalist
        self.test_mode = args.test_mode
        classes_list.remove('background')
        self.target_list = classes_list
        self.image_size = args.image_size
        self.mask_num = args.mask_num
        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        item_dict = self.datalist[idx]
        image_path, label_path = os.path.join(self.data_dir, item_dict['image']), os.path.join(self.data_dir,item_dict['label'])
    
        # load image, label and pseudo
        image_array = np.array(Image.open(image_path))

        gt_shape = ast.literal_eval(label_path.split('.')[-2])
        allmatrix_sp= sparse.load_npz(label_path)
        label_array = allmatrix_sp.toarray().reshape(gt_shape)

        if self.test_mode:
            item_ori = {'image': image_array, 'label': label_array}
            item = self.transform(item_ori)
            _, H, W = item['image'].shape

            point_coords, point_labels, bboxes = [], [], []

            label_ids = torch.sum(item['label'], dim=(1,2))
            label_ids = torch.nonzero(label_ids != 0, as_tuple=True)[0].tolist()
        
            if len(label_ids) == 0:
                return self.__getitem__(np.random.randint(self.__len__()))

            # assert len(label_ids) >= 1, 'Please check the test data. The test data cannot be pure background.'

            nonzero_labels = torch.zeros(len(label_ids), 1, H, W)
            nonzero_category = []
            nonzero_ori_labels = []
            for idx, region_id in enumerate(label_ids):
                nonzero_labels[idx][0] = item['label'][region_id]
                nonzero_ori_labels.append(torch.tensor(np.moveaxis(label_array[region_id], -1, 0)))

                point_and_labels = get_points_from_mask(nonzero_labels[idx], top_num=0.5)
                point_coords.append(torch.as_tensor(point_and_labels[0]))
                point_labels.append(torch.as_tensor(point_and_labels[1]))

                bboxes.append(torch.as_tensor(get_bboxes_from_mask(nonzero_labels[idx], offset=0)))

                nonzero_category.append(self.target_list[region_id])

            item['gt'] = nonzero_labels
            item['ori_gt'] = torch.stack(nonzero_ori_labels, dim=0)

            item['gt_target'] = nonzero_category
            item['gt_point_coords'] = torch.stack(point_coords)
            item['gt_point_labels'] = torch.stack(point_labels)
            item['gt_bboxes'] = torch.stack(bboxes)

            item['image_root'] = [image_path]
         
        else:
            pseudo_path = os.path.join(self.data_dir, item_dict['imask'])

            try:
                pseudo_array = np.load(pseudo_path).astype(np.float32)
            except:
                print(f'{pseudo_path} not load')
                return self.__getitem__(np.random.randint(self.__len__()))

            item_ori = {'image': image_array, 'label': label_array, 'pseudo': pseudo_array}
            item = self.transform(item_ori)
            item['pseudo'] = self.cleanse_pseudo_label(item['pseudo'])

            pseudo_ids = torch.unique(item['pseudo'])
            pseudo_ids = pseudo_ids[pseudo_ids != -1]

            if len(pseudo_ids) == 0:
                return self.__getitem__(np.random.randint(self.__len__()))

            _, H, W = item['image'].shape
            select_pseudo = torch.zeros(self.mask_num, 1, H, W)

            (
                select_pseudo, 
                point_coords_pseudo, 
                point_labels_pseudo, 
                bboxes_pseudo
                ) = self.preprocess_pseudo(item['pseudo'], pseudo_ids, select_pseudo)
            

            label_ids = torch.sum(item['label'], dim=(1,2))
            label_ids = torch.nonzero(label_ids != 0, as_tuple=True)[0].tolist()

            if len(label_ids) == 0:
                return self.__getitem__(np.random.randint(self.__len__()))


            select_labels = torch.zeros(self.mask_num, 1, H, W)
            (
                select_labels, 
                point_coords, 
                point_labels, 
                bboxes, 
                nonzero_category
                ) = self. preprocess_label(item['label'], label_ids, select_labels)


            item['gt'] = select_labels
            item['pseudo'] = select_pseudo

            item['gt_point_coords'] = point_coords
            item['gt_point_labels'] = point_labels
            item['gt_bboxes'] = bboxes
            item['gt_target'] = nonzero_category
            item['pseudo_point_coords'] = point_coords_pseudo
            item['pseudo_point_labels'] = point_labels_pseudo
            item['pseudo_bboxes'] = bboxes_pseudo

        if type(item) == list:
            assert len(item) == 1
            item = item[0]
        
        assert type(item) != list
  
        post_item = self.std_keys(item)
        return post_item
    
    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


    def preprocess_pseudo(self, pseudo_label, pseudo_ids, select_pseudo):
        point_coords, point_labels, bboxes = [], [], []
  
        pseudo_region_ids = random.sample(list(pseudo_ids), k=self.mask_num) if len(pseudo_ids) >= self.mask_num else random.choices(list(pseudo_ids), k=self.mask_num)
        for idx, region_id in enumerate(pseudo_region_ids):
            select_pseudo[idx][pseudo_label==region_id.item()] = 1

            point_and_labels = get_points_from_mask(select_pseudo[idx], top_num=0.5)
            point_coords.append(torch.as_tensor(point_and_labels[0]))
            point_labels.append(torch.as_tensor(point_and_labels[1]))

            bboxes.append(torch.as_tensor(get_bboxes_from_mask(select_pseudo[idx], offset=5)))

        point_coords = torch.stack(point_coords)
        point_labels = torch.stack(point_labels)
        bboxes = torch.stack(bboxes)

        return select_pseudo, point_coords, point_labels, bboxes

    def preprocess_label(self, gt_label, label_ids, select_labels):
        point_coords, point_labels, bboxes, categories = [], [], [], []
        label_region_ids = random.sample(list(label_ids), k=self.mask_num) if len(label_ids) >= self.mask_num else random.choices(list(label_ids), k=self.mask_num)
        
        for idx, region_id in enumerate(label_region_ids):
            select_labels[idx][0] = gt_label[region_id]

            point_and_labels = get_points_from_mask(select_labels[idx], top_num=0.5)
            point_coords.append(torch.as_tensor(point_and_labels[0]))
            point_labels.append(torch.as_tensor(point_and_labels[1]))

            bboxes.append(torch.as_tensor(get_bboxes_from_mask(select_labels[idx], offset=5)))
            categories.append(self.target_list[region_id])

        point_coords = torch.stack(point_coords)
        point_labels = torch.stack(point_labels)
        bboxes = torch.stack(bboxes)

        return select_labels, point_coords, point_labels, bboxes, categories


    def std_keys(self, post_item):
        keys_to_remain = ['image', 'gt', 'ori_gt', 'image_root',  
                          'gt_point_coords', 'gt_point_labels', 'gt_bboxes', 'gt_target', 
                          'pseudo', 'pseudo_point_coords','pseudo_point_labels', 'pseudo_bboxes']
        keys_to_remove = post_item.keys() - keys_to_remain
        for key in keys_to_remove:
            del post_item[key]
        return post_item


    def cleanse_pseudo_label(self, pseudo_seg):
        total_voxels = pseudo_seg.numel()
        threshold = total_voxels * 0.0005
        unique_values = torch.unique(pseudo_seg)

        for value in unique_values:
            voxel_count = (pseudo_seg == value).sum()
            if voxel_count < threshold:
                pseudo_seg[pseudo_seg == value] = -1

        for label in torch.unique(pseudo_seg):
            if label == -1:
                continue

            binary_mask = pseudo_seg == label
            open = binary_opening(binary_mask.squeeze())
            close = binary_closing(open)
            processed_mask = torch.tensor(close)

            labeled_mask, num_labels = label_structure(processed_mask)
            label_sizes = sum_structure(processed_mask, labeled_mask, range(num_labels + 1))
            small_labels = np.where(label_sizes < threshold)[0]
            for label_del in small_labels:
                processed_mask[labeled_mask == label_del] = False

            pseudo_seg[binary_mask] = -1
            pseudo_seg[processed_mask.unsqueeze(0)] = label

        return pseudo_seg


def test_collate_fn(batch):
    assert len(batch) == 1, 'Please set batch size to 1 when testing mode'
    gt_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    gt_prompt['point_coords'] = batch[0]['gt_point_coords']
    gt_prompt['point_labels'] = batch[0]['gt_point_labels']
    gt_prompt['bboxes'] = batch[0]['gt_bboxes']
    image_root = batch[0]['image_root']
    target_list = batch[0]['gt_target']
    return {
        'image': batch[0]['image'].unsqueeze(0),
        'label': batch[0]['gt'],
        'ori_label': batch[0]['ori_gt'],
        'gt_prompt': gt_prompt,
        'target_list': target_list,
        'image_root': image_root
    }



def train_collate_fn(batch):
    images, labels, pseudos, target_list = [], [], [], []
    gt_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    pseudo_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    
    for sample in batch:
        images.append(sample['image'])
        labels.append(sample['gt'])
        gt_prompt['point_coords'].append(sample['gt_point_coords'])
        gt_prompt['point_labels'].append(sample['gt_point_labels'])
        gt_prompt['bboxes'].append(sample['gt_bboxes'])
        target_list += sample['gt_target']

        pseudos.append(sample['pseudo'])
        pseudo_prompt['point_coords'].append(sample['pseudo_point_coords'])
        pseudo_prompt['point_labels'].append(sample['pseudo_point_labels'])
        pseudo_prompt['bboxes'].append(sample['pseudo_bboxes'])

    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    pseudos = torch.cat(pseudos, dim=0)
    
    gt_prompt = {key: torch.cat(value, dim=0) if len(value) !=0 else None for key, value in gt_prompt.items()}
    pseudo_prompt = {key: torch.cat(value, dim=0) if len(value) !=0 else None for key, value in pseudo_prompt.items()}

    return {
        'image': images,
        'label': labels,
        'pseudo': pseudos,
        'target_list': target_list,
        'gt_prompt': gt_prompt,
        'pseudo_prompt': pseudo_prompt,
    }
        


def get_loader(args):

    dataset_json = os.path.join(args.data_dir, 'dataset.json')
    dataset_dict = json.load(open(dataset_json, 'r'))
    target_size = (args.image_size, args.image_size)

    if args.test_mode:
        datalist = dataset_dict['test']
        collate_fn = test_collate_fn
        transform = transforms.Compose(
                        [
                            Resize(keys=["image", "label"], target_size=target_size), 
                            PermuteTransform(keys=["image"], dims=(2,0,1)),
                            transforms.ToTensord(keys=["image", "label"]),
                            Normalization(keys=["image"]),
                        ]
                    )

    else:
        datalist = dataset_dict['training']
        collate_fn = train_collate_fn
        transform = transforms.Compose(
                [
                    Resize(keys=["image", "label", "pseudo"], target_size=target_size),  #
                    PermuteTransform(keys=["image"], dims=(2,0,1)),
                    transforms.ToTensord(keys=["image", "label", "pseudo"]),
                    Normalization(keys=["image"]),
                    transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=0.2),
                    transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=0.2),
                ]
            )
    
    classes_list = list(dataset_dict['labels'].values())

    dataset = UniversalDataset(
        args=args, 
        datalist=datalist, 
        classes_list=classes_list,
        transform = transform
        )

    sampler = DistributedSampler(dataset) if args.dist else None

    data_loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    return data_loader

if __name__ == "__main__":
    import argparse
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)

    def set_parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_dir", type=str, default='dataset/BTCV')
        parser.add_argument('--image_size', type=int, default=256)
        parser.add_argument('--test_mode', type=bool, default=False)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--dist', dest='dist', type=bool, default=True,help='distributed training or not')
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--mask_num', type=int, default=5)
        args = parser.parse_args()
        return args

    args = set_parse()
    train_loader = get_loader(args)

    for idx, batch in enumerate(train_loader):
        image, label = batch["image"], batch["label"]
        # pseudo = batch['pseudo']
        print(batch['target_list'])
        print(image.shape, label.shape) #, pseudo.shape
        print(batch['gt_prompt']['bboxes'].shape)
        # print(batch['image_root'])
        # save_path = 'ataloaders\dataloader_test'
        # os.makedirs(save_path, exist_ok=True)

        # pre_image = image[0].permute(1,2,0).numpy()
        # pre_label = label[0:2].squeeze().numpy()
        # pre_pseudo = pseudo[0:2].squeeze().numpy()
        # gt_bboxes = batch['gt_prompt']['bboxes'].squeeze().numpy()
        # pseudo_bboxes = batch['pseudo_prompt']['bboxes'].squeeze().numpy()
        # print(pre_image.shape, pre_label.shape, pre_pseudo.shape)

        # cv2.imwrite(f'{save_path}/image_{idx}.png', pre_image)
        # result_label = np.copy(pre_image)
        # result_pseudo = np.copy(pre_image)
        # for i in range(2):
        #     label_image = cv2.convertScaleAbs(pre_label[i] * 255)
        #     pseudo_image = cv2.convertScaleAbs(pre_pseudo[i] * 255)
        #     label_color = cv2.applyColorMap(label_image, cv2.COLORMAP_JET)

        #     x_min, y_min, x_max, y_max = gt_bboxes[i]
        #     cv2.rectangle(label_color, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), thickness=2)

        #     pseudo_color = cv2.applyColorMap(pseudo_image, cv2.COLORMAP_JET)
        #     x_min, y_min, x_max, y_max = pseudo_bboxes[i]
        #     cv2.rectangle(pseudo_color, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), thickness=2)

        #     result_label = cv2.convertScaleAbs(result_label)
        #     result_pseudo = cv2.convertScaleAbs(result_pseudo)

        #     result_label = cv2.addWeighted(result_label, 0.8, label_color, 0.4, 0)
        #     result_pseudo = cv2.addWeighted(result_pseudo, 0.8, pseudo_color, 0.4, 0)

        # cv2.imwrite(f'{save_path}/label_{idx}.png', result_label)
        # cv2.imwrite(f'{save_path}/pseudo_{idx}.png', result_pseudo)