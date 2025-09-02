import os
import ast
import json
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist

from omegaconf import OmegaConf
from monai import data, transforms
from scipy import sparse
from scipy import ndimage
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Tuple, Any
from pathlib import Path
from PIL import Image

from src.data.data_utils import (
    cleanse_pseudo_label,
    get_points_from_mask, 
    get_bboxes_from_mask
)
from src.utils.logging_utils import LoggingManager


class UniversalDataset(Dataset):
    """Universal dataset for medical image segmentation with interactive prompts.
    
    Supports both training and testing modes with different data processing pipelines.
    """
    def __init__(
        self, 
        args, 
        datalist: List[Dict], 
        classes_list: List[str], 
        transform
    ):
        """Initialize the dataset.
        
        Args:
            args: Configuration arguments containing data_dir, test_mode, image_size, mask_num
            datalist: List of data items with image/label paths
            classes_list: List of class names including 'background'
            transform: Transform pipeline for data augmentation
        """
        self.data_dir = Path(args.data_dir)
        self.datalist = datalist
        self.test_mode = args.model.test_mode
        self.image_size = args.model.image_size
        self.mask_num = args.dataset.mask_num
        self.transform = transform
        
        # Remove background from target list
        self.target_list = [c for c in classes_list if c != 'background']


    def __len__(self) -> int:
        return len(self.datalist)


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing processed image, labels, and prompts
        """
        item_dict = self.datalist[idx]
        
        # Load base data
        try:
            image_array, label_array = self._load_image_and_label(item_dict)
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return self._get_random_sample()
        
        if self.test_mode:
            return self._process_test_sample(image_array, label_array, item_dict)
        else:
            return self._process_train_sample(image_array, label_array, item_dict)


    def _load_image_and_label(self, item_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and label arrays from disk."""
        image_path = self.data_dir / item_dict['image']
        label_path = self.data_dir / item_dict['label']
        
        # Load image
        image_array = np.array(Image.open(image_path))
        image_array = np.transpose(image_array, (2, 0, 1))
        
        # Load sparse label
        gt_shape = ast.literal_eval(str(label_path).split('.')[-2])
        allmatrix_sp = sparse.load_npz(label_path)
        label_array = allmatrix_sp.toarray().reshape(gt_shape)
        label_array = np.squeeze(label_array, axis=-1)

        return image_array, label_array


    def _process_test_sample(
        self, 
        image_array: np.ndarray, 
        label_array: np.ndarray, 
        item_dict: Dict
    ) -> Dict[str, Any]:
        """Process a sample for testing mode.
        
        Args:
            image_array: Input image array
            label_array: Ground truth label array
            item_dict: Original data dictionary
            
        Returns:
            Processed sample dictionary
        """
        # Apply transforms
        item_ori = {'image': image_array, 'label': label_array}
        data = self.transform(item_ori)
        _, H, W = data['image'].shape
        
        # Find non-empty labels
        label_ids = self._get_valid_label_ids(data['label'])
        if len(label_ids) == 0:
            return self._get_random_sample()
        
        # Process each valid label
        processed_data = self._extract_test_annotations(
            data['label'], 
            label_array, 
            label_ids, 
            H, 
            W
        )
        
        # Update item with processed data
        data.update(processed_data)
        data['image_root'] = [str(self.data_dir / item_dict['image'])]
        
        return self._standardize_keys(data)


    def _process_train_sample(
        self, 
        image_array: np.ndarray, 
        label_array: np.ndarray, 
        item_dict: Dict
    ) -> Dict[str, Any]:
        """Process a sample for training mode.
        
        Args:
            image_array: Input image array
            label_array: Ground truth label array
            item_dict: Original data dictionary
            
        Returns:
            Processed sample dictionary
        """
        # Load pseudo labels
        pseudo_path = self.data_dir / item_dict['imask']
        try:
            pseudo_array = np.load(pseudo_path).astype(np.float32)
            pseudo_array = np.transpose(pseudo_array, (2, 0, 1))
        except Exception as e:
            print(f'Failed to load {pseudo_path}: {e}')
            return self._get_random_sample()
        
        # Apply transforms
        item_ori = {
            'image': image_array, 
            'label': label_array, 
            'pseudo': pseudo_array
        }
        data = self.transform(item_ori)
        
        # Process pseudo labels
        data['pseudo'] = cleanse_pseudo_label(data['pseudo'])
        pseudo_ids = self._get_valid_pseudo_ids(data['pseudo'])
        if len(pseudo_ids) == 0:
            return self._get_random_sample()
        
        # Process ground truth labels
        label_ids = self._get_valid_label_ids(data['label'])
        if len(label_ids) == 0:
            return self._get_random_sample()
        
        _, H, W = data['image'].shape
        
        # Process pseudo and ground truth masks
        pseudo_data = self._process_pseudo_masks(data['pseudo'], pseudo_ids, H, W)
        gt_data = self._process_gt_masks(data['label'], label_ids, H, W)
        
        # Update item
        data.update({
            'gt': gt_data['masks'],
            'pseudo': pseudo_data['masks'],
            'gt_point_coords': gt_data['point_coords'],
            'gt_point_labels': gt_data['point_labels'],
            'gt_bboxes': gt_data['bboxes'],
            'gt_target': gt_data['categories'],
            'pseudo_point_coords': pseudo_data['point_coords'],
            'pseudo_point_labels': pseudo_data['point_labels'],
            'pseudo_bboxes': pseudo_data['bboxes']
        })
        
        return self._standardize_keys(data)


    def _get_valid_label_ids(self, label: torch.Tensor) -> List[int]:
        """Get indices of non-empty labels.
        
        Args:
            label: Label tensor of shape (num_classes, H, W)
            
        Returns:
            List of valid label indices
        """
        label_sums = torch.sum(label, dim=(1, 2))
        return torch.nonzero(label_sums != 0, as_tuple=True)[0].tolist()


    def _get_valid_pseudo_ids(self, pseudo: torch.Tensor) -> torch.Tensor:
        """Get unique pseudo label IDs (excluding -1).
        
        Args:
            pseudo: Pseudo label tensor
            
        Returns:
            Tensor of valid pseudo IDs
        """
        pseudo_ids = torch.unique(pseudo)
        return pseudo_ids[pseudo_ids != -1]


    def _extract_test_annotations(
        self, 
        label: torch.Tensor, 
        label_array: np.ndarray,
        label_ids: List[int], 
        H: int, 
        W: int
    ) -> Dict[str, Any]:
        """Extract annotations for test samples.
        
        Args:
            label: Transformed label tensor
            label_array: Original label array
            label_ids: List of valid label indices
            H, W: Height and width of the image
            
        Returns:
            Dictionary with extracted annotations
        """
        num_labels = len(label_ids)
        nonzero_labels = torch.zeros(num_labels, 1, H, W)
        nonzero_ori_labels = []
        nonzero_category = []
        point_coords = []
        point_labels = []
        bboxes = []

        for idx, region_id in enumerate(label_ids):
            # Extract mask
            nonzero_labels[idx][0] = label[region_id]
            nonzero_ori_labels.append(
                torch.tensor(np.expand_dims(label_array[region_id], 0))
            )

            # Extract prompts
            coords, labels = get_points_from_mask(nonzero_labels[idx])
            point_coords.append(coords)
            point_labels.append(labels)

            box = get_bboxes_from_mask(nonzero_labels[idx], offset=0)
            bboxes.append(box)
            
            nonzero_category.append(self.target_list[region_id])
        
        return {
            'gt': nonzero_labels,
            'ori_gt': torch.stack(nonzero_ori_labels, dim=0),
            'gt_target': nonzero_category,
            'gt_point_coords': torch.stack(point_coords),
            'gt_point_labels': torch.stack(point_labels),
            'gt_bboxes': torch.stack(bboxes)
        }


    def _process_pseudo_masks(
        self, 
        pseudo_label: torch.Tensor, 
        pseudo_ids: torch.Tensor, 
        H: int, 
        W: int
    ) -> Dict[str, torch.Tensor]:
        """Process pseudo label masks and extract prompts.
        
        Args:
            pseudo_label: Pseudo label tensor
            pseudo_ids: Valid pseudo IDs
            H, W: Height and width
            
        Returns:
            Dictionary with masks and prompts
        """
        select_pseudo = torch.zeros(self.mask_num, 1, H, W)
        point_coords = []
        point_labels = []
        bboxes = []
        
        # Sample pseudo regions
        pseudo_region_ids = self._sample_regions(pseudo_ids, self.mask_num)
        
        for idx, region_id in enumerate(pseudo_region_ids):
            # Create mask
            select_pseudo[idx][pseudo_label == region_id.item()] = 1
            
            # Extract prompts
            coords, labels = get_points_from_mask(select_pseudo[idx])
            point_coords.append(coords)
            point_labels.append(labels)
            bboxes.append(
                torch.as_tensor(
                    get_bboxes_from_mask(select_pseudo[idx], offset=5)
                )
            )
        
        return {
            'masks': select_pseudo,
            'point_coords': torch.stack(point_coords),
            'point_labels': torch.stack(point_labels),
            'bboxes': torch.stack(bboxes)
        }


    def _process_gt_masks(
        self, 
        gt_label: torch.Tensor, 
        label_ids: List[int], 
        H: int, 
        W: int
    ) -> Dict[str, Any]:
        """Process ground truth masks and extract prompts.
        
        Args:
            gt_label: Ground truth label tensor
            label_ids: Valid label indices
            H, W: Height and width
            
        Returns:
            Dictionary with masks, prompts, and categories
        """
        select_labels = torch.zeros(self.mask_num, 1, H, W)
        point_coords = []
        point_labels = []
        bboxes = []
        categories = []
        
        # Sample label regions
        label_region_ids = self._sample_regions(label_ids, self.mask_num)
        
        for idx, region_id in enumerate(label_region_ids):
            # Create mask
            select_labels[idx][0] = gt_label[region_id]
            
            # Extract prompts
            coords, labels = get_points_from_mask(select_labels[idx])
            point_coords.append(coords)
            point_labels.append(labels)

            box = get_bboxes_from_mask(select_labels[idx], offset=5)
            bboxes.append(box)
            categories.append(self.target_list[region_id])
        
        return {
            'masks': select_labels,
            'point_coords': torch.stack(point_coords),
            'point_labels': torch.stack(point_labels),
            'bboxes': torch.stack(bboxes),
            'categories': categories
        }


    def _sample_regions(
        self, 
        region_ids: List, 
        k: int
    ) -> List:
        """Sample k regions from available IDs.
        
        Args:
            region_ids: Available region IDs
            k: Number of regions to sample
            
        Returns:
            List of sampled region IDs
        """
        if len(region_ids) >= k:
            return random.sample(list(region_ids), k=k)
        return random.choices(list(region_ids), k=k)


    def _standardize_keys(self, item: Dict) -> Dict:
        """Keep only necessary keys in the output dictionary.
        
        Args:
            item: Input dictionary
            
        Returns:
            Dictionary with standardized keys
        """
        keys_to_keep = {
            'image', 'gt', 'ori_gt', 'image_root',
            'gt_point_coords', 'gt_point_labels', 'gt_bboxes', 'gt_target',
            'pseudo', 'pseudo_point_coords', 'pseudo_point_labels', 'pseudo_bboxes'
        }
        return {k: v for k, v in item.items() if k in keys_to_keep}


    def _get_random_sample(self) -> Dict:
        """Get a random valid sample (used for error recovery).
        
        Returns:
            A valid sample dictionary
        """
        return self.__getitem__(random.randint(0, len(self) - 1))


def test_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for test dataloader.
    
    Args:
        batch: List of sample dictionaries (should contain only 1 item)
        
    Returns:
        Collated batch dictionary
        
    Raises:
        AssertionError: If batch size is not 1
    """
    assert len(batch) == 1, 'Batch size must be 1 in test mode'
    
    sample = batch[0]
    gt_prompt = {
        'point_coords': sample['gt_point_coords'],
        'point_labels': sample['gt_point_labels'],
        'bboxes': sample['gt_bboxes']
    }
    
    return {
        'image': sample['image'].unsqueeze(0),
        'label': sample['gt'],
        'ori_label': sample['ori_gt'],
        'gt_prompt': gt_prompt,
        'target_list': sample['gt_target'],
        'image_root': sample['image_root']
    }


def train_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for training dataloader.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Collated batch dictionary with combined tensors
    """
    # Initialize collectors
    images = []
    labels = []
    pseudos = []
    target_list = []
    gt_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    pseudo_prompt = {'point_coords': [], 'point_labels': [], 'bboxes': []}
    
    # Collect from each sample
    for sample in batch:
        images.append(sample['image'])
        labels.append(sample['gt'])
        pseudos.append(sample['pseudo'])
        target_list.extend(sample['gt_target'])
        
        # Collect GT prompts
        gt_prompt['point_coords'].append(sample['gt_point_coords'])
        gt_prompt['point_labels'].append(sample['gt_point_labels'])
        gt_prompt['bboxes'].append(sample['gt_bboxes'])
        
        # Collect pseudo prompts
        pseudo_prompt['point_coords'].append(sample['pseudo_point_coords'])
        pseudo_prompt['point_labels'].append(sample['pseudo_point_labels'])
        pseudo_prompt['bboxes'].append(sample['pseudo_bboxes'])
    
    # Stack/concatenate tensors
    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    pseudos = torch.cat(pseudos, dim=0)
    
    # Concatenate prompt tensors
    gt_prompt = {
        key: torch.cat(value, dim=0) if value else None 
        for key, value in gt_prompt.items()
    }
    pseudo_prompt = {
        key: torch.cat(value, dim=0) if value else None 
        for key, value in pseudo_prompt.items()
    }
    
    return {
        'image': images,
        'label': labels,
        'pseudo': pseudos,
        'target_list': target_list,
        'gt_prompt': gt_prompt,
        'pseudo_prompt': pseudo_prompt
    }


def get_loader(args):

    dataset_json = os.path.join(args.data_dir, 'dataset.json')
    dataset_dict = json.load(open(dataset_json, 'r'))
    #target_size = (args.image_size, args.image_size)
    target_size = (args.model.image_size, args.model.image_size)

    mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    # if args.test_mode:
    if args.model.test_mode:
        datalist = dataset_dict['test']
        collate_fn = test_collate_fn
        transform = transforms.Compose(
                        [
                            transforms.ToTensord(keys=["image", "label"]),
                            transforms.NormalizeIntensityd(keys=["image"], subtrahend=mean, divisor=std),
                            transforms.Resized(keys=["image", "label"], spatial_size=target_size, mode="nearest"),
                        ]
                    )
    else:
        datalist = dataset_dict['training']
        collate_fn = train_collate_fn
        transform = transforms.Compose(
                [
                    transforms.ToTensord(keys=["image", "label", "pseudo"]),
                    transforms.NormalizeIntensityd(keys=["image"], subtrahend=mean, divisor=std),
                    transforms.Resized(keys=["image", "label", "pseudo"], spatial_size=target_size, mode="nearest"),
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

    sampler = DistributedSampler(dataset) if args.device.multi_gpu.enabled else None

    data_loader = data.DataLoader(
        dataset,
        batch_size=args.model.batch_size,
        shuffle=(sampler is None),
        num_workers=args.dataset.num_workers,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    return data_loader


def setup_distributed_training():
    """Setup distributed training environment."""
    os.environ["USE_LIBUV"] = "0"
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend='gloo',
            init_method='tcp://localhost:23456',
            rank=0,
            world_size=1
        )


def create_argument_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Medical Image Segmentation Data Loader")
    
    parser.add_argument("--data_dir", type=str, default="D:/Kai/DATA_Set_2/medical-segmentation/BTCV", help="Path to dataset directory")
    parser.add_argument('--image_size', type=int, default=256, help="Target image size for resizing")
    parser.add_argument('--test_mode', type=bool, default=False, help='Use test dataset instead of training dataset')
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for data loading")
    parser.add_argument('--dist', type=bool, default=True, help='Enable distributed training')
    parser.add_argument('--num_workers', type=int, default=1,help="Number of worker processes for data loading")
    parser.add_argument('--mask_num', type=int, default=5, help="Number of masks")

    parsed_args = parser.parse_args()
    return parsed_args


def main():
    setup_distributed_training()

    # config = create_argument_parser()
    config = OmegaConf.load("./configs/dataset_test_config.yaml")
    LoggingManager.print_config(config, "Configuration")
    
    loader = get_loader(config)
    # print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of batches: {len(loader)}\n")

    for batch_idx, batch in enumerate(loader):
        image = batch["image"]
        label = batch["label"]
        bbox = batch['gt_prompt']['bboxes']
        # pseudo = batch['pseudo']
        
        print(f"Batch {batch_idx + 1}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Label shape: {label.shape}")
        print(f"  Bounding Box shape: {bbox.shape}")
        # print(f"  Pseudo shape: {batch['pseudo'].shape}")

        print(f"  Data type: {image.dtype}")
        print(f"  Data range: {image.max()}, {image.min()}")

        print(f"  Target Box size: {len(batch['target_list'])}")
        print(f"  Target list: {batch['target_list']}\n")
        
        # Only show first 2 batches
        if batch_idx >= 1:
            break


if __name__ == "__main__":
    main()

"""
Number of batches: 480

Batch 1:
  Image shape: torch.Size([4, 3, 256, 256])
  Label shape: torch.Size([20, 1, 256, 256])
  Bounding Box shape: torch.Size([20, 1, 1, 4])
  Data type: torch.float32
  Data range: 2.98840069770813, -2.4133365154266357
  Target Box size: 20
  Target list: ['inferior_vena_cava', 'aorta', 'aorta', 'aorta', 'esophagus', 'liver', 'liver', 'liver', 'inferior_vena_cava', 'inferior_vena_cava', 'aorta', 'inferior_vena_cava', 'inferior_vena_cava', 'inferior_vena_cava', 'inferior_vena_cava', 'inferior_vena_cava', 'pancreas', 'stomach', 'liver', 'kidney_left']

Batch 2:
  Image shape: torch.Size([4, 3, 256, 256])
  Label shape: torch.Size([20, 1, 256, 256])
  Bounding Box shape: torch.Size([20, 1, 1, 4])
  Data type: torch.float32
  Data range: 2.622570753097534, -2.1179039478302
  Target Box size: 20
  Target list: ['kidney_left', 'liver', 'spleen', 'gallbladder', 'inferior_vena_cava', 'kidney_left', 'aorta', 'inferior_vena_cava', 'kidney_right', 'liver', 'pancreas', 'aorta', 'inferior_vena_cava', 'gallbladder', 'kidney_left', 'aorta', 'aorta', 'aorta', 'esophagus', 'esophagus']
"""