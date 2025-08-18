import os
import torch
import random
import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from rich.logging import RichHandler
from pathlib import Path
from tqdm import tqdm
from torch.backends import cudnn
from torch.utils.data import DataLoader 
from typing import Any, Dict, List, Tuple, Union
from omegaconf import OmegaConf
from utils import FocalDiceMSELoss

# Import custom modules
from data_loader import get_loader 
from configs.config import parse_args
from src.utils.inference import determine_device, load_model


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # DEBUG INFO
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


class BaseTester:
    """
    Base class for model testing with checkpoint loading capabilities.
    
    This class provides a unified interface for testing PyTorch models with support
    for distributed training, checkpoint loading, and interactive refinement.
    
    Args:
        config: Configuration object containing test parameters
        model: PyTorch model to test
        dataloaders: DataLoader for test data
        device: Device to run testing on
        
    Attributes:
        model: The PyTorch model
        dataloaders: Test data loaders
        config: Configuration settings
        device: Computation device
        start_epoch: Starting epoch for testing
        seg_loss: Segmentation loss function
        ce_loss: Cross-entropy loss function
    """
    
    def __init__(
        self, 
        config: OmegaConf,
        model: nn.Module, 
        dataloaders: DataLoader, 
        device: torch.device
    ) -> None:
        self.config = config
        self.model = model
        self.dataloaders = dataloaders
        self.device = device
        self.start_epoch = 0
        
        # Initialize loss functions
        self._setup_loss_functions()
        
        # Load checkpoint if specified
        if config.get('pretrain_path'):
            self._load_checkpoint(config.pretrain_path)
    

    def _setup_loss_functions(self) -> None:
        """Initialize loss functions used during testing."""
        self.seg_loss = FocalDiceMSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    

    def _load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model checkpoint from specified path."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(
                f"Checkpoint not found at {checkpoint_path}. Starting from epoch 0."
            )
            return
        
        try:
            # Handle distributed training barrier
            if self._is_distributed():
                torch.distributed.barrier()
            
            # Load checkpoint and model state
            checkpoint = torch.load(
                checkpoint_path, 
                map_location=self.device, 
                weights_only=True
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Update start epoch
            self.start_epoch = checkpoint.get('epoch', 0)
            
            logger.info(
                f"Successfully loaded checkpoint from {checkpoint_path} "
                f"(epoch {self.start_epoch})"
            )
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise RuntimeError(f"Checkpoint loading failed: {e}") from e
    

    def _is_distributed(self) -> bool:
        """Check if distributed training is enabled."""
        return (
            hasattr(self.config, 'device') and 
            hasattr(self.config.device, 'multi_gpu') and
            self.config.device.multi_gpu.get('enabled', False)
        )
    

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about the loaded checkpoint."""
        return {
            'start_epoch': self.start_epoch,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'device': next(self.model.parameters()).device,
        }
    

    def _compute_metrics(self, pred: torch.Tensor, label: torch.Tensor) -> Tuple[float, float]:
        """
        Compute IoU and Dice metrics for predictions and labels.
        
        Args:
            pred: Predicted masks [B, C, H, W]
            label: Ground truth labels [B, C, H, W]
            
        Returns:
            Tuple of (IoU, Dice) scores
        """
        assert pred.shape == label.shape, f"Shape mismatch: {pred.shape} vs {label.shape}"
        
        # Apply sigmoid and threshold for predictions
        pred_binary = (torch.sigmoid(pred) > 0.5)
        label_binary = (label > 0)
        
        # Calculate intersection and union
        intersection = torch.logical_and(pred_binary, label_binary).sum(dim=(1, 2, 3))
        union = torch.logical_or(pred_binary, label_binary).sum(dim=(1, 2, 3))
        
        # Calculate metrics with epsilon for numerical stability
        eps = 1e-8
        iou = intersection.float() / (union.float() + eps)
        
        pred_sum = pred_binary.sum(dim=(1, 2, 3))
        label_sum = label_binary.sum(dim=(1, 2, 3))
        dice = (2 * intersection.float()) / (pred_sum + label_sum + eps)
        
        return iou.mean().item(), dice.mean().item()
    

    def _interactive_refinement(
        self, 
        model: nn.Module, 
        image_embedding: torch.Tensor, 
        low_masks: torch.Tensor, 
        mask_preds: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform interactive mask refinement.
        
        Args:
            model: The model to use for refinement
            image_embedding: Encoded image features
            low_masks: Low resolution masks
            mask_preds: Current mask predictions
            labels: Ground truth labels
            
        Returns:
            Tuple of (final_loss, refined_mask_predictions)
        """
        with torch.no_grad():
            for _ in range(self.config.dataset.inter_num - 1):
                prompts = model.supervised_prompts(
                    None, labels, mask_preds, low_masks, 'points'
                )
                outputs = model.decode_masks(image_embedding, prompts)
                mask_preds, low_masks = outputs['masks'], outputs['low_res_masks']
            
            loss = self.seg_loss(mask_preds, labels.float(), outputs['iou_pred'])
        
        return loss, mask_preds
    

    def _prepare_prompts(
        self, 
        prompt_mode: str, 
        gt_prompt: Dict[str, torch.Tensor], 
        text_prompt: Dict[str, torch.Tensor], 
        cls_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare prompts based on the specified mode.
        
        Args:
            prompt_mode: Type of prompt ('bboxes', 'points', 'text')
            gt_prompt: Ground truth prompt data
            text_prompt: Text prompt data
            cls_idx: Class index
            
        Returns:
            Prepared prompts dictionary
            
        Raises:
            ValueError: If prompt mode is not supported
        """
        prompts = {}
        
        if prompt_mode == 'bboxes':
            prompts['bboxes'] = gt_prompt['bboxes'][cls_idx:cls_idx+1].to(self.device)
        elif prompt_mode == 'points':
            prompts['point_coords'] = gt_prompt['point_coords'][cls_idx:cls_idx+1].to(self.device)
            prompts['point_labels'] = gt_prompt['point_labels'][cls_idx:cls_idx+1].to(self.device)
        elif prompt_mode == 'text':
            prompts['text_inputs'] = text_prompt['text_inputs'][cls_idx:cls_idx+1].to(self.device)
        else:
            raise ValueError(f"Unsupported prompt mode: {prompt_mode}")
        
        return prompts
    

    def _process_single_class(
        self,
        model: nn.Module,
        image_embedding: torch.Tensor,
        text_prompt: Dict[str, torch.Tensor],
        gt_prompt: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        ori_labels: torch.Tensor,
        cls_idx: int
    ) -> Dict[str, float]:
        """
        Process a single class prediction and compute metrics.
        
        Args:
            model: The model to use for prediction
            image_embedding: Encoded image features
            text_prompt: Text prompt data
            gt_prompt: Ground truth prompt data
            labels: Ground truth labels
            ori_labels: Original resolution labels
            cls_idx: Class index
            
        Returns:
            Dictionary containing loss, IoU, and Dice metrics
        """
        # Extract class-specific data
        labels_cls = labels[cls_idx:cls_idx+1]
        ori_labels_cls = ori_labels[cls_idx:cls_idx+1]
        
        # Prepare prompts
        test_prompts = self._prepare_prompts(
            self.config.dataset.prompt_mode,
            gt_prompt,
            text_prompt,
            cls_idx
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = model.decode_masks(image_embedding, test_prompts)
            mask_preds, low_masks = outputs['masks'], outputs['low_res_masks']
            loss = self.seg_loss(mask_preds, labels_cls.float(), outputs['iou_pred'])
        
        # Interactive refinement if enabled
        if self.config.dataset.inter_num > 1:
            image_embedding_detached = image_embedding.detach()
            loss, mask_preds = self._interactive_refinement(
                model, image_embedding_detached, low_masks, mask_preds, labels_cls
            )
        
        # Resize predictions to original resolution
        ori_preds = F.interpolate(
            mask_preds, 
            ori_labels.shape[-2:], 
            mode='bilinear',
            align_corners=False
        )
        
        # Compute metrics
        category_iou, category_dice = self._compute_metrics(ori_preds, ori_labels_cls)
        
        return {
            'loss': loss.item(),
            'iou': category_iou,
            'dice': category_dice
        }
    

    def _aggregate_distributed_metrics(self, local_metrics: List[float]) -> float:
        """
        Aggregate metrics across distributed processes.
        
        Args:
            local_metrics: List of local metric values
            
        Returns:
            Aggregated metric value
        """
        if self._is_distributed():
            local_tensor = torch.tensor([np.mean(local_metrics)]).to(self.device)
            torch.distributed.all_reduce(local_tensor, op=torch.distributed.ReduceOp.SUM)
            return local_tensor.item() / torch.distributed.get_world_size()
        else:
            return np.mean(local_metrics)
    

    def test(self) -> Dict[str, float]:
        """
        Run the full testing loop.
        
        Returns:
            Dictionary containing final test metrics
        """
        self.model.eval()
        
        # Get the actual model (handle distributed wrapper)
        if self._is_distributed():
            model = self.model.module
            torch.distributed.barrier()
        else:
            model = self.model
        
        # Initialize metrics tracking
        all_dice_scores = []
        all_iou =[]
        all_loss = []

        # Create progress bar
        progress_bar = tqdm(self.dataloaders, desc="Testing")
        
        for step, batch_input in enumerate(progress_bar):
            # Move data to device
            images = batch_input["image"].to(self.device)
            labels = batch_input["label"].to(self.device).type(torch.long)
            ori_labels = batch_input["ori_label"].to(self.device).type(torch.long)
            target_list = batch_input['target_list']
            gt_prompt = batch_input["gt_prompt"]
            image_root = batch_input["image_root"][0]
            
            # Encode image and text
            text_prompt = model.text_tokenizer(target_list)
            image_embedding = model.encode_image(images)
            
            # Process each class
            image_metrics = {'loss': [], 'iou': [], 'dice': []}
            
            for cls_idx in range(len(target_list)):
                class_metrics = self._process_single_class(
                    model, image_embedding, text_prompt, gt_prompt,
                    labels, ori_labels, cls_idx
                )
                
                # Accumulate metrics
                for key in image_metrics:
                    image_metrics[key].append(class_metrics[key])
            
            # Compute average metrics for this image
            avg_loss = np.mean(image_metrics['loss'])
            avg_iou = np.mean(image_metrics['iou'])
            avg_dice = np.mean(image_metrics['dice'])
            
            all_dice_scores.append(avg_dice)
            all_iou.append(avg_iou)
            all_loss.append(avg_loss)

            # Update progress bar and log
            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'iou': f"{avg_iou:.4f}",
                'dice': f"{avg_dice:.4f}"
            })
            
            logger.info(
                f"{image_root} - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, "
                f"Dice: {avg_dice:.4f}"
            )
        
        # Aggregate final metrics
        final_dice = self._aggregate_distributed_metrics(all_dice_scores)
        final_iou = self._aggregate_distributed_metrics(all_iou)
        final_loss = self._aggregate_distributed_metrics(all_loss)

        logger.info(f"{'='*50}")
        logger.info(f"Final Average Dice Score: {final_dice:.4f}")
        logger.info(f"Final Average IoU: {final_iou:.4f}")
        logger.info(f"Final Average Loss: {final_loss:.4f}")
        logger.info(f"{'='*50}")
        
        return {
            'average_dice': final_dice,
            'total_samples': len(all_dice_scores)
        }


def init_seeds(seed: int = 0, cuda_deterministic: bool = True) -> None:
    """Initialize random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        cuda_deterministic: If True, use deterministic CUDA operations (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if cuda_deterministic:
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            cudnn.benchmark = True


def setup_distributed(rank: int, world_size: int, port: int = 12355) -> None:
    """Initialize distributed training setup.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        port: Master port for communication
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    
    # Use NCCL for CUDA, gloo for CPU/MPS or Windows
    backend = 'nccl' if torch.cuda.is_available() and os.name != 'nt' else 'gloo'
    
    torch.distributed.init_process_group(
        backend=backend, 
        init_method='env://', 
        rank=rank, 
        world_size=world_size
    )


def cleanup_distributed() -> None:
    """Clean up distributed training resources."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def run_single_gpu(config) -> None:
    """Run training/testing on single GPU or CPU."""
    # Initialize seeds for reproducibility
    init_seeds(seed=42, cuda_deterministic=True)
    
    # Load data and model
    device = determine_device(config)
    dataloaders = get_loader(config)
    model, _ = load_model(config, device)
    
    # Initialize tester and run
    tester = BaseTester(config, model, dataloaders, device)
    tester.test()


def main_worker(rank: int, config) -> None:
    """Main worker function for distributed training.
    
    Args:
        rank: Process rank
        config: Configuration object
    """
    try:
        # Setup distributed training
        setup_distributed(rank, config.world_size, getattr(config, 'port', 12355))
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            config.device = torch.device(f"cuda:{rank}")
        else:
            config.device = torch.device('cpu')
            
        config.rank = rank
        config.gpu_info = {"gpu_count": config.world_size, 'gpu_name': rank}
        
        # Initialize seeds with rank offset for different initialization per process
        init_seeds(2023 + rank, cuda_deterministic=True)
        
        # Load data and model
        dataloaders = get_loader(config)
        model, _ = load_model(config, config.device)
        
        # Initialize tester and run
        tester = BaseTester(model, dataloaders, config)
        tester.test()
        
    except Exception as e:
        logging.error(f"Error in worker {rank}: {e}")
        raise
    finally:
        cleanup_distributed()


def main():
    """Main entry point for the application."""
    config = parse_args()
    
    # Set multiprocessing sharing strategy
    mp.set_sharing_strategy('file_system')
    
    # Run training/testing
    if config.device.multi_gpu.enabled:
        mp.spawn(
            main_worker, 
            nprocs=config.world_size, 
            args=(config,)
        )
    else:
        run_single_gpu(config)
        

if __name__ == '__main__':
    # Set multiprocessing start method
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    main()

# parser = argparse.ArgumentParser()
# parser.add_argument('--work_dir', type=str, default='work_dir')
# parser.add_argument('--task_name', type=str, default='ft-IMISNet')
# #load data
# parser.add_argument("--data_dir", type = str, default="D:/Kai/DATA_Set_2/medical-segmentation/BTCV")
# parser.add_argument('--image_size', type=int, default=1024)
# parser.add_argument('--test_mode', type=bool, default=True)
# parser.add_argument('--batch_size', type=int, default=1)
# #load model
# parser.add_argument('--model_type', type=str, default='vit_b')
# parser.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')
# parser.add_argument('--pretrain_path', type=str, default=None)
# parser.add_argument('--device', type=str, default='cuda')
# parser.add_argument('--mask_num', type=int, default=None)
# parser.add_argument('--prompt_mode', type=str, default='points')
# parser.add_argument('--inter_num', type=int, default=1)
# # train
# parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0]) 
# parser.add_argument('--multi_gpu', action='store_true', default=False)
# parser.add_argument('--port', type=int, default=12361)
# parser.add_argument('--dist', dest='dist', type=bool, default=False, help='distributed training or not')
# parser.add_argument('-num_workers', type=int, default=1)

# args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])



# LOG_OUT_DIR = join(args.work_dir, args.task_name)

# device = args.device
# MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
# os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
