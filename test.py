import os
import torch
import random
import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from rich.logging import RichHandler
from tqdm import tqdm
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader 
from omegaconf import OmegaConf
from utils import FocalDice_MSELoss

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
    def __init__(self, model: nn.Module, dataloaders: DataLoader, args: OmegaConf):
        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self._setup_loss_functions()
        if args.pretrain_path is not None:
            self.load_checkpoint(args.pretrain_path)
        else:
            self.start_epoch = 0

    def _setup_loss_functions(self):
        self.seg_loss = FocalDice_MSELoss()
        self.ce_loss = CrossEntropyLoss()


    def load_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.device.multi_gpu.enabled:
                torch.distributed.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device.device)
        
        if last_ckpt:
            self.model.load_state_dict(last_ckpt['model_state_dict'])
            self.start_epoch = last_ckpt['epoch']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
            
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    
    def get_iou_and_dice(self, pred, label):
        assert pred.shape == label.shape
        pred = (torch.sigmoid(pred) > 0.5)
        label = (label > 0)
        intersection = torch.logical_and(pred, label).sum(dim=(1, 2, 3)) 
        union = torch.logical_or(pred, label).sum(dim=(1, 2, 3))  
        iou = intersection.float() / (union.float() + 1e-8) 
        dice = (2 * intersection.float()) / (pred.sum(dim=(1, 2, 3)) + label.sum(dim=(1, 2, 3)) + 1e-8) 
        return iou.mean().item(), dice.mean().item()

    def postprocessing_mask(self, pred_masks, ori_size):
        masks = F.interpolate(pred_masks, ori_size, mode='bilinear')
        return masks

    def interaction(self, model, image_embedding, low_masks, mask_preds, labels):
        with torch.no_grad():
            for inter in range(self.args.dataset.inter_num-1):
                prompts = model.supervised_prompts(None, labels, mask_preds, low_masks, 'points')
                #outputs = model.forward_decoder(image_embedding, prompts)
                outputs = model.decode_masks(image_embedding, prompts)
                
                mask_preds, low_masks = outputs['masks'], outputs['low_res_masks']
            loss = self.seg_loss(mask_preds, labels.float(), outputs['iou_pred'])
        return loss, mask_preds


    def test(self):
        self.model.eval()
        if self.args.device.multi_gpu.enabled:
            model = self.model.module
            torch.distributed.barrier()
        else:
            model = self.model
   
        tbar = tqdm(self.dataloaders)
        l = len(self.dataloaders)

        category_level_metrics = {}
        avg_dice = []
        for step, batch_input in enumerate(tbar): 
            images, labels = batch_input["image"].to(self.args.device.device), batch_input["label"].to(self.args.device.device).type(torch.long)
            ori_labels = batch_input["ori_label"].to(self.args.device.device).type(torch.long)
            target_list = batch_input['target_list']

            gt_prompt = batch_input["gt_prompt"]
            image_root = batch_input["image_root"][0]
      
            #text_prompt = model.process_text_prompt(target_list)
            text_prompt = model.text_tokenizer(target_list)
            #image_embedding = model.image_forward(images)
            image_embedding = model.encode_image(images)

            test_prompts = {}
            image_level_metrics = {'loss':[],'iou':[],'dice':[],'category_pred':[]}
            for cls_idx in range(len(target_list)):
                labels_cls = labels[cls_idx:cls_idx+1]
                ori_labels_cls = ori_labels[cls_idx:cls_idx+1]
                if self.args.dataset.prompt_mode == 'bboxes':
                    test_prompts['bboxes'] = gt_prompt['bboxes'][cls_idx:cls_idx+1].to(self.args.device.device)
                elif self.args.dataset.prompt_mode == 'points':
                    test_prompts['point_coords'] = gt_prompt['point_coords'][cls_idx:cls_idx+1].to(self.args.device.device)
                    test_prompts['point_labels'] = gt_prompt['point_labels'][cls_idx:cls_idx+1].to(self.args.device.device)
                elif self.args.dataset.prompt_mode == 'text':
                    test_prompts['text_inputs'] = text_prompt['text_inputs'][cls_idx:cls_idx+1].to(self.args.device.device)
                else:
                    print('Please setting correct prompt mode')

                with torch.no_grad():
                    #outputs = model.forward_decoder(image_embedding, test_prompts)
                    outputs = model.decode_masks(image_embedding, test_prompts)
                    
                    mask_preds, low_masks = outputs['masks'], outputs['low_res_masks']
                    loss = self.seg_loss(mask_preds, labels_cls.float(), outputs['iou_pred'])
          
                if self.args.dataset.inter_num > 1:
                    image_embedding = image_embedding.detach()
                    loss, mask_preds = self.interaction(model, image_embedding, low_masks,  mask_preds, labels_cls)

                ori_preds = self.postprocessing_mask(mask_preds, ori_labels.shape[-2:])

                category_iou, category_dice = self.get_iou_and_dice(ori_preds, ori_labels_cls)

                image_level_metrics['loss'].append(loss.item())
                image_level_metrics['iou'].append(category_iou)
                image_level_metrics['dice'].append(category_dice)
    
            loss = np.mean(image_level_metrics['loss'])
            iou = np.mean(image_level_metrics['iou'])
            dice = np.mean(image_level_metrics['dice'])
            avg_dice.append(dice)
       
            logger.info(f"{image_root}, loss: {loss:.4f}, iou: {iou:.4f}, dice: {dice:.4f}")

        if self.args.device.multi_gpu.enabled:
            local_dice = torch.tensor([float(np.mean(avg_dice))]).to(self.args.device.device)
            torch.distributed.all_reduce(local_dice, op=torch.distributed.ReduceOp.SUM) 
            Avg_dice = local_dice.item() / torch.distributed.get_world_size()
        else:
            Avg_dice = np.mean(avg_dice)

        logger.info(f"{'*'*10} Image Avg Dice: {Avg_dice:.4f} {'*'*10}")

        logger.info(f'args : {self.args}')
        logger.info('=====================================================================')


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


def print_config(config) -> None:
    """Print configuration parameters in a formatted way.
    
    Args:
        config: Configuration object to print
    """
    print('=' * 100)
    print('Configuration:')
    for key, value in vars(config).items():
        print(f'  {key}: {value}')
    print('=' * 100)


def run_single_gpu(config) -> None:
    """Run training/testing on single GPU or CPU."""
    # Initialize seeds for reproducibility
    init_seeds(seed=42, cuda_deterministic=True)
    
    # Load data and model
    device = determine_device(config)
    dataloaders = get_loader(config)
    model, _ = load_model(config, device)
    
    # Initialize tester and run
    tester = BaseTester(model, dataloaders, config)
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
    print_config(config)
    
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
