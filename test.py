# set up environment
import numpy as np
import random 
import matplotlib.pyplot as plt
import os
import csv
import ast
join = os.path.join
from tqdm import tqdm
from torch.backends import cudnn
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from segment_anything import sam_model_registry
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import logging
from model import IMISNet
from utils import FocalDice_MSELoss
from torch.nn import CrossEntropyLoss
import warnings
import re
from data_loader import get_loader 

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--task_name', type=str, default='ft-IMISNet')
#load data
parser.add_argument("--data_dir", type = str, default='dataset/BTCV')
parser.add_argument('--image_size', type=int, default=1024)
parser.add_argument('--test_mode', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=1)
#load model
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')
parser.add_argument('--pretrain_path', type=str, default=None)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--mask_num', type=int, default=None)
parser.add_argument('--prompt_mode', type=str, default='points')
parser.add_argument('--inter_num', type=int, default=1)
# train
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0]) 
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--port', type=int, default=12361)
parser.add_argument('--dist', dest='dist', type=bool, default=False, help='distributed training or not')
parser.add_argument('-num_workers', type=int, default=1)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])

logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)

device = args.device
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def build_model(args):
    category_weights = 'dataloaders/categories_weight.pkl'
    sam = sam_model_registry[args.model_type](args).to(device)
    imis = IMISNet(sam, test_mode=args.test_mode, category_weights=category_weights).to(device)
    if args.multi_gpu:
        imis = DDP(imis, device_ids=[args.rank], output_device=args.rank)
    return imis


class BaseTester:
    def __init__(self, model, dataloaders, args):
        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.set_loss_fn()
        if args.pretrain_path is not None:
            self.load_checkpoint(args.pretrain_path)
        else:
            self.start_epoch = 0

    def set_loss_fn(self):
        self.seg_loss = FocalDice_MSELoss()
        self.ce_loss = CrossEntropyLoss()


    def load_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device)
        
        if last_ckpt:
            if self.args.multi_gpu:
                try:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'])
                except:
                    self.model.load_state_dict(last_ckpt['model_state_dict'], False)
            else:
                try:
                    self.model.load_state_dict(last_ckpt['model_state_dict'])
                except:
                    self.model.load_state_dict(last_ckpt['model_state_dict'], False)
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
            for inter in range(self.args.inter_num-1):
                prompts = model.supervised_prompts(None, labels, mask_preds, low_masks, 'points')
                outputs = model.forward_decoder(image_embedding, prompts)
                mask_preds, low_masks = outputs['masks'], outputs['low_res_masks']
            loss = self.seg_loss(mask_preds, labels.float(), outputs['iou_pred'])
        return loss, mask_preds


    def test(self):
        self.model.eval()
        if self.args.multi_gpu:
            model = self.model.module
            dist.barrier()
        else:
            model = self.model
   
        tbar = tqdm(self.dataloaders)
        l = len(self.dataloaders)

        category_level_metrics = {}
        avg_dice = []
        for step, batch_input in enumerate(tbar): 
            images, labels = batch_input["image"].to(device), batch_input["label"].to(device).type(torch.long)
            ori_labels = batch_input["ori_label"].to(device).type(torch.long)
            target_list = batch_input['target_list']

            gt_prompt = batch_input["gt_prompt"]
            image_root = batch_input["image_root"][0]
      
            text_prompt = model.process_text_prompt(target_list)
            image_embedding = model.image_forward(images)

            test_prompts = {}
            image_level_metrics = {'loss':[],'iou':[],'dice':[],'category_pred':[]}
            for cls_idx in range(len(target_list)):
                labels_cls = labels[cls_idx:cls_idx+1]
                ori_labels_cls = ori_labels[cls_idx:cls_idx+1]
                if self.args.prompt_mode == 'bboxes':
                    test_prompts['bboxes'] = gt_prompt['bboxes'][cls_idx:cls_idx+1].to(device)
                elif self.args.prompt_mode == 'points':
                    test_prompts['point_coords'] = gt_prompt['point_coords'][cls_idx:cls_idx+1].to(device)
                    test_prompts['point_labels'] = gt_prompt['point_labels'][cls_idx:cls_idx+1].to(device)
                elif self.args.prompt_mode == 'text':
                    test_prompts['text_inputs'] = text_prompt['text_inputs'][cls_idx:cls_idx+1].to(device)
                else:
                    print('Please setting correct prompt mode')

                with torch.no_grad():
                    outputs = model.forward_decoder(image_embedding, test_prompts)
                    mask_preds, low_masks = outputs['masks'], outputs['low_res_masks']
                    loss = self.seg_loss(mask_preds, labels_cls.float(), outputs['iou_pred'])
          
                if self.args.inter_num > 1:
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

        if self.args.multi_gpu:
            local_dice = torch.tensor([float(np.mean(avg_dice))]).to(self.args.device)
            dist.all_reduce(local_dice, op=dist.ReduceOp.SUM) 
            Avg_dice = local_dice.item() / dist.get_world_size()
        else:
            Avg_dice = np.mean(avg_dice)

        logger.info(f"{'*'*10} Image Avg Dice: {Avg_dice:.4f} {'*'*10}")

        logger.info(f'args : {self.args}')
        logger.info('=====================================================================')


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic: 
        cudnn.deterministic = True
        cudnn.benchmark = False
    else: 
        cudnn.deterministic = False
        cudnn.benchmark = True

def device_config(args):
    try:
        if not args.multi_gpu:
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node
    except RuntimeError as e:
        print(e)

def main():
    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args, ))
    else:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        dataloaders = get_loader(args)
        model = build_model(args)
        tester = BaseTester(model, dataloaders, args)
        tester.test()

def main_worker(rank, args):
    setup(rank, args.world_size)
    torch.cuda.set_device(rank)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank
    args.gpu_info = {"gpu_count":args.world_size, 'gpu_name':rank}
    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'{args.prompt_mode}_output_{cur_time}.log')) 
   
    dataloaders = get_loader(args)
    model = build_model(args)
    tester = BaseTester(model, dataloaders, args)
    tester.test()
    cleanup()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = f'{args.port}'
    dist.init_process_group(backend='NCCL', init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
