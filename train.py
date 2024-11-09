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
from segment_anything import sam_model_registry
import argparse
from torch.cuda import amp
import torch.multiprocessing as mp
from multiprocessing import Manager
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import logging
from data_loader import get_loader 
from model import IMISNet
from utils import FocalDice_MSELoss
from torch.nn import CrossEntropyLoss
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', type=str, default='work_dir')
parser.add_argument('--task_name', type=str, default='ft-IMISNet')
#load data
parser.add_argument("--data_dir", type = str, default='dataset/BTCV')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--test_mode', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=10)
#load model
parser.add_argument('--model_type', type=str, default='vit_b')
parser.add_argument('--sam_checkpoint', type=str, default='ckpt/IMISNet-B.pth')
parser.add_argument('--pretrain_path', type=str, default='work_dir/ft-IMISNet/IMIS_latest.pth')
parser.add_argument('--resume', action='store_true', default=True)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--mask_num', type=int, default=2)
parser.add_argument('--inter_num', type=int, default=4)
# train
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--lr_scheduler', type=str, default=None)
parser.add_argument('--step_size', type=list, default=[7,12]) 
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--port', type=int, default=12305)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])
parser.add_argument('--multi_gpu', action='store_true', default=False)
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
    imis = IMISNet(sam, test_mode=args.test_mode, select_mask_num=args.mask_num, category_weights=category_weights).to(device)
    if args.multi_gpu:
        imis = DDP(imis, device_ids=[args.rank], output_device=args.rank)
    return imis

class BaseTrainer:
    def __init__(self, model, dataloaders, args):
        self.model = model
        self.dataloaders = dataloaders
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.best_iou = 0.0
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if args.pretrain_path is not None:
            self.load_checkpoint(args.pretrain_path, args.resume)
        else:
            self.start_epoch = 0

    def set_loss_fn(self):
        self.seg_loss = FocalDice_MSELoss()
        self.ce_loss = CrossEntropyLoss()

    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay) #

    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                self.args.step_size,
                                                                self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def load_checkpoint(self, ckp_path, resume):
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
                except Exception as e:
                    print(f"Failed to load model state dict: {e}")
                    self.model.load_state_dict(last_ckpt['model_state_dict'], False)
            else:
                try:
                    self.model.load_state_dict(last_ckpt['model_state_dict'])
                except Exception as e:
                    print(f"Failed to load model state dict: {e}")
                    self.model.load_state_dict(last_ckpt['model_state_dict'], False)
            if resume:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.ious = last_ckpt['ious']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
            else:
                self.start_epoch = 0
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
            
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    
    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "losses": self.losses,
            "ious": self.ious,
            "dices": self.dices,
            "best_loss": self.best_loss,
            "best_iou": self.best_iou,
            "best_dice": self.best_dice,
            "args": self.args,
        }, join(MODEL_SAVE_PATH, f"IMIS_{describe}.pth"))


    def get_iou_and_dice(self, pred, label):
        assert pred.shape == label.shape
        pred = (torch.sigmoid(pred) > 0.5)
        label = (label > 0)
        intersection = torch.logical_and(pred, label).sum(dim=(1, 2, 3)) 
        union = torch.logical_or(pred, label).sum(dim=(1, 2, 3))  
        iou = intersection.float() / (union.float() + 1e-8) 
        dice = (2 * intersection.float()) / (pred.sum(dim=(1, 2, 3)) + label.sum(dim=(1, 2, 3)) + 1e-8) 
        return iou.mean().item(), dice.mean().item()

    def plot_result(self, plot_data, description, save_name):
        plt.plot(plot_data)
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()


    def interaction(
        self, 
        model,
        image_embedding, 
        gt_low_masks, 
        pseudo_low_masks,
        gt_preds, 
        pseudo_preds, 
        labels, 
        pseudos,
                ):
        
        total_loss = 0
        text_and_mask_inter = np.random.randint(0, self.args.inter_num-1)
        with amp.autocast():
            for inter in range(self.args.inter_num):
                if inter == text_and_mask_inter or inter == self.args.inter_num-1:
                    gt_prompts = model.process_mask_prompt(gt_low_masks)
                    gt_prompts.update(self.text_prompt)

                    gt_outputs = model.forward_decoder(image_embedding, gt_prompts)
                    gt_preds, gt_low_masks = gt_outputs['masks'], gt_outputs['low_res_masks']
                    gt_loss = self.seg_loss(gt_preds, labels.float(), gt_outputs['iou_pred'])
           
                    pseudo_prompts = model.process_mask_prompt(pseudo_low_masks)

                else:
                    gt_prompts = model.supervised_prompts(None, labels, gt_preds, gt_low_masks, 'points')
                    if random.random() > 0.6:
                        gt_prompts.update(self.text_prompt)
                        del gt_prompts['mask_inputs']

                    gt_outputs = model.forward_decoder(image_embedding, gt_prompts)
                    gt_preds, gt_low_masks = gt_outputs['masks'], gt_outputs['low_res_masks']
                    gt_loss = self.seg_loss(gt_preds, labels.float(), gt_outputs['iou_pred'])

                    pseudo_prompts = model.unsupervised_prompts(pseudos, pseudo_preds, pseudo_low_masks, 'points')
                
                pseudo_outputs = model.forward_decoder(image_embedding,  pseudo_prompts)
                pseudo_preds, pseudo_low_masks = pseudo_outputs['masks'], pseudo_outputs['low_res_masks']
                pseudo_loss = self.seg_loss(pseudo_preds, pseudos.float(), pseudo_outputs['iou_pred'])

                loss = gt_loss + pseudo_loss
                if torch.isnan(loss).any():  
                    print(f"Detected NaN loss. Skipping this inter.")  
                    total_loss += 0
                    continue  
                else:
                    total_loss += loss.item()
                    self.scaler.scale(loss).backward(retain_graph=True) 
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        loss = total_loss / self.args.inter_num
        return loss, gt_preds, pseudo_preds


    def train_epoch(self, epoch):
        step_loss, step_iou, step_dice = 0, 0, 0
        self.model.train()

        if self.args.multi_gpu:
            model = self.model.module
        else:
            model = self.model

        tbar = tqdm(self.dataloaders)
        l = len(self.dataloaders)
        for step, batch_input in enumerate(tbar): 
            images, labels = batch_input["image"].to(device), batch_input["label"].to(device).type(torch.long)
            pseudos = batch_input["pseudo"].to(device)
            self.target_list = batch_input['target_list']
  
            gt_prompt = batch_input["gt_prompt"]
            pseudo_prompt = batch_input["pseudo_prompt"]
            
            gt_prompts, pseudo_prompts = {}, {}
            if torch.sum(labels) == 0 or torch.sum(pseudos) == 0:
                continue
            
            self.text_prompt = model.process_text_prompt(self.target_list)

            self.img_shape = images.shape
            image_embedding = model.image_forward(images)
   
            gt_prm = random.choices(['bboxes', 'points', 'text'], [0.4, 0.3, 0.3])[0]  #supervised specify prompt
            pse_prm = random.choices(['bboxes', 'points'], [0.5, 0.5])[0]

            if gt_prm == 'bboxes':
                gt_prompts['bboxes'] = gt_prompt['bboxes'].to(device)
            elif gt_prm == 'points':
                gt_prompts['point_coords'] = gt_prompt['point_coords'].to(device)
                gt_prompts['point_labels'] = gt_prompt['point_labels'].to(device)
            else:
                gt_prompts.update(self.text_prompt)

            if pse_prm == 'bboxes':
                pseudo_prompts['bboxes'] = pseudo_prompt['bboxes'].to(device)
            else:
                pseudo_prompts['point_coords'] = pseudo_prompt['point_coords'].to(device)
                pseudo_prompts['point_labels'] = pseudo_prompt['point_labels'].to(device)
   
            with amp.autocast():
                gt_outputs = model.forward_decoder(image_embedding, gt_prompts)
                gt_loss = self.seg_loss(gt_outputs['masks'], labels.float(), gt_outputs['iou_pred'])
       
                pseudo_outputs = model.forward_decoder(image_embedding, pseudo_prompts)
                pseudo_loss = self.seg_loss(pseudo_outputs['masks'], pseudos.float(), pseudo_outputs['iou_pred'])

                loss = gt_loss + pseudo_loss

                if torch.isnan(loss).any():  
                    print(f"Detected NaN loss at epoch {epoch}, batch {step}. Skipping this batch.")  
                    continue 
                else:
                    self.scaler.scale(loss).backward(retain_graph=False) #

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            gt_preds, gt_low_masks = gt_outputs['masks'], gt_outputs['low_res_masks']
            pseudo_preds, pseudo_low_masks = pseudo_outputs['masks'], pseudo_outputs['low_res_masks']

            image_embedding = image_embedding.detach().clone()
            self.text_prompt['text_inputs'] = self.text_prompt['text_inputs'].detach().clone()

            loss, gt_preds, pseudo_preds = self.interaction(model, image_embedding, gt_low_masks, pseudo_low_masks,
                                                            gt_preds, pseudo_preds, 
                                                            labels, pseudos
                                                            )

            gt_iou, gt_dice = self.get_iou_and_dice(gt_preds, labels)

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                
                if (step + 1) % 100 == 0:
                    pseudo_iou, pseudo_dice = self.get_iou_and_dice(pseudo_preds, pseudos)
                    print(f'Epoch: {epoch}, Step: {step+1}, Loss: {loss:.4f}, IoU: {gt_iou:.4f}, Dice: {gt_dice:.4f}, pseudo_IoU: {pseudo_iou:.4f}, pseudo_Dice: {pseudo_dice:.4f}')
                
                if gt_dice > self.step_best_dice:
                    self.step_best_dice = gt_dice
                    if gt_dice > 0.95:
                        self.save_checkpoint(epoch, model.state_dict(),
                        describe=f'{epoch}_step_dice:{"{:.4f}".format(gt_dice)}_best')

            step_loss += loss
            step_iou += gt_iou
            step_dice += gt_dice

        if self.args.multi_gpu:
            dist.barrier()
            local_loss = torch.tensor([step_loss / l]).to(self.args.device)
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM) 
            avg_loss = local_loss.item() / dist.get_world_size()

            local_iou = torch.tensor([float(step_iou / l)]).to(self.args.device)
            dist.all_reduce(local_iou, op=dist.ReduceOp.SUM) 
            avg_iou = local_iou.item() / dist.get_world_size()

            local_dice = torch.tensor([float(step_dice / l)]).to(self.args.device)
            dist.all_reduce(local_dice, op=dist.ReduceOp.SUM) 
            avg_dice = local_dice.item() / dist.get_world_size()
        else:
            avg_loss, avg_iou, avg_dice = step_loss / l, step_iou / l, step_dice / l
        
        return avg_loss, avg_iou, avg_dice

    def train(self):
        self.scaler = amp.GradScaler()
        for epoch in range(self.start_epoch, self.args.num_epochs):
            
            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')
            
            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)

            avg_loss, avg_iou, avg_dice = self.train_epoch(epoch)
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(avg_loss)
                self.ious.append(avg_iou)
                self.dices.append(avg_dice)
                print(f'Epochs: {epoch}, LR: {self.lr_scheduler.get_last_lr()}, Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}')
                logger.info(f'Epoch\t {epoch}\t LR\t {self.lr_scheduler.get_last_lr()}\t: loss: {avg_loss:.4f}, iou: {avg_iou:.4f}, dice: {avg_dice:.4f}')

                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                
                self.save_checkpoint(epoch, state_dict, describe='latest')

                if avg_loss < self.best_loss: 
                    self.best_loss = avg_loss
                    # self.save_checkpoint(epoch, state_dict, describe='loss_best')
                
                if avg_iou > self.best_iou: 
                    self.best_iou = avg_iou
                    # self.save_checkpoint(epoch, state_dict, describe='iou_best')

                # save train dice best checkpoint
                if avg_dice > self.best_dice: 
                    self.best_dice = avg_dice
                    self.save_checkpoint(epoch, state_dict, describe='dice_best')

                self.plot_result(self.losses, 'Loss', 'Loss')
                self.plot_result(self.dices, 'Dice', 'Dice')
                self.plot_result(self.ious, 'IoU', 'IoU')
      
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}, Best iou: {self.best_iou}, Best dice: {self.best_dice}')
        logger.info(f'args : {self.args}')
        logger.info('=====================================================================')



########################################## Trainer ##########################################
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
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
        # Load datasets
        dataloaders = get_loader(args)
        # Build model
        model = build_model(args)
        # Create trainer
        trainer = BaseTrainer(model, dataloaders, args)
        # Train
        trainer.train()

def main_worker(rank, args):
    setup(rank, args.world_size)
    torch.cuda.set_device(rank)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank
    args.gpu_info = {"gpu_count":args.world_size, 'gpu_name':rank}
    init_seeds(2024 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        filemode='w',
        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))
    
    dataloaders = get_loader(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = f'{args.port}'
    dist.init_process_group(backend='NCCL', init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
