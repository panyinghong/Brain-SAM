
import argparse
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import logging
import sys
import time
from monai.losses import  DiceLoss 
import torch.nn.functional as F
from modeling.Mask_decoder import new_UNETR_Decoder_combine_auto_prompt

import torch
from modeling.prompt_encoder import PromptEncoder3D , TwoWayTransformer
import torch.nn as nn
from functools import partial
import os
import random
from collections import defaultdict
from utils.util import setup_logger
from modeling.Hiera_encoder import  Hiera
from modeling.encoder_decoder import SAM2_3D_prompt_after_decoder
from modeling.Image_ecoder import FpnNeck
from scipy.ndimage import distance_transform_edt as distance
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchio as tio
import torch.backends.cudnn as cudnn
from dataset.data_loader import Dataset_Union_ALL,Union_Dataloader

def init_seeds(seed=0, cuda_deterministic=True):
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def plot_loss_curve(losses, losses_val, snapshot_path):

    plt.figure(figsize=(10, 6)) 
    plt.plot(losses, label="Training Loss", color="blue", marker='o')  
    plt.plot(losses_val, label="Validation Loss", color="red", marker='s')  
    plt.title("Training & Validation Loss Curve", fontsize=16)  
    plt.xlabel("Epoch", fontsize=14)  
    plt.ylabel("Loss", fontsize=14)  
    plt.grid(True)  
    plt.legend(fontsize=12)  

    save_path = os.path.join(snapshot_path, "loss_curve.png")
    plt.savefig(save_path)  
    plt.close()  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="brain", type=str, choices=["brain"]
    )
    parser.add_argument(
        "--snapshot_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        nargs='+',
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )
    parser.add_argument(
        "--resume1", 
        type=str,
        default=None,
        help="Resuming training from checkpoint"
    )
    parser.add_argument(
    "--data_name",
    default=None,
    type=str,
    )
    parser.add_argument(
    "--ckpt", 
    type=str,
    default=None,
    help=""
    )
    parser.add_argument(
        "--config", 
        type=str,
        default="sam2_hiera_t.yaml",
        help=""
    )
    parser.add_argument("-bs", "--batch_size", default=4, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--max_epoch", default=300, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=8, type=int)
    parser.add_argument("-tolerance", default=5, type=int)
    parser.add_argument("--local_rank", type=int, help="For distributed training")
    parser.add_argument("--pretrain_weight", type=str, default=None,help="our pretrain_weight")

    args = parser.parse_args()

    # 初始化分布式训练环境
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
    
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    # 根据设备设置
    device = torch.device(f'cuda:{args.local_rank}')

    if args.rand_crop_size == 0:
        if args.data in ["brain"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    

    npz_paths_by_category = {apath: [] for apath in args.data_prefix}

    npz_paths = []
    npz_paths_by_category = defaultdict(list)
    for apath in args.data_prefix:
        files = [os.path.join(apath, f) for f in os.listdir(apath) if os.path.isfile(os.path.join(apath, f))]
        selected_files = random.sample(files, min(len(files), 100000))  # 关键修改点
        npz_paths_by_category[apath].extend(selected_files)
        npz_paths.extend(selected_files)
    train_files = []
    val_files = []
    # 对每个类别进行分层抽样
    for category, files in npz_paths_by_category.items():
        random.shuffle(files)
        split_idx = int(len(files) * 1)
        train_files.extend(files[:split_idx])

    print(f"Total files: {sum(len(v) for v in npz_paths_by_category.values())}")
    print(f"Training set: {len(train_files)}")

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    args.snapshot_path = os.path.join(args.snapshot_path, args.data_name + run_id)


    train_data=Dataset_Union_ALL(paths=train_files,
                                 transform=tio.Compose([ 
                                    tio.ToCanonical(),
                                    tio.ZNormalization(),
                                    tio.CropOrPad(mask_name='label', target_shape=args.rand_crop_size),
                                    tio.RandomFlip(axes=(0, 1, 2)),
                                    tio.RandomGamma(log_gamma=(0.9, 1.1), p=0.3),
                                    tio.RandomBiasField(coefficients=0.5, p=0.3),
                                ]), 
                                threshold=0)




    if args.rank == 0:
        if not os.path.exists(args.snapshot_path):
            os.makedirs(args.snapshot_path)
        setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
        logger = logging.getLogger(f"train")
        logger.info(str(args))
    else:
        logger = logging.getLogger(f"train")

    train_sampler = DistributedSampler(dataset=train_data,shuffle=True)
    train_data = Union_Dataloader(
    train_data,
    sampler=train_sampler,
    batch_size=args.batch_size, 
    # shuffle=False,
    num_workers=args.num_worker,
    pin_memory=True,
    drop_last=True
    )

    trunk = Hiera(
    embed_dim=96,
    num_heads=2,
    in_chans=3,
    stages=[1,2,7,2],
    global_att_blocks=[5,7,9],
    )
    neck = FpnNeck(
    d_model=256,
    backbone_channel_list=[768, 384, 192, 96],
    fpn_top_down_levels=[2, 3],
    fpn_interp_model="nearest"
    )

    mask_decoder=new_UNETR_Decoder_combine_auto_prompt(
        in_channels=1,
        out_channels=2,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        )

    mask_decoder.to(device)
    prompt_encoder_1=PromptEncoder3D(
    embed_dim=768,
    transformer=TwoWayTransformer(depth=2,
        embedding_dim=768,
        mlp_dim=2048,
        num_heads=8),
    image_embedding_size=(4,4,4),
    input_image_size=args.rand_crop_size,
    mask_in_chans=16,
    )


    model=SAM2_3D_prompt_after_decoder(trunk=trunk, 
    neck=neck, 
    prompt_encoder=prompt_encoder_1,
    mask_decoder=mask_decoder,
    batch_size=args.batch_size, 
    patch_size=args.rand_crop_size[0],
    device=device,
    multi_click=5
    )


    checkpoint = torch.load(args.pretrain_weight, map_location="cpu")

    state_dict = checkpoint["encoder_dict"]
    missing_keys, unexpected_keys=model.load_state_dict(state_dict, strict=False)
    print("=== Missing keys ===")
    for k in missing_keys:
        print(k)

    print("=== Unexpected keys ===")
    for k in unexpected_keys:
        print(k)


    model = model.to(device)
    
        
    # 设置参数可训练状态
    for p in model.img_encoder.parameters():
        p.requires_grad = False
    
    for p in model.img_encoder.trunk.patch_embed.parameters():
        p.requires_grad = True
    for i in model.img_encoder.trunk.blocks:  
        for p in i.norm1.parameters():
            p.requires_grad = True
        for p in i.adapter.parameters():
            p.requires_grad = True
        for p in i.norm2.parameters():
            p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = sum(p.numel() for p in model.parameters() )
    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Number of all parameters: {params}")
    dice_loss = DiceLoss(include_background=False, sigmoid=True,softmax=False, to_onehot_y=False, reduction="none")

    
    if args.resume1 is not None:
        checkpoint = torch.load(args.resume1, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)


   
    encoder_opt = AdamW([i for i in model.module.img_encoder.parameters() if i.requires_grad],lr=args.lr, weight_decay=0 )
    encoder_scheduler = torch.optim.lr_scheduler.LinearLR(encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=300)
    feature_opt = AdamW([i for i in model.module.prompt_encoder.parameters()if i.requires_grad == True],lr=args.lr, weight_decay=0)
    feature_scheduler = torch.optim.lr_scheduler.LinearLR(feature_opt, start_factor=1.0, end_factor=0.01, total_iters=300)
    decoder_opt = AdamW([i for i in model.module.mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=300)

    start_epoch = 0
    best_loss = np.inf

    if args.resume1 is not None:
        
        encoder_opt.load_state_dict(checkpoint['encoder_opt'])
        decoder_opt.load_state_dict(checkpoint['decoder_opt'])
        feature_opt.load_state_dict(checkpoint['feature_opt'])
        encoder_scheduler.load_state_dict(checkpoint['encoder_scheduler'])
        feature_scheduler.load_state_dict(checkpoint['feature_scheduler'])
        decoder_scheduler.load_state_dict(checkpoint['decoder_scheduler'])
        start_epoch = checkpoint['epoch']
        for _ in range(start_epoch):
            encoder_scheduler.step()
            feature_scheduler.step()
            decoder_scheduler.step()

        
        best_loss = checkpoint['best_val_loss']

        model.train()




    losses = []
    losses_val=[]

    counter=0
    for epoch_num in range(start_epoch,args.max_epoch):
        train_sampler.set_epoch(epoch_num) 
        loss_summary = []
        model.train()
        train_total_time=0
        for idx, data3D in enumerate(tqdm(train_data)):
            img,seg=data3D["image"], data3D["label"]
            dataset_ids = data3D["adapter_id"] 
            img = img.to(device)
            seg = seg.to(device)
            model=model.to(device)
            num_clicks = np.random.randint(1,6)

            masks,img_loss=model(img,seg,num_clicks,dataset_ids,epoch_num)


            dice_loss1 = dice_loss(masks, seg)
            dice = 1 - dice_loss1
            loss=img_loss
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            feature_opt.zero_grad()

            loss.backward()
  
            if num_clicks>0:
                loss=loss/num_clicks/2
            loss_summary.append(loss.detach().cpu().numpy())
            if args.rank == 0:
                logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data)) +
                    ": loss:" + str(torch.mean(torch.tensor(loss_summary[-1].flatten())).item()) +
                    ", dice:" + str(torch.mean(torch.tensor(dice)).item())+",num_clicks:"+str(num_clicks)
                )
            torch.nn.utils.clip_grad_norm_(model.module.img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.module.mask_decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.module.prompt_encoder.parameters(), 1.0)

            encoder_opt.step()
            feature_opt.step()
            decoder_opt.step()



        encoder_scheduler.step()
        feature_scheduler.step()
        decoder_scheduler.step()


        if len(loss_summary) > 0:
            avg_loss = np.mean(loss_summary).item()
            avg_loss_tensor = torch.tensor(avg_loss).to(device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / args.world_size
            losses.append(avg_loss)
        if args.rank == 0:
            logger.info("- Train metrics: " + str(avg_loss))

        if args.rank == 0:
            if (epoch_num + 1) % 2 == 0 or (epoch_num + 1) == args.max_epoch:
                plot_loss_curve(losses, losses_val,args.snapshot_path)
            

            is_best = False
            if avg_loss < best_loss:
                best_loss = avg_loss
                is_best = True
                counter=0
            else:
                counter+=1
                
            checkpoint = {
                "epoch": epoch_num + 1,
                "best_val_loss": best_loss,
                "model_state_dict": model.module.state_dict(),
                "prompt_encoder":model.module.prompt_encoder.state_dict(),
                "encoder_opt": encoder_opt.state_dict(),
                "decoder_opt": decoder_opt.state_dict(),
                "feature_opt": feature_opt.state_dict(),
                "encoder_scheduler": encoder_scheduler.state_dict(),
                "feature_scheduler": feature_scheduler.state_dict(),
                "decoder_scheduler": decoder_scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.snapshot_path, "last.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.snapshot_path, "best.pth"))

 

if __name__ == "__main__":
    main()


   

