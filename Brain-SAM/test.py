
import argparse
import numpy as np
import logging
from medpy.metric.binary import hd95
import torch.nn.functional as F
from modeling.Mask_decoder import new_UNETR_Decoder_combine_auto_prompt
from modeling.encoder_decoder import SAM2_3D_prompt_after_decoder_with_auto_test
from modeling.prompt_encoder_after_decoder import PromptEncoder3D , TwoWayTransformer
import torch
import os
import pandas as pd
from datetime import datetime
from utils.util import setup_logger
import surface_distance
from surface_distance import metrics
import csv
from modeling.Hiera_encoder_v2 import  Hiera
from modeling.Image_ecoder_hiera import FpnNeck
import SimpleITK as sitk
import torchio as tio
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
from torch.utils.data.distributed import DistributedSampler
from dataset.data_loader import Dataset_Union_ALL_Val

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



def remove_module_prefix(state_dict):

    return {k.replace("module.", ""): v for k, v in state_dict.items()}
def get_confusion_matrix(pred_mask, gt_mask):

    pred_bin = pred_mask.long()
    gt_bin = gt_mask.long()

    TP = ((pred_bin == 1) & (gt_bin == 1)).sum().item()
    FP = ((pred_bin == 1) & (gt_bin == 0)).sum().item()
    TN = ((pred_bin == 0) & (gt_bin == 0)).sum().item()
    FN = ((pred_bin == 0) & (gt_bin == 1)).sum().item()

    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}
def iou_3d(pred, target, threshold=0.6, epsilon=1e-6):

        if isinstance(pred, torch.Tensor) is False:
            pred = pred.tensor 


        pred = torch.sigmoid(pred) 
        pred = (pred > threshold).float()  
        
        target = target.float()  

        # 计算交集和并集
        intersection = torch.sum(pred * target) 
        union = torch.sum(pred) + torch.sum(target) - intersection  

        # 计算 IoU
        iou = (intersection + epsilon) / (union + epsilon)  

        return iou


def get_dice_score( prev_masks, gt3D):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)
            
            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2*volume_intersect / volume_sum
    
        pred_masks = (prev_masks > 0.1)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list)/len(dice_list))

def get_hd95_score(pred_masks, gt3D):
    """
    Compute HD95 for each 3D volume (pred vs. ground truth), then average.
    """

    pred_masks = (pred_masks > 0.5).cpu().numpy().astype(np.uint8)
    gt_masks = (gt3D > 0).cpu().numpy().astype(np.uint8)
    
    hd95_list = []
    for i in range(gt_masks.shape[0]):
        pred = pred_masks[i]
        gt = gt_masks[i]
        
        if np.sum(pred) == 0 and np.sum(gt) == 0:

            hd95_list.append(0.0)
        elif np.sum(pred) == 0 or np.sum(gt) == 0:

            hd95_list.append(np.nan)
        else:
            try:
                hd = hd95(pred, gt)
                hd95_list.append(hd)
            except Exception as e:
                print(f"Error at case {i}: {e}")
                hd95_list.append(np.nan)
    
    # 过滤掉 nan 后取平均
    hd95_list = [v for v in hd95_list if not np.isnan(v)]
    return np.mean(hd95_list) if hd95_list else np.nan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="brain", type=str, choices=["kits", "pancreas", "lits", "colon","brain"]
    )
    parser.add_argument(
        "--snapshot_path",
        default="/data/pyhData/SAM2/work_dir/prompt_DDP_epilepsy_20250422-1735",
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
        nargs='+',
        default="",
        type=str,
    )
    parser.add_argument(
        "--data_name",
        default="",
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
    )
    parser.add_argument(
        "--num_prompts",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--in_chans",
        default=3,
        type=int,
    )
    parser.add_argument("-bs", "--batch_size", default=2, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=4, type=int)
    parser.add_argument("--save",default=False,type=bool)
    parser.add_argument("--local_rank", type=int, help="For distributed training")
    parser.add_argument(
        "--checkpoint",
        default="last",
        type=str,
    )
    parser.add_argument("-tolerance", default=5, type=int)
    args = parser.parse_args()
    if args.checkpoint == "last":
        file = "last.pth"
    elif args.checkpoint=="best":
        file = "best.pth"
    else:
        filename = os.path.basename(args.checkpoint)
        file  = filename
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:

        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.distributed = True
    else:

        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        torch.cuda.set_device(0)
        args.distributed = False

    device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")


    if args.rand_crop_size == 0:
        if args.data in ["colon", "pancreas", "lits", "kits","brain"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    


    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    pred_save_dir=os.path.join(args.snapshot_path, "result",args.data_name+run_id)
    os.makedirs(pred_save_dir, exist_ok=True)
    if args.rank == 0:  
        os.makedirs(pred_save_dir, exist_ok=True)
        setup_logger(logger_name="test", root=args.snapshot_path, screen=True, tofile=True)
        logger = logging.getLogger(f"test")
        logger.info(str(args))
    else:
        logger = logging.getLogger(f"test")

    npz_paths = []
    for apath in args.data_prefix:
        npz_paths.extend([os.path.join(apath, f) for f in os.listdir(apath) if os.path.isfile(os.path.join(apath, f))])

    infer_transform = [
    tio.ToCanonical(),
    tio.ZNormalization(), 
    tio.CropOrPad(mask_name='label', target_shape=args.rand_crop_size),
    ]


    val_data = Dataset_Union_ALL_Val(
        paths=npz_paths, 
        mode="Val", 
        transform=tio.Compose(infer_transform),
        threshold=0,
        pcc=False,
    )
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        val_sampler = DistributedSampler(val_data, shuffle=False)
    else:
        val_sampler=None
    test_data = torch.utils.data.DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        shuffle=(val_sampler is None),  
        num_workers=args.num_worker, 
        drop_last=False,
        sampler=val_sampler,
        pin_memory=True
    )

    trunk = Hiera(
    embed_dim=96,
    num_heads=2,
    in_chans=args.in_chans,
    stages=[1,2,7,2],
    global_att_blocks=[5,7,9],
    )
    neck = FpnNeck(
    d_model=256,
    backbone_channel_list=[768, 384, 192, 96],
    fpn_top_down_levels=[2, 3],
    fpn_interp_model="nearest"
    )

    mask_decoder=new_UNETR_Decoder_combine_auto_prompt(in_channels=1,
        out_channels=2,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,

        norm_name="instance")

    mask_decoder.to(device)
    prompt_encoder_1=PromptEncoder3D(
    embed_dim=768,
    transformer=TwoWayTransformer(depth=2,
        embedding_dim=768,
        mlp_dim=2048,
        num_heads=8),
    image_embedding_size=(4,4, 4),
    input_image_size=args.rand_crop_size,
    mask_in_chans=16,
    )

    model=SAM2_3D_prompt_after_decoder_with_auto_test(trunk=trunk, 
    neck=neck, 
    prompt_encoder=prompt_encoder_1,
    # prompt_encoder_2=prompt_encoder_2,
    mask_decoder=mask_decoder,
    batch_size=args.batch_size, 
    patch_size=args.rand_crop_size[0],
    device=device,
    multi_click=0
    )



    checkpoint_path = os.path.join(args.snapshot_path, file)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    epoch=checkpoint["epoch"]
    model=model.to(device)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    


    
    model.eval()
    with torch.no_grad():
        loss_summary = []
        loss_nsd = []
        results=[]
        model=model.to(device)

        for batch_idx, batch_data in enumerate(tqdm(test_data)):
            img, seg, spacing,img_name = batch_data["image"],batch_data["label"],batch_data["spacing"],batch_data["path"][0]         
            print(f"gts sum: {seg.sum()}")
            seg = seg.to(device)

            
            img=img.to(device)

            masks_auto,prompt_masks,postpre_masks= model(img,seg,6)

            masks_auto=torch.softmax(masks_auto, dim=1)
            masks_auto = masks_auto[:, 1:]
            masks_auto_bin = masks_auto > 0.5
            prompt_masks_bin = prompt_masks > 0.5
            postpre_masks_bin = postpre_masks > 0.5
            prompt_cm = get_confusion_matrix(prompt_masks_bin, seg)
            prompt_TP, prompt_FP, prompt_TN, prompt_FN = prompt_cm["TP"], prompt_cm["FP"], prompt_cm["TN"], prompt_cm["FN"]


            auto_cm = get_confusion_matrix(masks_auto_bin, seg)
            auto_TP, auto_FP, auto_TN, auto_FN = auto_cm["TP"], auto_cm["FP"], auto_cm["TN"], auto_cm["FN"]


            auto_dice = get_dice_score(masks_auto_bin, seg).item()
            prompt_dice = get_dice_score(prompt_masks_bin, seg).item()
            postpre_dice = get_dice_score(postpre_masks_bin, seg).item()

            auto_iou = iou_3d(masks_auto_bin, seg).item()
            prompt_iou = iou_3d(prompt_masks_bin, seg).item()
            postpre_iou = iou_3d(postpre_masks_bin, seg).item()

            auto_hd95 = get_hd95_score(masks_auto_bin, seg)
            prompt_hd95 = get_hd95_score(prompt_masks_bin, seg)
            postpre_hd95 = get_hd95_score(postpre_masks_bin, seg)

            seg_np = (seg == 1)[0,0].cpu().numpy()
            auto_np = masks_auto_bin[0,0].cpu().numpy()
            prompt_np = prompt_masks_bin[0,0].cpu().numpy()
            postpre_np = postpre_masks_bin[0,0].cpu().numpy()

            spacing_np = spacing[0].numpy()

            ssd_auto = surface_distance.compute_surface_distances(seg_np, auto_np, spacing_np)
            ssd_prompt = surface_distance.compute_surface_distances(seg_np, prompt_np, spacing_np)
            ssd_postpre = surface_distance.compute_surface_distances(seg_np, postpre_np, spacing_np)

            auto_nsd = metrics.compute_surface_dice_at_tolerance(ssd_auto, args.tolerance)
            prompt_nsd = metrics.compute_surface_dice_at_tolerance(ssd_prompt, args.tolerance)
            postpre_nsd = metrics.compute_surface_dice_at_tolerance(ssd_postpre, args.tolerance)
            results.append({
                "File": img_name,
                "Gt_Sum":seg.sum().item(),
                "Auto_Dice": auto_dice,
                "Prompt_Dice": prompt_dice,
                "Postpre_Dice": postpre_dice,
                "Auto_IoU": auto_iou,
                "Prompt_IoU": prompt_iou,
                "Postpre_IoU": postpre_iou,
                "Auto_HD95": auto_hd95,
                "Prompt_HD95": prompt_hd95,
                "Postpre_HD95": postpre_hd95,
                "Auto_NSD": auto_nsd,
                "Prompt_NSD": prompt_nsd,
                "Postpre_NSD": postpre_nsd,
                "Auto_TP": auto_TP,
                "Auto_FP": auto_FP,
                "Auto_TN": auto_TN,
                "Auto_FN": auto_FN,
                "Prompt_TP": prompt_TP,
                "Prompt_FP": prompt_FP,
                "Prompt_TN": prompt_TN,
                "Prompt_FN": prompt_FN
            })
            logger.info(
                "Case {} | gts.sum() {:.4f} | Auto: Dice {:.4f}, IoU {:.4f}, HD95 {:.4f}, NSD {:.4f} || "
                "Prompt: Dice {:.4f}, IoU {:.4f}, HD95 {:.4f}, NSD {:.4f}, TP {}, FP {}, TN {}, FN {} || "
                "Postpre: Dice {:.4f}, IoU {:.4f}, HD95 {:.4f}, NSD {:.4f}".format(
                    img_name, seg.sum().item(),
                    auto_dice, auto_iou, auto_hd95, auto_nsd,
                    prompt_dice, prompt_iou, prompt_hd95, prompt_nsd,
                    prompt_TP, prompt_FP, prompt_TN, prompt_FN,
                    postpre_dice, postpre_iou, postpre_hd95, postpre_nsd
                )
            )
            if args.save:
                    name=os.path.basename(img_name)
                    target_size = img.shape[2:] 
                    img_np = img.cpu().numpy()  
                    img_sitk = sitk.GetImageFromArray(img_np) 
                    Spacing=img_sitk.GetSpacing()

                    sitk.WriteImage(img_sitk, os.path.join(pred_save_dir, name.replace('.npz', '.nii.gz')))

                    single_mask=F.interpolate(prompt_masks_bin.float(), size=target_size, mode="trilinear", align_corners=False)
                    masks_np = single_mask.cpu().numpy()  
                    masks_np = masks_np.astype(np.uint8)
                    masks_sitk = sitk.GetImageFromArray(masks_np)  
                    masks_sitk.SetSpacing(Spacing)  
                    sitk.WriteImage(masks_sitk, os.path.join(pred_save_dir, name.replace('.npz', '_masks.nii.gz')))

                    # 3. 将 seg (Tensor) 转换为 NumPy 数组并保存
                    single_seg=F.interpolate(seg.float(), size=target_size, mode="trilinear", align_corners=False)
                    seg_np = single_seg.cpu().numpy()  
                    seg_np = seg_np.astype(np.uint8)
                    seg_sitk = sitk.GetImageFromArray(seg_np)  
                    seg_sitk.SetSpacing(Spacing)  
                    sitk.WriteImage(seg_sitk, os.path.join(pred_save_dir, name.replace('.npz', '_gt.nii.gz')))

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            tmp_file = os.path.join(pred_save_dir, f"tmp_result_rank{args.rank}.csv")


            if args.rank != -1:
                fieldnames = [
                    "File", "Gt_Sum",
                    "Auto_Dice", "Prompt_Dice", "Postpre_Dice",
                    "Auto_HD95", "Prompt_HD95", "Postpre_HD95",
                    "Auto_NSD", "Prompt_NSD", "Postpre_NSD",
                    "Auto_IoU", "Prompt_IoU", "Postpre_IoU",
                    "Auto_TP", "Auto_FP", "Auto_TN", "Auto_FN",
                    "Prompt_TP", "Prompt_FP", "Prompt_TN", "Prompt_FN"
                ]
                with open(tmp_file, mode="w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)

                dist.barrier()


            if args.rank == 0:
                all_results = []
                for r in range(args.world_size):
                    fpath = os.path.join(pred_save_dir, f"tmp_result_rank{r}.csv")
                    df_r = pd.read_csv(fpath)
                    all_results.extend(df_r.to_dict(orient="records"))


                for r in range(args.world_size):
                    os.remove(os.path.join(pred_save_dir, f"tmp_result_rank{r}.csv"))


                final_csv_path = os.path.join(pred_save_dir, "loss_results.csv")
                df_final = pd.DataFrame(all_results)
                df_final.to_csv(final_csv_path, index=False)

                def mean_metric(name): return df_final[name].mean()

                logger.info("Auto   - Dice: {:.4f} | HD95: {:.4f} | NSD: {:.4f} | IoU: {:.4f}".format(
                    mean_metric("Auto_Dice"), mean_metric("Auto_HD95"),
                    mean_metric("Auto_NSD"), mean_metric("Auto_IoU")))
                logger.info("Prompt - Dice: {:.4f} | HD95: {:.4f} | NSD: {:.4f} | IoU: {:.4f}".format(
                    mean_metric("Prompt_Dice"), mean_metric("Prompt_HD95"),
                    mean_metric("Prompt_NSD"), mean_metric("Prompt_IoU")))
                logger.info("Postpr - Dice: {:.4f} | HD95: {:.4f} | NSD: {:.4f} | IoU: {:.4f}".format(
                    mean_metric("Postpre_Dice"), mean_metric("Postpre_HD95"),
                    mean_metric("Postpre_NSD"), mean_metric("Postpre_IoU")))

            dist.barrier()
        else:
            final_csv_path = os.path.join(pred_save_dir, "loss_results.csv")
            fieldnames = [
                    "File", "Gt_Sum",
                    "Auto_Dice", "Prompt_Dice", "Postpre_Dice",
                    "Auto_HD95", "Prompt_HD95", "Postpre_HD95",
                    "Auto_NSD", "Prompt_NSD", "Postpre_NSD",
                    "Auto_IoU", "Prompt_IoU", "Postpre_IoU",
                    "Auto_TP", "Auto_FP", "Auto_TN", "Auto_FN",
                    "Prompt_TP", "Prompt_FP", "Prompt_TN", "Prompt_FN"
                ]

            with open(final_csv_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)  # `results` 为 List[Dict[str, float]]


            df = pd.read_csv(final_csv_path)

            def mean_metric(name: str) -> float:
                return df[name].mean()

            logger.info(
                "Auto   - Dice: {:.4f} | HD95: {:.4f} | NSD: {:.4f} | IoU: {:.4f}".format(
                    mean_metric("Auto_Dice"), mean_metric("Auto_HD95"),
                    mean_metric("Auto_NSD"), mean_metric("Auto_IoU"))
            )
            logger.info(
                "Prompt - Dice: {:.4f} | HD95: {:.4f} | NSD: {:.4f} | IoU: {:.4f}".format(
                    mean_metric("Prompt_Dice"), mean_metric("Prompt_HD95"),
                    mean_metric("Prompt_NSD"), mean_metric("Prompt_IoU"))
            )
            logger.info(
                "Postpr - Dice: {:.4f} | HD95: {:.4f} | NSD: {:.4f} | IoU: {:.4f}".format(
                    mean_metric("Postpre_Dice"), mean_metric("Postpre_HD95"),
                    mean_metric("Postpre_NSD"), mean_metric("Postpre_IoU"))
            )

if __name__ == "__main__":
    main()

