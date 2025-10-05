import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from modeling.edge import ShapeStream3D,EdgeSupervisionLoss3D
from modeling.Image_ecoder_hiera import ImageEncoder,FpnNeck
from modeling.prompt_encoder import PromptEncoder, TwoWayTransformer
from modeling.edge_enhancer import Edge_Enhancer
import numpy as np
import edt
from monai.losses import DiceCELoss, DiceLoss,DiceFocalLoss
def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def get_3d_gaussian_kernel(radius=3, sigma=1.0, device="cpu"):
    """
    生成一个 3D 高斯核，shape 为 (2r+1, 2r+1, 2r+1)
    """
    size = 2 * radius + 1
    ax = torch.arange(-radius, radius + 1, device=device)
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.max()  # 归一化到 [0, 1]
    return kernel*5

def refine_mask_with_clicks_gaussian(masks, points_input, labels_input, radius=3, sigma=1.0, threshold=0.5):
    """
    使用 3D 高斯核对点击点周围区域进行启发式平滑修正

    Args:
        masks: Tensor [B, 1, D, H, W], float32 in [0,1]
        points_input: Tensor [B, N, 3], int (z, y, x)
        labels_input: Tensor [B, N], int (0=neg, 1=pos)
        radius: int, 高斯核半径
        sigma: float, 高斯标准差
        threshold: float, mask 二值化阈值

    Returns:
        refined_mask: Tensor [B, 1, D, H, W], binary
    """
    B, _, D, H, W = masks.shape
    device = masks.device
    refined = masks.clone()

    kernel = get_3d_gaussian_kernel(radius=radius, sigma=sigma, device=device)

    ksz = 2 * radius + 1

    for b in range(B):
        for i in range(points_input.shape[1]):
            z, y, x = points_input[b, i]
            label = labels_input[b, i]

            z_start = max(z - radius, 0)
            z_end   = min(z + radius + 1, D)
            y_start = max(y - radius, 0)
            y_end   = min(y + radius + 1, H)
            x_start = max(x - radius, 0)
            x_end   = min(x + radius + 1, W)

            # 对应 kernel 的有效区域（截断边缘）
            kz_start = radius - (z - z_start)
            ky_start = radius - (y - y_start)
            kx_start = radius - (x - x_start)

            kz_end = kz_start + (z_end - z_start)
            ky_end = ky_start + (y_end - y_start)
            kx_end = kx_start + (x_end - x_start)

            patch = kernel[kz_start:kz_end, ky_start:ky_end, kx_start:kx_end]

            if label == 1:
                refined[b, 0, z_start:z_end, y_start:y_end, x_start:x_end] += patch
            else:
                refined[b, 0, z_start:z_end, y_start:y_end, x_start:x_end] -= patch

    # 限制到 [0,1]
    refined = torch.clamp(refined, 0.0, 1.0)
    # 二值化
    refined_binary = (refined > threshold).float()
    diff = (refined_binary != (masks > threshold)).sum()
    return refined_binary



class SAM2_3D_prompt_after_decoder(nn.Module):
    def __init__(self, trunk,neck, mask_decoder,prompt_encoder,batch_size,device,multi_click=True, patch_size=256):
        super(SAM2_3D_prompt_after_decoder, self).__init__()

        self.img_encoder =ImageEncoder(trunk=trunk, neck=neck)
        self.prompt_encoder=prompt_encoder
        self.mask_decoder = mask_decoder 
        self.device=device
        self.multi_click=multi_click
        self.patch_size = patch_size
        # 获取骨干网络通道配置
        backbone_channels = self.img_encoder.neck.backbone_channel_list
        self.loss_boundary = nn.MSELoss()
        self.boundary_kernel_size=5
        self.pooling_layer = nn.AvgPool3d((self.boundary_kernel_size, self.boundary_kernel_size, 1), stride=1,
                                     padding=(int((self.boundary_kernel_size - 1) / 2),
                                              int((self.boundary_kernel_size - 1) / 2),
                                              0))
    
    def _get_next_click3D_torch_2(self,prev_seg, gt_semantic_seg):

        mask_threshold = 0.5

        batch_points = []
        batch_labels = []
        # dice_list = []

        pred_masks = prev_seg > mask_threshold
        true_masks = gt_semantic_seg > 0
        fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
        fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

        to_point_mask = torch.logical_or(fn_masks, fp_masks)

        for i in range(pred_masks.shape[0]):

            points = torch.argwhere(to_point_mask[i])
            if len(points)>0:
                point = points[np.random.randint(len(points))]
                # import pdb; pdb.set_trace()
                if fn_masks[i, 0, point[1], point[2], point[3]]:
                    is_positive = True
                else:
                    is_positive = False
            

                bp = point[1:].clone().detach().reshape(1, 1, 3)
                bl = torch.tensor(
                    [
                        int(is_positive),
                    ]
                ).reshape(1, 1)
                batch_points.append(bp)
                batch_labels.append(bl)
            else:
                point = torch.Tensor(
                    [np.random.randint(sz) for sz in pred_masks[i, 0].size()]
                ).to(torch.int64)
                spatial_coords = point[-3:] 
                is_positive = pred_masks[i, 0, spatial_coords[0], spatial_coords[1], spatial_coords[2]]

                bp = spatial_coords.clone().detach().reshape(1, 1, 3)
                bl = torch.tensor([int(is_positive)], dtype=torch.long).reshape(1, 1)
                batch_points.append(bp)
                batch_labels.append(bl)

        return batch_points, batch_labels  # , (sum(dice_list)/len(dice_list)).item()
    
    def _get_next_click3D_torch_ritm(self, prev_seg, gt_semantic_seg):

        mask_threshold = 0.5
        batch_points = []
        batch_labels = []

        pred_masks = prev_seg > mask_threshold  # shape: [B, 1, D, H, W]
        true_masks = gt_semantic_seg > 0        # shape: [B, 1, D, H, W]

        for i in range(gt_semantic_seg.shape[0]):  # iterate over batch
            pred_mask = pred_masks[i, 0]
            true_mask = true_masks[i, 0]

            fn_mask = torch.logical_and(true_mask, torch.logical_not(pred_mask))
            fp_mask = torch.logical_and(torch.logical_not(true_mask), pred_mask)

            # Pad and convert to uint8
            fn_mask_padded = F.pad(fn_mask, (1, 1, 1, 1, 1, 1), "constant", value=0).to(torch.uint8)
            fp_mask_padded = F.pad(fp_mask, (1, 1, 1, 1, 1, 1), "constant", value=0).to(torch.uint8)

            # EDT distance transform (remove padding)
            fn_dt = torch.tensor(
                edt.edt(fn_mask_padded.cpu().numpy(), black_border=True, parallel=4)
            )[1:-1, 1:-1, 1:-1]
            fp_dt = torch.tensor(
                edt.edt(fp_mask_padded.cpu().numpy(), black_border=True, parallel=4)
            )[1:-1, 1:-1, 1:-1]

            fn_max_dist = fn_dt.max()
            fp_max_dist = fp_dt.max()

            is_positive = fn_max_dist > fp_max_dist
            dt = fn_dt if is_positive else fp_dt

            to_point_mask = dt > (max(fn_max_dist, fp_max_dist) / 2.0)
            points = torch.argwhere(to_point_mask)

            if len(points) == 0:
                # fallback: center point
                D, H, W = dt.shape
                point = torch.tensor([D // 2, H // 2, W // 2])
            else:
                point = points[np.random.randint(len(points))]

            bp = point.reshape(1, 1, 3)  # shape [1, 1, 3]
            bl = torch.tensor([int(is_positive)]).reshape(1, 1)

            batch_points.append(bp)
            batch_labels.append(bl)

        return batch_points, batch_labels
    
    def _get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = self._get_next_click3D_torch_2(prev_masks, gt3D)
        # batch_points, batch_labels = self._get_next_click3D_torch_ritm(prev_masks, gt3D)
        batch_points = [p.to(self.device, non_blocking=True) for p in batch_points]
        batch_labels = [l.to(self.device, non_blocking=True) for l in batch_labels]
        points_co = torch.cat(batch_points, dim=0).to(self.device)
        points_la = torch.cat(batch_labels, dim=0).to(self.device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(self.device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(self.device)

        if self.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

        
    
    def forward(self, img,seg,num_clicks,dataset_ids,epoch):
        loss_cal_auto= DiceFocalLoss(
        include_background=False,
        sigmoid=False,
        softmax=True,
        to_onehot_y=True,
        lambda_dice=0.5,
        lambda_focal=0.5,
        gamma=3.0,
        reduction="mean",     
    )   
        loss_cal = DiceFocalLoss(
            include_background=True,   
            sigmoid=True,             
            softmax=False,             
            to_onehot_y=False,        
            lambda_dice=0.5,
            lambda_focal=0.5,
            gamma=3.0,
            reduction="mean"
        )

        out = F.interpolate(img.float(), scale_factor=512 / self.patch_size, mode='trilinear')
        input_batch = out
        batch_features, feature_list,sample= self.img_encoder(input_batch)
        prev_masks = torch.zeros_like(seg).to(seg.device)
        self.click_points = []
        self.click_labels = []
        epoch_loss=0
        logits=None
        masks_to_prompt=None
        for num_click in range(num_clicks):
            prev_masks=prev_masks.to(seg.device)
            points_input, labels_input = self._get_points(prev_masks, seg)
            new_feature = []
            for i, feature in enumerate(sample):
                new_feature.append(feature)
                if i==3:
                    _,point_embedding=self.prompt_encoder(image_embeddings=feature, points=[points_input, labels_input],boxes=None,masks=masks_to_prompt, img_size=list(feature.shape[2:]))
            new_feature.append(img[:, 0:1])
            if logits is None:
                masks_prompt,masks_auto, logits = self.mask_decoder(new_feature, point_embedding, logits)
                prev_masks = torch.sigmoid(masks_prompt)
                masks_to_prompt=prev_masks
                loss = loss_cal(masks_prompt, seg)+loss_cal_auto(masks_auto ,seg)*num_clicks
                epoch_loss+=loss
            else:
                prev_masks=(prev_masks>0.5).float()
                prev_masks=None
                masks_prompt,logits = self.mask_decoder(new_feature, point_embedding, logits,prev_masks)
                prev_masks = torch.sigmoid(masks_prompt)
                masks_to_prompt=prev_masks
                loss = loss_cal(masks_prompt, seg)
                epoch_loss += loss
        return masks_prompt ,epoch_loss   


class SAM2_3D_prompt_after_decoder_with_auto_test(nn.Module):
    def __init__(self, trunk,neck, mask_decoder,prompt_encoder,batch_size,device,multi_click=True, patch_size=256):
        super(SAM2_3D_prompt_after_decoder_with_auto_test, self).__init__()

        self.img_encoder =ImageEncoder(trunk=trunk, neck=neck)
        self.prompt_encoder=prompt_encoder
        self.mask_decoder = mask_decoder 
        self.device=device
        self.multi_click=multi_click
        self.patch_size = patch_size
        # 获取骨干网络通道配置
        backbone_channels = self.img_encoder.neck.backbone_channel_list
    
    def _get_next_click3D_torch_2(self,prev_seg, gt_semantic_seg):

        mask_threshold = 0.5

        batch_points = []
        batch_labels = []
        # dice_list = []

        pred_masks = prev_seg > mask_threshold
        true_masks = gt_semantic_seg > 0
        fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
        fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

        to_point_mask = torch.logical_or(fn_masks, fp_masks)
        for i in range(pred_masks.shape[0]):

            points = torch.argwhere(to_point_mask[i])
            if len(points)>0:
                point = points[np.random.randint(len(points))]
                # import pdb; pdb.set_trace()
                if fn_masks[i, 0, point[1], point[2], point[3]]:
                    is_positive = True
                else:
                    is_positive = False
            

                bp = point[1:].clone().detach().reshape(1, 1, 3)
                bl = torch.tensor(
                    [
                        int(is_positive),
                    ]
                ).reshape(1, 1)
                batch_points.append(bp)
                batch_labels.append(bl)
            else:
                point = torch.Tensor(
                    [np.random.randint(sz) for sz in pred_masks[i, 0].size()]
                ).to(torch.int64)
                spatial_coords = point[-3:] 
                is_positive = pred_masks[i, 0, spatial_coords[0], spatial_coords[1], spatial_coords[2]]

                bp = spatial_coords.clone().detach().reshape(1, 1, 3)
                bl = torch.tensor([int(is_positive)], dtype=torch.long).reshape(1, 1)
                batch_points.append(bp)
                batch_labels.append(bl)

        return batch_points, batch_labels  # , (sum(dice_list)/len(dice_list)).item()
    
    def _get_next_click3D_torch_ritm(self, prev_seg, gt_semantic_seg):

        mask_threshold = 0.5
        batch_points = []
        batch_labels = []

        pred_masks = prev_seg > mask_threshold  # shape: [B, 1, D, H, W]
        true_masks = gt_semantic_seg > 0        # shape: [B, 1, D, H, W]

        for i in range(gt_semantic_seg.shape[0]):  # iterate over batch
            pred_mask = pred_masks[i, 0]
            true_mask = true_masks[i, 0]

            fn_mask = torch.logical_and(true_mask, torch.logical_not(pred_mask))
            fp_mask = torch.logical_and(torch.logical_not(true_mask), pred_mask)

            # Pad and convert to uint8
            fn_mask_padded = F.pad(fn_mask, (1, 1, 1, 1, 1, 1), "constant", value=0).to(torch.uint8)
            fp_mask_padded = F.pad(fp_mask, (1, 1, 1, 1, 1, 1), "constant", value=0).to(torch.uint8)

            # EDT distance transform (remove padding)
            fn_dt = torch.tensor(
                edt.edt(fn_mask_padded.cpu().numpy(), black_border=True, parallel=4)
            )[1:-1, 1:-1, 1:-1]
            fp_dt = torch.tensor(
                edt.edt(fp_mask_padded.cpu().numpy(), black_border=True, parallel=4)
            )[1:-1, 1:-1, 1:-1]

            fn_max_dist = fn_dt.max()
            fp_max_dist = fp_dt.max()

            is_positive = fn_max_dist > fp_max_dist
            dt = fn_dt if is_positive else fp_dt

            to_point_mask = dt > (max(fn_max_dist, fp_max_dist) / 2.0)
            points = torch.argwhere(to_point_mask)

            if len(points) == 0:
                # fallback: center point
                D, H, W = dt.shape
                point = torch.tensor([D // 2, H // 2, W // 2])
            else:
                point = points[np.random.randint(len(points))]

            bp = point.reshape(1, 1, 3)  # shape [1, 1, 3]
            bl = torch.tensor([int(is_positive)]).reshape(1, 1)

            batch_points.append(bp)
            batch_labels.append(bl)

        return batch_points, batch_labels
    
    def _get_points(self, prev_masks, gt3D):

        #batch_points, batch_labels = self._get_next_click3D_torch_2(prev_masks, gt3D)
        batch_points, batch_labels = self._get_next_click3D_torch_ritm(prev_masks, gt3D)
        batch_points = [p.to(self.device, non_blocking=True) for p in batch_points]
        batch_labels = [l.to(self.device, non_blocking=True) for l in batch_labels]
        points_co = torch.cat(batch_points, dim=0).to(self.device)
        points_la = torch.cat(batch_labels, dim=0).to(self.device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(self.device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(self.device)

        if self.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

        
    
    def forward(self, img,seg,num_clicks):
        loss_cal_auto = DiceFocalLoss(
        include_background=False,
        sigmoid=False,
        softmax=True,
        to_onehot_y=True,
        lambda_dice=0.4,
        lambda_focal=0.6,
        gamma=3.0,
        reduction="mean"
    )   
        loss_cal = DiceFocalLoss(
            include_background=True,   # 二分类通常需要包含背景通道
            sigmoid=True,              # ✅ 二分类，用 sigmoid
            softmax=False,             # ❌ 不能开 softmax
            to_onehot_y=False,         # ❌ 不需要 one-hot，如果 label 是 [B,1,H,W,D] 或 [B,H,W,D]
            lambda_dice=0.4,
            lambda_focal=0.6,
            gamma=3.0,
            reduction="mean"
        )
        
        loss_fn = DiceLoss(sigmoid=False, to_onehot_y=False, squared_pred=False, reduction='mean')
        dice_loss = DiceLoss(include_background=True, softmax=False,sigmoid=False, to_onehot_y=False, reduction="none")
        dice_loss_auto=DiceLoss(include_background=False, softmax=True,sigmoid=False, to_onehot_y=True, reduction="none")
    #     loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
        # 图像预处理：调整大小
        out = F.interpolate(img.float(), scale_factor=512 / self.patch_size, mode='trilinear')
        
        # input_batch = (out.cuda() - pixel_mean) / pixel_std
        input_batch = out
        # input_batch=img
        batch_features, feature_list,sample= self.img_encoder(input_batch)
        prev_masks = torch.zeros_like(seg).to(seg.device)
        # prev_masks=prev_masks.unsqueeze(1)
        # print(prev_masks.shape)
        # low_res_masks = F.interpolate(prev_masks.float(), size=(256,256,256))
        self.click_points = []
        self.click_labels = []
        epoch_loss=0
        B,_,_,_,_=out.shape
        # masks_to_prompt = torch.zeros((B, 768, 4, 4, 4), 
        #                       device=seg.device, 
        #                       dtype=seg.dtype)
        if num_clicks==0:
            img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1), scale_factor=64/self.patch_size,mode='trilinear')
            new_feature = []
            for i, feature in enumerate(feature_list):
                new_feature.append(feature)
            new_feature.append(img_resize)
            masks = self.mask_decoder(new_feature, 2, self.patch_size // 64)
            masks = masks.permute(0, 1, 4, 2, 3)
            epoch_loss = loss_cal(masks, seg)
        else:
            prompt_pred_list = []
            prompt_dice_list = []
            postpre_pred_list=[]
            postpre_dice_list=[]
            auto=False
            logits=None
            masks_to_prompt=None
            for num_click in range(num_clicks):
                points_input, labels_input = self._get_points(prev_masks, seg)
                new_feature = []
                for i, feature in enumerate(sample):
                    new_feature.append(feature)
                    if i==3:
                        _,point_embedding=self.prompt_encoder(image_embeddings=feature, points=[points_input, labels_input],boxes=None,masks=masks_to_prompt, img_size=list(feature.shape[2:]))
                # img_resize = F.interpolate(img[:, 0:1], scale_factor=64 / self.patch_size,mode='trilinear',align_corners=False)
                # new_feature.append(img_resize)\
                new_feature.append(img[:, 0:1])
                # masks=self.mask_decoder(new_feature,point_embedding)

                if logits is None:
                    masks_prompt,masks_auto,logits = self.mask_decoder(new_feature, point_embedding,logits)
                    # masks_to_prompt=masks_prompt
                    masks_auto_test=torch.softmax(masks_auto, dim=1)
                    masks_auto_test = masks_auto_test[:, 1:]
                    prompt_pred_list.append((masks_auto_test > 0.5).float())
                    dice=1-loss_fn((masks_auto_test > 0.5).float(),seg).item()
                    prompt_dice_list.append(round(dice, 4))
                    # masks_auto = F.interpolate(masks_auto.float(), scale_factor=self.patch_size /128, mode='trilinear')
                    # masks_prompt = F.interpolate(masks_prompt.float(), scale_factor=self.patch_size /128, mode='trilinear')
                    # dice=1-dice_loss_auto(masks_auto,seg)
                    prev_masks = torch.sigmoid(masks_prompt)
                    # masks_to_prompt =self.mask_decoder.mask_encode(prev_masks)
                    prompt_pred_list.append((prev_masks>0.5).float())
                    dice=1-loss_fn((prev_masks>0.5).float(),seg).item()
                    prompt_dice_list.append(round(dice, 4))
                    postpre_pred_list.append((prev_masks>0.5).float())
                    postpre_dice_list.append(round(dice, 4))
                    refined_mask = refine_mask_with_clicks_gaussian(
                        prev_masks,
                        points_input,
                        labels_input,
                        radius=10,
                        sigma=1.2,   # 越大越平滑
                        threshold=0.5
                    )
                    # prev_masks=refined_mask
                    refine_dice=1-loss_fn(refined_mask,seg).item()
                    postpre_pred_list.append(refined_mask)
                    postpre_dice_list.append(round(refine_dice, 4))
                else:
                    prev_masks=None
                    masks_prompt ,logits= self.mask_decoder(new_feature, point_embedding, logits,prev_masks)
                    # masks_to_prompt=masks_prompt
                    prev_masks = torch.sigmoid(masks_prompt)
                    # masks_to_prompt =self.mask_decoder.mask_encode(prev_masks)
                    prompt_pred_list.append((prev_masks>0.5).float())
                    dice=1-loss_fn((prev_masks>0.5).float(),seg).item()
                    prompt_dice_list.append(round(dice, 4))
                    postpre_pred_list.append((prev_masks>0.5).float())
                    postpre_dice_list.append(round(dice, 4))
                    refined_mask = refine_mask_with_clicks_gaussian(
                        prev_masks,
                        points_input,
                        labels_input,
                        radius=10,
                        sigma=1.2,   # 越大越平滑
                        threshold=0.5
                    )
                    # prev_masks=refined_mask
                    refine_dice=1-loss_fn(refined_mask,seg).item()
                    postpre_pred_list.append(refined_mask)
                    postpre_dice_list.append(round(refine_dice, 4))

                    
            print(prompt_dice_list)

            prompt_max_dice = max(prompt_dice_list)
            prompt_max_idx = prompt_dice_list.index(prompt_max_dice)
            print(f"有提示没后处理最大 Dice 值: {prompt_max_dice}，对应索引: {prompt_max_idx }")
            # 取出对应的预测 mask
            prompt_masks = prompt_pred_list[prompt_max_idx] 

            print(postpre_dice_list)
            postpre_max_dice = max(postpre_dice_list)
            postpre_max_idx = postpre_dice_list.index(postpre_max_dice)
            print(f"后处理最大 Dice 值: {postpre_max_dice}，对应索引: {postpre_max_idx }")
            # 取出对应的预测 mask
            postpre_masks = postpre_pred_list[postpre_max_idx] 
        return masks_auto,prompt_masks,postpre_masks
