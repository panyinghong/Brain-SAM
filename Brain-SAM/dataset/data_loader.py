from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchio as tio
from torchio.data.io import sitk_to_nib
import torch
import numpy as np
import os
import time
import torch
import SimpleITK as sitk
from prefetch_generator import BackgroundGenerator
from torchio import SubjectsDataset, Subject
from skimage.segmentation import find_boundaries
import random

class Dataset_Union_ALL(Dataset):
    def __init__(
        self,
        paths,
        mode="train",
        data_type="Tr",
        image_size=128,
        transform=None,
        threshold=0,
        split_num=1,
        split_idx=0,
        pcc=False,
        get_all_meta_info=False,
    ):
        # super().__init__(subjects)
        self.threshold = threshold
        self.mode = mode
        self.transform=transform
        self.paths = paths
        self.data_type = data_type
        self.image_size = image_size
        self.pcc = pcc
        self.get_all_meta_info = get_all_meta_info
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        npz_path=self.paths[index]
        with np.load(npz_path) as npz_data:
            if "epilepsy" in  npz_path:
                adapter_id=0
            else:
                adapter_id=1
            img = npz_data['imgs'].astype(np.float32)
            seg = npz_data['gts'].astype(np.float32)
            img = np.expand_dims(img, axis=0).repeat(3, axis=0)  # (3, D, H, W)
            seg = np.expand_dims(seg, axis=0)                    # (1, D, H, W)
            spacing = npz_data["spacing"]

            if img.shape[1] != min(img.shape[1:]):
                img = np.transpose(img, (0, 3, 1, 2))
                seg = np.transpose(seg, (0, 3, 1, 2))
                spacing = spacing[[2, 0, 1]] 

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img),
            label=tio.LabelMap(tensor=seg),
            spacing=spacing,
            path=npz_path,
        )
        if self.pcc:
            # 添加 crop_mask
            random_index = torch.argwhere(subject.label.data == 1)
            if len(random_index) >= 1:
                random_index = random_index[np.random.randint(0, len(random_index))]
                crop_mask = torch.zeros_like(subject.label.data)
                crop_mask[random_index[0], random_index[1], random_index[2], random_index[3]] = 1
                subject.add_image(tio.LabelMap(tensor=crop_mask), image_name="crop_mask")
                subject = tio.CropOrPad(mask_name="crop_mask", target_shape=(self.image_size,) * 3)(subject)

        if self.transform:
            try:
                torch.manual_seed(torch.randint(0, 1000000, (1,)).item())
                np.random.seed(np.random.randint(0, 1000000))
                random.seed(random.randint(0, 1000000))
                subject = self.transform(subject)

            except:
                print(self.paths[index])


        if self.mode == "train" and self.data_type == "Tr":
            return {
                "image": subject.image.data.clone().detach(),
                "label": subject.label.data.clone().detach(),
                "adapter_id": torch.tensor(adapter_id, dtype=torch.long)  #
            }
        if self.mode!= "train":
            spacing_tensor = torch.tensor(subject.spacing, dtype=torch.float32)  # 转为 tensor
            return {
                "image": subject.image.data.clone().detach(),
                "label": subject.label.data.clone().detach(),
                "spacing": spacing_tensor,
                "path":subject.path,
                "adapter_id": torch.tensor(adapter_id, dtype=torch.long),  #

            }


class Dataset_Union_ALL_Val(Dataset_Union_ALL):
    def _set_file_paths(self, paths):
        self.image_paths = []
        self.label_paths = []

        # if ${path}/labelsTr exists, search all .nii.gz
        for path in paths:
            for dt in ["Tr", "Val", "Ts"]:
                d = os.path.join(path, f"labels{dt}")
                if os.path.exists(d):
                    for name in os.listdir(d):
                        base = os.path.basename(name).split(".nii.gz")[0]
                        label_path = os.path.join(path, f"labels{dt}", f"{base}.nii.gz")
                        self.image_paths.append(label_path.replace("labels", "images"))
                        self.label_paths.append(label_path)
        self.image_paths = self.image_paths[self.split_idx :: self.split_num]
        self.label_paths = self.label_paths[self.split_idx :: self.split_num]







