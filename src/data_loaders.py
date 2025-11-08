import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
from glob import glob
import numpy as np

from monai.transforms import (
    Compose, 
    ResizeD,
    NormalizeIntensityd, 
    RandFlipd,
    RandAffined,
    ToTensord,
)

from monai.data import Dataset as MonaiDataset

class BraTSDataset(Dataset):
    '''
    This class is to load and convert .nii files to tensors
    '''
    def __init__(self, root, is_train=True):
        self.is_train = is_train
        self.cases = sorted([dir for dir in glob(os.path.join(root, "*")) if os.path.isdir(dir)])

    def __len__(self):
        return len(self.cases)
    
    def load_niifti(self, path):
        return nib.load(path).get_fdata().astype(np.float32)

    def __getitem__(self, idx):
        case = self.cases[idx]

        t1 = self.load_niifti(glob(os.path.join(case, "*-t1n.nii.gz"))[0])
        t1ce = self.load_niifti(glob(os.path.join(case, "*-t1c.nii.gz"))[0])
        t2 = self.load_niifti(glob(os.path.join(case, "*-t2w.nii.gz"))[0])
        flair = self.load_niifti(glob(os.path.join(case, "*-t2f.nii.gz"))[0])

        image = np.stack([t1, t1ce, t2, flair])
        image = torch.tensor(image)

        sample = {"image" : image}

        if self.is_train:
            seg = self.load_niifti(glob(os.path.join(case, "*-seg.nii.gz"))[0])
            sample["mask"] = torch.tensor(seg).unsqueeze(0)

        return sample

def get_dataloaders(root, batch_size=1, num_workers=4, val_split=0.2):

    full_ds = BraTSDataset(root, is_train=True)

    total_size = len(full_ds)
    train_count = int((1-val_split) * total_size)
    val_count = total_size - train_count

    train_ds, val_ds = random_split(full_ds, [train_count, val_count])

    # Transformations & loaders
    train_transform = Compose([
        ResizeD(keys=["image"], spatial_size=(128, 128, 160), mode="trilinear"),
        ResizeD(keys=["mask"], spatial_size=(128, 128, 160), mode="nearest"),
        NormalizeIntensityd(keys=["image"]),
        RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
        RandAffined(
            keys=["image", "mask"], 
            prob=0.3, 
            rotate_range=(0, 0, 0.2),
            mode=["trilinear", "nearest"]),
        
        ToTensord(keys=["image", "mask"]),
    ])

    val_transform = Compose([
        ResizeD(keys=["image"], spatial_size=(128, 128, 160), mode="trilinear", align_corners=False),
        ResizeD(keys=["mask"], spatial_size=(128, 128, 160), mode="nearest"),
        NormalizeIntensityd(keys=["image"]),

        ToTensord(keys=["image"]),
    ])

    train_ds = MonaiDataset(data=train_ds, transform=train_transform)
    val_ds = MonaiDataset(data=val_ds, transform=val_transform)
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader