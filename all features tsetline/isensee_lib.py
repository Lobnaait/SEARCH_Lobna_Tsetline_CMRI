"""Isensee et al. 2017 (DKFZ): 2D + 3D U-Net ensemble. Re-implementation in PyTorch.

Follows the public Lasagne code (github.com/MIC-DKFZ/ACDC2017): resampling to 1.25x1.25 mm
in-plane (3D net: also 10 mm through-plane), per-volume z-scoring, U-Net with 4 poolings,
BN + LeakyReLU, nearest-neighbour upsampling + concatenation, dropout, deep supervision.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from common.data import TRAIN_IDS  # noqa: E402

TARGET_INPLANE = 1.25
TARGET_Z = 10.0
METHOD_DIR = os.path.dirname(os.path.abspath(__file__))


def get_split(fold):
    """Same 5-fold split as the original code: KFold(5, shuffle=True, random_state=12345)."""
    from sklearn.model_selection import KFold
    ids = np.array(TRAIN_IDS)
    tr, va = list(KFold(5, shuffle=True, random_state=12345).split(ids))[fold]
    return list(ids[tr]), list(ids[va])


def target_spacing(spacing, dim):
    return np.array([spacing[0] if dim == 2 else TARGET_Z, TARGET_INPLANE, TARGET_INPLANE], np.float32)


def new_shape(shape, spacing, tgt):
    return tuple(int(np.round(s * sp / t)) for s, sp, t in zip(shape, spacing, tgt))


def resample_image(img, spacing, dim):
    tgt = target_spacing(spacing, dim)
    order = 3 if dim == 2 else 1  # original: order 3 when z kept, else 1
    out = resize(img.astype(np.float64), new_shape(img.shape, spacing, tgt), order=order, mode='edge',
                 anti_aliasing=False)
    return out.astype(np.float32)


def resample_seg(seg, spacing, dim, n_classes=4):
    tgt = target_spacing(spacing, dim)
    shp = new_shape(seg.shape, spacing, tgt)
    probs = np.stack([resize((seg == c).astype(np.float64), shp, order=1, mode='edge', anti_aliasing=False)
                      for c in range(n_classes)])
    return probs.argmax(0).astype(np.uint8)


def zscore(img):
    return (img - img.mean()) / (img.std() + 1e-8)


# ----------------------------------------------------------------------------- network
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dim):
        super().__init__()
        Conv, BN = (nn.Conv2d, nn.BatchNorm2d) if dim == 2 else (nn.Conv3d, nn.BatchNorm3d)
        self.net = nn.Sequential(Conv(cin, cout, 3, padding=1), BN(cout), nn.LeakyReLU(0.01, inplace=True),
                                 Conv(cout, cout, 3, padding=1), BN(cout), nn.LeakyReLU(0.01, inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """U-Net with 4 poolings, dropout after pool2-4 and concat1-3, deep supervision (Isensee et al.)."""

    def __init__(self, dim=2, in_ch=1, n_classes=4, base=48, dropout=0.3):
        super().__init__()
        self.dim = dim
        self.pool_k = 2 if dim == 2 else (1, 2, 2)
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        self.drop = (nn.Dropout2d if dim == 2 else nn.Dropout3d)(dropout) if dropout > 0 else nn.Identity()
        f = [base * 2 ** i for i in range(5)]
        self.enc = nn.ModuleList([ConvBlock(in_ch, f[0], dim)] + [ConvBlock(f[i - 1], f[i], dim) for i in range(1, 5)])
        self.dec = nn.ModuleList([ConvBlock(f[i + 1] + f[i], f[i], dim) for i in reversed(range(4))])
        self.ds2 = Conv(f[2], n_classes, 1)  # 1/4 resolution
        self.ds1 = Conv(f[1], n_classes, 1)  # 1/2 resolution
        self.out = Conv(f[0], n_classes, 1)

    def up(self, x):
        return F.interpolate(x, scale_factor=(2, 2) if self.dim == 2 else (1, 2, 2), mode='nearest')

    def pool(self, x):
        return F.max_pool2d(x, 2) if self.dim == 2 else F.max_pool3d(x, self.pool_k)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            if i > 0:
                x = self.pool(x)
                if i >= 2:
                    x = self.drop(x)
            x = blk(x)
            skips.append(x)
        x = skips[-1]
        ds = []
        for j, blk in enumerate(self.dec):  # j=0: deepest
            x = torch.cat([self.up(x), skips[3 - j]], 1)
            if j < 3:
                x = self.drop(x)
            x = blk(x)
            if j == 1:
                ds.append(self.ds2(x))
            elif j == 2:
                ds.append(self.ds1(x))
        logits = self.out(x) + self.up(self.up(ds[0]) + ds[1])
        return logits


def soft_dice_loss(logits, target, n_classes=4, eps=1e-6):
    """-mean over classes of soft Dice computed over the whole batch (all 4 classes)."""
    p = torch.softmax(logits.float(), 1)
    oh = F.one_hot(target, n_classes).movedim(-1, 1).float()
    dims = [0] + list(range(2, p.ndim))
    inter = (p * oh).sum(dims)
    den = p.sum(dims) + oh.sum(dims)
    return -(2 * inter / (den + eps)).mean()


def build_net(dim):
    return UNet(dim=dim, base=48 if dim == 2 else 26, dropout=0.3 if dim == 2 else 0.5)


def ckpt_path(dim, fold):
    return os.path.join(METHOD_DIR, 'checkpoints', 'unet%dd_fold%d.pt' % (dim, fold))
