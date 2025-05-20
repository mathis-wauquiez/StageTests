from os.path import join
import math
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import ToTensor, Normalize, Compose


# --------------------------------------------------------------------------- #
#  Utility helpers
# --------------------------------------------------------------------------- #


def get_mean_std(pil_image):
    """Compute per‑channel mean and std in the [0, 1] range."""
    arr = np.asarray(pil_image).astype(np.float32) / 255.0  # H,W,3
    mean = arr.mean(axis=(0, 1)).tolist()
    std = arr.std(axis=(0, 1)).tolist()
    std = [s if s > 1e-6 else 1.0 for s in std]  # avoid div/0
    return mean, std


def make_transform(pil_image):
    mean, std = get_mean_std(pil_image)
    return Compose([ToTensor(), Normalize(mean=mean, std=std)])


def get_mask_box(mask):
    """Return the (x1,y1,x2,y2) box around non‑zero mask pixels."""
    mask = np.asarray(mask)
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def enlarge_box(box, image_size, target_size):
    """Grow/shrink *box* so its final size equals *target_size*."""
    x1, y1, x2, y2 = box
    img_w, img_h = image_size

    if isinstance(target_size, int):
        tgt_w = tgt_h = target_size
    else:
        tgt_w, tgt_h = target_size

    tgt_w = min(tgt_w, img_w)+1
    tgt_h = min(tgt_h, img_h)+1

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    def _place(c, length, max_val):
        length = int(length)
        left = int(round(c - length / 2))
        right = left + length
        if left < 0:
            right -= left
            left = 0
        if right > max_val:
            left -= (right - max_val)
            right = max_val
        return left, right

    new_x1, new_x2 = _place(cx, tgt_w, img_w)
    new_y1, new_y2 = _place(cy, tgt_h, img_h)
    return new_x1, new_y1, new_x2 - 1, new_y2 - 1   # inclusive


def choose_region_uniform(image_size, mask_box, region_size, max_tries=10_000):
    """Sample a *region_size* rectangle not intersecting *mask_box*."""
    img_w, img_h = image_size
    region_w, region_h = region_size

    if region_w > img_w or region_h > img_h:
        raise ValueError("region_size bigger than image")

    x1m, y1m, x2m, y2m = mask_box

    for _ in range(max_tries):
        x1 = np.random.randint(0, img_w - region_w + 1)
        y1 = np.random.randint(0, img_h - region_h + 1)
        x2, y2 = x1 + region_w, y1 + region_h
        if not (x1 <= x2m and x2 >= x1m and y1 <= y2m and y2 >= y1m):
            return x1, y1, x2, y2

    raise RuntimeError("Unable to sample a region")


def corrupt_tensor(tensor, mask):
    """Replace pixels where mask==1 with random noise."""
    noise = torch.randn_like(tensor)
    return tensor * (1.0 - mask) + noise * mask


def mask_generator(H, W):
    """Yield random free‑form masks of shape (1,H,W)."""
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2 * math.pi / 5
    angle_range = 2 * math.pi / 15
    min_width = 12
    max_width = 40
    avg_radius = math.hypot(H, W) / 8

    while True:
        mask_img = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(mask_img)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            a_min = mean_angle - np.random.uniform(0, angle_range)
            a_max = mean_angle + np.random.uniform(0, angle_range)
            angles = [
                (2 * math.pi - np.random.uniform(a_min, a_max)) if i % 2 == 0
                else np.random.uniform(a_min, a_max)
                for i in range(num_vertex)
            ]

            verts = [(
                np.random.randint(0, W),
                np.random.randint(0, H)
            )]
            for a in angles:
                r = np.clip(np.random.normal(avg_radius, avg_radius / 2),
                            0, 2 * avg_radius)
                lx, ly = verts[-1]
                nx = np.clip(lx + r * math.cos(a), 0, W - 1)
                ny = np.clip(ly + r * math.sin(a), 0, H - 1)
                verts.append((int(nx), int(ny)))

            width = int(np.random.uniform(min_width, max_width))
            draw.line(verts, fill=1, width=width)
            for vx, vy in verts:
                draw.ellipse((vx - width // 2, vy - width // 2,
                              vx + width // 2, vy + width // 2),
                             fill=1)

        if random.random() < 0.5:
            mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            mask_img = mask_img.transpose(Image.FLIP_TOP_BOTTOM)

        mask = np.asarray(mask_img, np.float32)
        mask = np.expand_dims(mask, 0)  # (1,H,W)
        yield mask


# --------------------------------------------------------------------------- #
#  Datasets
# --------------------------------------------------------------------------- #


class SingleImageTrainDataset(IterableDataset):
    """Infinite train dataset returning (corrupt, clean, mask) tuples."""

    files = ["diffuse.png", "normal.png", "roughness.png", "specular.png"]
    size = 20000 # one epoch = k iterations

    def __init__(self, image_path, mask_path, region_size=(256, 256), seed=None, sigma=None):

        super().__init__()

        self.sigma = sigma

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.region_w, self.region_h = region_size

        # Mask box
        mask = np.array(Image.open(mask_path))[..., 3]
        self.mask_box = get_mask_box(mask)

        # Images and transforms
        self.imgs = [Image.open(join(image_path, f)).convert("RGB") for f in self.files]
        self.transforms = [make_transform(img) for img in self.imgs]

        self.mean_std = [get_mean_std(img) for img in self.imgs]
        self.mean = [mean for mean, _ in self.mean_std]
        self.std = [std for _, std in self.mean_std]

        self.img_w, self.img_h = self.imgs[0].size

        # Mask generator
        self._mask_gen = mask_generator(self.region_h, self.region_w)

    def _crop_and_transform(self, region_box, mask_region, corrupt=False):
        """
        Crop the region, apply the transforms and potentially corrupt it.
        mask_region is of the same size as the region.
        """
        # x1, y1, _, _ = region_box
        # pil_box = (x1, y1, x1 + self.region_w, y1 + self.region_h)
        transformed_crops = [tfm(img.crop(region_box)) for img, tfm in zip(self.imgs, self.transforms)]
        sample = torch.cat(transformed_crops, 0)  # (12,h,w)
        
        if corrupt:
            corr = corrupt_tensor(sample, mask_region)

            if self.sigma is not None:
                epsilon = torch.randn_like(corr)
                corr = corr * (1 - self.sigma)**.5 + epsilon * (self.sigma)**.5 # Variance-preserving

            return corr, sample
        return sample

    def __iter__(self):
        for _ in range(self.size):
            # Sample a region and a mask, then crop, transform and corrupt
            region_box = choose_region_uniform(
                (self.img_w, self.img_h), self.mask_box, (self.region_w, self.region_h)
            )
            
            mask_region = torch.from_numpy(next(self._mask_gen)).float()
            
            corrupt, clean = self._crop_and_transform(region_box, mask_region, corrupt=True)
            
            yield corrupt, clean, mask_region

    def __len__(self):
        return self.size
    
    def to_natural(self, x):
        """Inverse the normalization of the input tensor."""
        x = x.clone()
        for i in range(len(self.mean)):
            mean = torch.tensor(self.mean[i]).view(1, 3, 1, 1).to(x.device)
            std = torch.tensor(self.std[i]).view(1, 3, 1, 1).to(x.device)
            x[i] = x[i] * std + mean

        x = torch.clamp(x, 0, 1)
        return x


class SingleImageTestDataset(SingleImageTrainDataset):
    """Single‑sample dataset for evaluation."""

    def __init__(self, image_path, mask_path, region_size=(256, 256), seed=None):
        super().__init__(image_path, mask_path, region_size, seed=seed)
        

        # Load the mask and convert to grayscale (alpha channel -> grayscale)
        mask = Image.open(mask_path)
        mask = np.array(mask)[..., 3]
        mask = Image.fromarray(mask.astype(np.uint8), mode="L")

        # Get the original mask box
        # and enlarge it to the region size
        original_box = get_mask_box(mask)
        self.region_box = enlarge_box(original_box, self.imgs[0].size, region_size)

        # Crop the mask to the region box
        self.mask_region = ToTensor()(mask.crop(self.region_box)).float()
        

    def __iter__(self):
        yield self._make_sample()

    def _make_sample(self):
        corrupt, clean = self._crop_and_transform(self.region_box, self.mask_region, corrupt=True)
        return corrupt, clean, self.mask_region

    def __len__(self):
        return 1


class TestOverride(SingleImageTestDataset):
    files = [""]

class TrainOverride(SingleImageTrainDataset):
    files = [""]


# --------------------------------------------------------------------------- #
#  Convenience
# --------------------------------------------------------------------------- #


def get_datasets(image_path, mask, region_size=(256, 256), seed=None):
    test_ds = SingleImageTestDataset(image_path, mask, region_size, seed=seed)
    train_ds = SingleImageTrainDataset(image_path, test_ds.region_box, region_size, seed=seed)
    return train_ds, test_ds
