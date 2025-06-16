# src/internal_flow/data.py
# ------------------------------------------------------------
#  Efficient dataset – v4 (06 / 06 / 2025)
# ------------------------------------------------------------
#  * banque de 15 000 masques free-form (≈ 3,8 Go, CPU → GPU as-is)
#  * seed décalé par worker -> pas de doublons
#  * mean/std pré-alloués, zéro alloc dans la boucle
#  * bruit limité à la zone masquée et option sigma conservée
#  * gère INDÉFECTIBLEMENT 1 seule image        (files = [""])
#    ou 4 textures “diffuse/normal/roughness/specular”
#  * même signature : chaque sample = (corrupt, clean, mask)
# ------------------------------------------------------------

from __future__ import annotations
import math, random
from itertools import islice
from os.path import join
from typing import Iterable, Sequence, Tuple, List

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import IterableDataset, get_worker_info

import os


# ------------------------------------------------------------
#  utilitaires de base
# ------------------------------------------------------------
def _mean_std(arr: np.ndarray):
    m = arr.mean(axis=(0, 1)).astype(np.float32)
    s = arr.std(axis=(0, 1)).astype(np.float32)
    s[s < 1e-6] = 1.0
    return m, s


def _mask_box(alpha: np.ndarray):
    ys, xs = np.nonzero(alpha)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _enlarge(box: Sequence[int], img_sz: Tuple[int, int], tgt_sz: Tuple[int, int] | int):
    x1, y1, x2, y2 = box
    iw, ih = img_sz
    tw, th = (tgt_sz, tgt_sz) if isinstance(tgt_sz, int) else tgt_sz
    tw, th = min(tw, iw), min(th, ih)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    def _place(c, length, max_val):
        l = int(round(c - length / 2))
        r = l + int(length)
        if l < 0:
            r -= l
            l = 0
        if r > max_val:
            l -= r - max_val
            r = max_val
        return l, r

    nx1, nx2 = _place(cx, tw, iw)
    ny1, ny2 = _place(cy, th, ih)
    return nx1, ny1, nx2 - 1, ny2 - 1  # inclusive


def _mask_generator(H: int, W: int) -> Iterable[np.ndarray]:
    """Génère indéfiniment des masques libres (1,H,W) float32 binaire."""
    rng = np.random.default_rng()
    min_nv, max_nv = 4, 12
    mean_ang, ang_rng = 2 * math.pi / 5, 2 * math.pi / 15
    min_w, max_w = 12, 40
    avg_r = math.hypot(H, W) / 8

    while True:
        img = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(img)
        for _ in range(rng.integers(1, 4)):
            nv = rng.integers(min_nv, max_nv)
            a_min = mean_ang - rng.uniform(0, ang_rng)
            a_max = mean_ang + rng.uniform(0, ang_rng)
            angles = [
                (2 * math.pi - rng.uniform(a_min, a_max)) if i % 2 == 0 else rng.uniform(a_min, a_max)
                for i in range(nv)
            ]
            verts = [(rng.integers(0, W), rng.integers(0, H))]
            for a in angles:
                r = np.clip(rng.normal(avg_r, avg_r / 2), 0, 2 * avg_r)
                lx, ly = verts[-1]
                nx = np.clip(lx + r * math.cos(a), 0, W - 1)
                ny = np.clip(ly + r * math.sin(a), 0, H - 1)
                verts.append((int(nx), int(ny)))
            width = int(rng.uniform(min_w, max_w))
            draw.line(verts, fill=1, width=width)
            for vx, vy in verts:
                draw.ellipse((vx - width // 2, vy - width // 2,
                              vx + width // 2, vy + width // 2),
                             fill=1)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        yield np.asarray(img, dtype=np.float32)[None]  # (1,H,W)


def _apply_corruption(clean: torch.Tensor,
                      mask: torch.Tensor,
                      sigma: float | None) -> torch.Tensor:
    """Add noise only inside the masked region, keep σ-variance if requested."""
    noise = torch.randn_like(clean) * mask
    corrupt = clean * (1.0 - mask) + noise
    if sigma is not None:
        eps = torch.randn_like(clean) * mask
        corrupt = corrupt * (1 - sigma) ** 0.5 + eps * sigma ** 0.5
    return corrupt

# ------------------------------------------------------------
#  Dataset de base (commune train / test)
# ------------------------------------------------------------
class _BaseDataset(IterableDataset):
    # Par défaut : 4 textures matérielles.
    # Les classes **Override** ci-dessous écrasent avec [""] pour 1 seule image.
    files: List[str] = ["diffuse.png", "normal.png", "roughness.png", "specular.png"]

    def __init__(
        self,
        image_path: str,
        mask_path: str | None = None,
        region_size: Tuple[int, int] = (256, 256),
        mask_bank_size: int = 15_000,
        seed: int | None = None,
        sigma: float | None = None,
        use_bank: bool = True
    ) -> None:
        super().__init__()
        from os.path import isfile, isdir

        self.region_w, self.region_h = region_size
        self.sigma = sigma

        # Seed for base RNG (also passed to workers later)
        self.base_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)
        torch.manual_seed(self.base_seed)

        # ---------------------------------------------
        # Load all images (either file or folder)
        # ---------------------------------------------
        import os
        if self.files and self.files[0] == "":
            if isdir(image_path):
                file_paths = sorted([
                    join(image_path, f)
                    for f in os.listdir(image_path)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
            elif isfile(image_path):
                file_paths = [image_path]
            else:
                raise ValueError(f"Invalid image_path: {image_path}")
        else:
            file_paths = [join(image_path, fn) for fn in self.files]

        img_tensors, mean_l, std_l = [], [], []
        for fp in file_paths:
            arr = np.asarray(Image.open(fp).convert("RGB"), dtype=np.float32) / 255.0
            m, s = _mean_std(arr)
            mean_l.append(m)
            std_l.append(s)
            img_tensors.append(torch.from_numpy(arr.transpose(2, 0, 1)))  # (3,H,W)

        self.imgs = torch.stack(img_tensors)   # (n_tex,3,H,W)
        self.n_tex, _, self.img_h, self.img_w = self.imgs.shape

        self.mean_t = torch.tensor(np.stack(mean_l), dtype=torch.float32).view(self.n_tex, 3, 1, 1)
        self.std_t  = torch.tensor(np.stack(std_l ), dtype=torch.float32).view(self.n_tex, 3, 1, 1)

        # ---------------------------------------------
        # Compute region_box
        # ---------------------------------------------
        if mask_path and isfile(mask_path):
            alpha = np.asarray(Image.open(mask_path))[..., 3]
            self.mask_box = _mask_box(alpha)
            self.region_box = _enlarge(self.mask_box, (self.img_w, self.img_h), (self.region_w, self.region_h))
        else:
            self.mask_box = (0, 0, 0, 0)
            self.region_box = (0, 0, self.region_w, self.region_h)

        # ---------------------------------------------
        # Generate mask bank
        # ---------------------------------------------
        if use_bank:
            masks_np = np.stack(list(islice(_mask_generator(*region_size), mask_bank_size)))
            self.mask_bank = torch.from_numpy(masks_np).float()  # (N,1,H,W)
            self.mask_bank_size = mask_bank_size

    # --------------------------------------------------------
    #  helpers
    # --------------------------------------------------------
    def _seed_worker(self, worker_id: int):
        seed = self.base_seed + worker_id
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    def _crop_norm(self, x1: int, y1: int) -> torch.Tensor:
        x2, y2 = x1 + self.region_w, y1 + self.region_h
        crop = self.imgs[:, :, y1:y2, x1:x2].clone()          # (n_tex,3,h,w)
        crop.sub_(self.mean_t).div_(self.std_t)
        return crop.view(-1, self.region_h, self.region_w)   # (n_tex*3,h,w)

    def _apply_corruption(self, clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Bruit uniquement dans la zone masquée
        noise = torch.randn_like(clean) * mask
        corrupt = clean * (1.0 - mask) + noise
        if self.sigma is not None:
            eps = torch.randn_like(clean) * mask
            corrupt = corrupt * (1 - self.sigma) ** 0.5 + eps * self.sigma ** 0.5
        return corrupt

    # --------------------------------------------------------
    #  Inverser normalisation (facultatif)
    # --------------------------------------------------------
    def to_natural(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, H, W) où C = n_tex*3."""
        b, c, h, w = x.shape
        x = x.view(b, self.n_tex, 3, h, w).clone()
        mean = self.mean_t.to(x.device)
        std  = self.std_t.to(x.device)
        x.mul_(std.unsqueeze(0)).add_(mean.unsqueeze(0))
        return x.view(b, c, h, w).clamp_(0, 1)


# ------------------------------------------------------------
#  Dataset entraînement
# ------------------------------------------------------------
class SingleImageTrainDataset(_BaseDataset):
    size = 50_000  # 1 "epoch" arbitraire

    def __iter__(self):
        info = get_worker_info()
        if info is not None:
            self._seed_worker(info.id)
        else:
            # Fallback for single-process case
            self._seed_worker(0)
        x1m, y1m, x2m, y2m = self.region_box   # boîte à éviter
        iw, ih, rw, rh = self.img_w, self.img_h, self.region_w, self.region_h
        mb, mb_size = self.mask_bank, self.mask_bank_size

        while True:
            # --- sample region hors de la boîte masque globale
            while True:
                x1 = random.randint(0, iw - rw)
                y1 = random.randint(0, ih - rh)
                x2, y2 = x1 + rw, y1 + rh
                if not (x1 <= x2m and x2 >= x1m and y1 <= y2m and y2 >= y1m):
                    break
            clean  = self._crop_norm(x1, y1)
            mask   = mb[random.randrange(mb_size)].clone()
            corrupt = self._apply_corruption(clean, mask)
            yield corrupt, clean, mask

    def __len__(self):
        return self.size


# ------------------------------------------------------------
#  Dataset test : un seul échantillon fixe
# ------------------------------------------------------------
class SingleImageTestDataset(_BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, use_bank=False)
        # région centrée sur la boîte masque originelle
        self.region_box = _enlarge(self.mask_box,
                                   (self.img_w, self.img_h),
                                   (self.region_w, self.region_h))
        x1, y1, *_ = self.region_box
        self.clean_ref  = self._crop_norm(x1, y1)
        alpha = np.asarray(Image.open(kwargs["mask_path"]))[..., 3]
        mask_np = (alpha[y1:y1+self.region_h, x1:x1+self.region_w] > 0).astype(np.float32)
        self.mask_ref   = torch.from_numpy(mask_np)[None]
        self.corrupt_ref = self._apply_corruption(self.clean_ref, self.mask_ref)

    def __iter__(self):
        yield self.corrupt_ref.clone(), self.clean_ref.clone(), self.mask_ref.clone()

    def __len__(self):
        return 1


# ------------------------------------------------------------
#  Aliases
# ------------------------------------------------------------
class TrainOverride(SingleImageTrainDataset):
    files: List[str] = [""]      # 1 seule image passée par image_path


class TestOverride(SingleImageTestDataset):
    files: List[str] = [""]


# ------------------------------------------------------------
#  Base class for variable-size, map-style datasets
# ------------------------------------------------------------
class _MultiImageBase(torch.utils.data.Dataset):
    """Map-style dataset indexing every possible patch in every image,
       with global dataset normalization.
    """

    def __init__(
        self,
        image_path: str,                 # folder with .png/.jpg
        region_size: Tuple[int, int] = (256, 256),
        mask_bank_size: int = 15_000,
        seed: int | None = None,
        sigma: float | None = None,
    ):
        super().__init__()
        from os.path import isdir
        assert isdir(image_path), f"{image_path} is not a directory"

        self.region_w, self.region_h = region_size
        self.sigma = sigma

        # Global RNG for reproducible mask choice
        self.base_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)
        torch.manual_seed(self.base_seed)

        # Load image paths
        img_files = sorted(
            f for f in os.listdir(image_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not img_files:
            raise RuntimeError(f"No images found in {image_path}")

        # Prepare lists
        self.imgs: list[torch.Tensor] = []
        self.counts: list[int] = []

        # Accumulate for global normalization
        sums = np.zeros(3, dtype=np.float64)
        sumsq = np.zeros(3, dtype=np.float64)
        npix_total = 0

        rw, rh = self.region_w, self.region_h
        for fn in img_files:
            arr = np.asarray(Image.open(join(image_path, fn)).convert("RGB"),
                             dtype=np.float32) / 255.0
            h, w = arr.shape[:2]
            if w <= rw or h <= rh:
                continue

            # Accumulate global stats
            pixels = arr.reshape(-1, 3)
            sums += pixels.sum(axis=0)
            sumsq += (pixels ** 2).sum(axis=0)
            npix_total += pixels.shape[0]

            self.imgs.append(torch.from_numpy(arr.transpose(2, 0, 1)))  # (3,H,W)
            self.counts.append((w - rw) * (h - rh))

        if not self.imgs:
            raise RuntimeError("No image large enough to contain one patch")

        # Compute dataset-wide mean and std
        mean = sums / npix_total
        std = np.sqrt(sumsq / npix_total - mean ** 2)
        self.mean = torch.from_numpy(mean).view(3, 1, 1).float()
        self.std  = torch.from_numpy(std).view(3, 1, 1).float()

        print(f"[Dataset] Global mean: {self.mean.flatten().tolist()}")
        print(f"[Dataset] Global std : {self.std.flatten().tolist()}")

        self.cumsums = np.cumsum(self.counts)
        self.total_patches = int(self.cumsums[-1])

        # Mask bank
        masks_np = np.stack(list(islice(_mask_generator(*region_size), mask_bank_size)))
        self.mask_bank = torch.from_numpy(masks_np).float()  # (N,1,H,W)
        self.mask_bank_size = mask_bank_size

    def __len__(self) -> int:
        return self.total_patches

    def __getitem__(self, idx: int):
        i = int(np.searchsorted(self.cumsums, idx, side="right"))
        start = 0 if i == 0 else self.cumsums[i - 1]
        k = idx - start

        img = self.imgs[i]
        _, H, W = img.shape
        rw, rh = self.region_w, self.region_h
        patches_per_row = W - rw

        y_off = k // patches_per_row
        x_off = k %  patches_per_row

        clean = img[:, y_off:y_off + rh, x_off:x_off + rw].clone()
        clean.sub_(self.mean).div_(self.std)

        mask_idx = (self.base_seed + idx) % self.mask_bank_size
        mask = self.mask_bank[mask_idx].clone()
        corrupt = _apply_corruption(clean, mask, self.sigma)

        return corrupt, clean, mask

    def to_natural(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x.mul(std).add(mean).clamp_(0, 1)


# ------------------------------------------------------------
#  Train / test wrappers (identical behaviour here)
# ------------------------------------------------------------
class MultiImageTrainDataset(_MultiImageBase):
    """Same behaviour for training – the DataLoader's sampler/shuffle decide
    the order.  Nothing else to change."""
    pass


class MultiImageTestDataset(_MultiImageBase):
    """Identical to train: random mask choice, exhaustive patch indexing."""
    pass