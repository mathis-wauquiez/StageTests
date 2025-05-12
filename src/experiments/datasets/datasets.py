from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch
import numpy as np

class CorruptedDataset(Dataset, ABC):
    def __init__(self, base_dataset, transform=None, target_transform=None):
        """
        Args:
            base_dataset (Dataset): Any dataset returning image or (image, label)
            transform (callable, optional): Applied to corrupted input x
            target_transform (callable, optional): Applied to clean target y
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        y = item[0] if isinstance(item, tuple) else item
        label = item[1] if isinstance(item, tuple) and len(item) > 1 else None

        x = self.corruption_fn(y)

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        if label is not None:
            return x, y, label
        else:
            return x, y

    @abstractmethod
    def corruption_fn(self, y):
        """Return x = corrupt(y)"""
        pass

class GaussianNoiseDataset(CorruptedDataset):
    def __init__(self, base_dataset, noise_std=0.1, transform=None, target_transform=None):
        super().__init__(base_dataset, transform, target_transform)
        self.noise_std = noise_std

    def corruption_fn(self, y):
        noise = torch.randn_like(y) * self.noise_std
        return noise
    


class InpaintingDataset(CorruptedDataset):
    def __init__(self, base_dataset, transform=None, target_transform=None):
        """
        Args:
            base_dataset (Dataset): Any dataset returning image or (image, label)
            transform (callable, optional): Applied to x (corrupted image)
            target_transform (callable, optional): Applied to y (clean image)
        """
        super().__init__(base_dataset, transform=transform, target_transform=target_transform)

    def corruption_fn(self, y):
        x = y.clone()
        _, H, W = x.shape
        h, w = H // 2, W // 2
        top, left = (H - h) // 2, (W - w) // 2

        # Replace central region with N(0,1) noise
        x[:, top:top+h, left:left+w] = torch.randn_like(x[:, top:top+h, left:left+w])
        return x
    


class GMM2GMM(Dataset):
    def __init__(self, means1, covs1, means2, covs2, num_samples):
        """
        Args:
            means1 (Tensor): (K1, 2) means of the first GMM  (source)
            covs1  (List[Tensor]): list of (2, 2) covariances for first GMM
            means2 (Tensor): (K2, 2) means of the second GMM (target)
            covs2  (List[Tensor]): list of (2, 2) covariances for second GMM
            num_samples (int): number of sample pairs to generate
        """
        super().__init__()
        self.means1 = means1
        self.covs1 = covs1
        self.means2 = means2
        self.covs2 = covs2
        self.num_samples = num_samples

        # -- generate and cache everything up front
        self.data_src, self.data_tgt, self.labels_tgt = self._generate_samples()

    @staticmethod
    def _sample_from_gmm(means, covs, n_samples, *, return_labels=False):
        """
        Sample from a Gaussian Mixture Model.
        If `return_labels` is True, also return the component index (LongTensor).

        Returns
        -------
        samples : (n_samples, 2) Tensor
        labels  : (n_samples,) LongTensor  (only if return_labels=True)
        """
        n_components = len(means)
        # draw a component index for every point
        comp_idx = torch.randint(0, n_components, (n_samples,))
        samples = torch.empty((n_samples, 2))

        for k in range(n_components):
            mask = comp_idx == k
            count = mask.sum().item()
            if count:
                dist = torch.distributions.MultivariateNormal(means[k], covs[k])
                samples[mask] = dist.sample((count,))

        return (samples, comp_idx.long()) if return_labels else samples

    def _generate_samples(self):
        """Generate (x_src, x_tgt, y_tgt) triplets."""
        x_src = self._sample_from_gmm(self.means1, self.covs1, self.num_samples)
        x_tgt, y_tgt = self._sample_from_gmm(
            self.means2, self.covs2, self.num_samples, return_labels=True
        )
        return x_src, x_tgt, y_tgt

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns
        -------
        x_src : (2,) Tensor         – a point from the source GMM
        x_tgt : (2,) Tensor         – a point from the target GMM
        y_tgt : ()   LongTensor int – component index of the target point
        """
        return (
            self.data_src[idx],
            self.data_tgt[idx],
            self.labels_tgt[idx],
        )

def get_first_example_dataset(num_samples):
    # Parameters
    l = 1.5           # vertical offset
    std_x = 0.15      # standard deviation along x
    std_y = 0.05      # standard deviation along y

    # Utility: create rotated anisotropic covariance
    def make_anisotropic_cov(angle_deg, std_x, std_y):
        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
        D = np.diag([std_x**2, std_y**2])
        return R @ D @ R.T

    # === Left-side components ===
    means1 = np.array([
        [-1.0,  l*2/3],   # top left
        [-1.0, -l*2/3]    # bottom left
    ])

    covs1 = [
        make_anisotropic_cov(10, std_x, std_y),   # slight tilt
        make_anisotropic_cov(-10, std_x, std_y)
    ]

    # === Right-side components ===
    means2 = np.array([
        [ 1.0,  l],    # top right
        [ 1.0,  0.0],  # middle right
        [ 1.0, -l]     # bottom right
    ])

    covs2 = [
        make_anisotropic_cov(80, std_x, std_y),   # vertical
        make_anisotropic_cov(45, std_x, std_y),   # diagonal
        make_anisotropic_cov(-75, std_x, std_y)   # other diagonal
    ]
    means1 = torch.tensor(means1, dtype=torch.float32)
    means2 = torch.tensor(means2, dtype=torch.float32)
    covs1 = [torch.tensor(cov, dtype=torch.float32) for cov in covs1]
    covs2 = [torch.tensor(cov, dtype=torch.float32) for cov in covs2]

    dataset = GMM2GMM(
        means1=means1,
        covs1=covs1,
        means2=means2,
        covs2=covs2,
        num_samples=num_samples
    )

    return dataset

def get_second_example_dataset(num_samples):
    import numpy as np
    import torch

    # Parameters
    std0 = 0.4       # std of initial single Gaussian
    std_target = 0.3 # std of each target GMM component
    radius = 2.0      # radius of the circular GMM
    num_components = 5

    # === Single centered Gaussian (source) ===
    means1 = np.array([[0.0, 0.0]])
    covs1 = [np.eye(2) * std0**2]

    # === Circular GMM (target) ===
    angles = np.linspace(0, 2 * np.pi, num_components, endpoint=False)
    means2 = np.stack([
        [radius * np.cos(a), radius * np.sin(a)]
        for a in angles
    ])

    covs2 = [np.eye(2) * std_target**2 for _ in range(num_components)]

    means1 = torch.tensor(means1, dtype=torch.float32)
    means2 = torch.tensor(means2, dtype=torch.float32)
    covs1 = [torch.tensor(cov, dtype=torch.float32) for cov in covs1]
    covs2 = [torch.tensor(cov, dtype=torch.float32) for cov in covs2]

    dataset = GMM2GMM(
        means1=means1,
        covs1=covs1,
        means2=means2,
        covs2=covs2,
        num_samples=num_samples
    )

    return dataset
