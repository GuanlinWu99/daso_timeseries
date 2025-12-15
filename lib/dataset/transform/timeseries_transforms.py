import numpy as np
import torch


class TimeSeriesNormalize:
    """
    Normalize time series data
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        """
        Args:
            x: numpy array of shape (seq_len, features)
        Returns:
            Normalized tensor
        """
        x = torch.from_numpy(x).float()
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        return x


class TimeSeriesGaussianNoise:
    """
    Add Gaussian noise to time series data (data augmentation)
    """
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std

    def __call__(self, x):
        """
        Args:
            x: tensor of shape (seq_len, features)
        Returns:
            Tensor with added noise
        """
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x


class TimeSeriesScale:
    """
    Random scaling augmentation for time series
    """
    def __init__(self, scale_range=(0.95, 1.05)):
        self.scale_range = scale_range

    def __call__(self, x):
        """
        Args:
            x: tensor of shape (seq_len, features)
        Returns:
            Scaled tensor
        """
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return x * scale


class TimeSeriesCompose:
    """
    Compose multiple transforms
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class TimeSeriesIdentity:
    """
    Identity transform (no operation)
    """
    def __call__(self, x):
        return torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x


class TimeSeriesSSLTransform:
    """
    SSL Transform for time series that returns multiple augmented versions
    Similar to GeneralizedSSLTransform for images
    """
    def __init__(self, transforms_list):
        """
        Args:
            transforms_list: List of transform functions
        """
        assert len(transforms_list) > 0
        self.transforms = transforms_list

    def __call__(self, x):
        """
        Args:
            x: numpy array of shape (seq_len, features)
        Returns:
            Tuple of transformed tensors if multiple transforms, single tensor otherwise
        """
        results = []
        for t in self.transforms:
            results.append(t(x))
        if len(results) == 1:
            return results[0]
        return tuple(results)


def get_timeseries_transforms(strong_aug=False, for_unlabeled=False):
    """
    Get transforms for time series data

    Args:
        strong_aug: Whether to apply strong augmentation
        for_unlabeled: Whether this is for unlabeled data (returns weak + strong)

    Returns:
        Transform function
    """
    # Weak augmentation (just normalization)
    weak_transform = TimeSeriesNormalize()

    # Strong augmentation
    strong_transform = TimeSeriesCompose([
        TimeSeriesNormalize(),
        TimeSeriesGaussianNoise(noise_std=0.02),
        TimeSeriesScale(scale_range=(0.9, 1.1))
    ])

    if for_unlabeled:
        # Return both weak and strong for SSL algorithms
        return TimeSeriesSSLTransform([weak_transform, strong_transform])
    elif strong_aug:
        return strong_transform
    else:
        return weak_transform
