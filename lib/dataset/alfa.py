import logging
import numpy as np
from yacs.config import CfgNode

from .base import BaseNumpyDataset
from .transform import build_transforms
from .utils import make_imbalance, split_trainval, x_u_split


def build_alfa_dataset(cfg: CfgNode) -> tuple():
    """
    Build ALFA dataset for aircraft fault detection

    Args:
        cfg: Configuration node

    Returns:
        Tuple of (labeled_train, unlabeled_train, valid, test) datasets
    """
    # fmt: off
    root = cfg.DATASET.ROOT
    algorithm = cfg.ALGORITHM.NAME
    num_l_head = cfg.DATASET.ALFA.NUM_LABELED_HEAD
    num_ul_head = cfg.DATASET.ALFA.NUM_UNLABELED_HEAD
    imb_factor_l = cfg.DATASET.ALFA.IMB_FACTOR_L
    imb_factor_ul = cfg.DATASET.ALFA.IMB_FACTOR_UL
    num_valid = cfg.DATASET.NUM_VALID
    test_split = cfg.DATASET.ALFA.TEST_SPLIT
    reverse_ul_dist = cfg.DATASET.REVERSE_UL_DISTRIBUTION
    num_classes = cfg.MODEL.NUM_CLASSES
    seed = cfg.SEED
    # fmt: on

    logger = logging.getLogger()

    # Load ALFA data from numpy files
    data_path = f"{root}/X_median-resampling_nine_anomalies.npy"
    label_path = f"{root}/y_median-resampling_nine_anomalies.npy"

    logger.info(f"Loading ALFA dataset from {root}")
    X = np.load(data_path)  # Shape: (4529, 25, 35)
    y = np.load(label_path)  # Shape: (4529,)

    logger.info(f"ALFA data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Unique labels: {np.unique(y)}")

    # Create label mapping for missing labels (6, 7, 8)
    # Original labels: [0, 1, 2, 3, 4, 5, 9]
    # Remap to continuous: [0, 1, 2, 3, 4, 5, 6]
    label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 9: 6}
    y_remapped = np.array([label_map[int(label)] for label in y])

    # Create dataset dictionary (use "images" key for compatibility with utils.py)
    data_dict = {"images": X.astype(np.float32), "labels": y_remapped.astype(np.int64)}

    # Split train and test
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    test_size = int(num_samples * test_split)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    # Create test dataset
    test_dict = {
        "images": data_dict["images"][test_indices],
        "labels": data_dict["labels"][test_indices]
    }

    # Create train dataset
    train_dict = {
        "images": data_dict["images"][train_indices],
        "labels": data_dict["labels"][train_indices]
    }

    # Train - valid set split
    alfa_valid = None
    if num_valid > 0:
        l_train, alfa_valid = split_trainval(train_dict, num_valid, seed=seed)
    else:
        l_train = train_dict

    # Unlabeled sample generation under SSL setting
    ul_train = None
    l_train, ul_train = x_u_split(l_train, num_l_head, num_ul_head, seed=seed)
    if algorithm == "Supervised":
        ul_train = None

    # Whether to shuffle the class order
    class_inds = list(range(num_classes))

    # Make synthetic imbalance for labeled set
    if imb_factor_l > 1:
        l_train, class_inds = make_imbalance(
            l_train, num_l_head, imb_factor_l, class_inds, seed=seed
        )

    # Make synthetic imbalance for unlabeled set
    if ul_train is not None and imb_factor_ul > 1:
        ul_train, class_inds = make_imbalance(
            ul_train,
            num_ul_head,
            imb_factor_ul,
            class_inds,
            reverse_ul_dist=reverse_ul_dist,
            seed=seed
        )

    # Build transforms for time series data
    l_trans, ul_trans, eval_trans = build_transforms(cfg, "alfa")

    # Create dataset objects (use default "images" key for compatibility)
    if ul_train is not None:
        ul_train = ALFADataset(ul_train, transforms=ul_trans)

    l_train = ALFADataset(l_train, transforms=l_trans)

    if alfa_valid is not None:
        alfa_valid = ALFADataset(alfa_valid, transforms=eval_trans)

    alfa_test = ALFADataset(test_dict, transforms=eval_trans)

    # Log dataset statistics
    logger.info("class distribution of labeled dataset")
    logger.info(
        ", ".join("idx{}: {}".format(item[0], item[1]) for item in l_train.num_samples_per_class)
    )
    logger.info(
        "=> number of labeled data: {}\n".format(
            sum([item[1] for item in l_train.num_samples_per_class])
        )
    )
    if ul_train is not None:
        logger.info("class distribution of unlabeled dataset")
        logger.info(
            ", ".join(
                ["idx{}: {}".format(item[0], item[1]) for item in ul_train.num_samples_per_class]
            )
        )
        logger.info(
            "=> number of unlabeled data: {}\n".format(
                sum([item[1] for item in ul_train.num_samples_per_class])
            )
        )

    return l_train, ul_train, alfa_valid, alfa_test


class ALFADataset(BaseNumpyDataset):
    """
    ALFA dataset for aircraft fault detection
    Time series data with shape (seq_len=25, features=35)
    """

    def __init__(self, *args, **kwargs):
        # Use default "images" key for compatibility with utils.py
        super(ALFADataset, self).__init__(*args, **kwargs)
