"""
Healthcare dataset loader for PhysioNet Challenge 2012 data.

This module provides utilities for loading and preprocessing the PhysioNet
Challenge 2012 ICU dataset for time series imputation tasks.
"""

import os
import pickle
import re

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


# 35 attributes which contains enough non-values
attributes = [
    'DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH',
    'Albumin', 'ALT', 'Glucose', 'SaO2', 'Temp', 'AST', 'Bilirubin', 'HCO3',
    'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS', 'Cholesterol',
    'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine',
    'NIMAP', 'Creatinine', 'ALP'
]


def extract_hour(x):
    """Extract hour from time string.
    
    Args:
        x: Time string in "HH:MM" format
        
    Returns:
        Hour as integer
    """
    h, _ = map(int, x.split(":"))
    return h


def parse_data(x):
    """Extract the last value for each attribute from a dataframe.
    
    Args:
        x: DataFrame with 'Parameter' and 'Value' columns
        
    Returns:
        List of values for each attribute (NaN if missing)
    """
    # extract the last value for each attribute
    x = x.set_index("Parameter").to_dict()["Value"]

    values = []

    for attr in attributes:
        if x.__contains__(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values


def parse_id(id_, missing_ratio=0.1):
    """Parse data for a specific patient ID.
    
    Args:
        id_: Patient ID string
        missing_ratio: Ratio of values to mask as missing for testing
        
    Returns:
        Tuple of (observed_values, observed_masks, gt_masks)
    """
    data = pd.read_csv("./data/physio/set-a/{}.txt".format(id_))
    # set hour
    data["Time"] = data["Time"].apply(lambda x: extract_hour(x))

    # create data for 48 hours x 35 attributes
    observed_values = []
    for h in range(48):
        observed_values.append(parse_data(data[data["Time"] == h]))
    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)

    # randomly set some percentage as ground-truth
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    )
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def get_idlist():
    """Get list of patient IDs from the data directory.
    
    Returns:
        Sorted array of patient ID strings
    """
    patient_id = []
    for filename in os.listdir("./data/physio/set-a"):
        match = re.search(r"\d{6}", filename)
        if match:
            patient_id.append(match.group())
    patient_id = np.sort(patient_id)
    return patient_id


class Physio_Dataset(Dataset):
    """PyTorch Dataset for PhysioNet Challenge 2012 data.
    
    Loads and preprocesses the healthcare time series data for imputation.
    """

    def __init__(self, eval_length=48, use_index_list=None, missing_ratio=0.0, seed=0):
        """Initialize the dataset.
        
        Args:
            eval_length: Length of time series (48 hours)
            use_index_list: Optional list of indices to use
            missing_ratio: Ratio of values to mask as missing
            seed: Random seed for reproducibility
        """
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        path = (
            "./data/physio_missing" + str(missing_ratio) + "_seed" + str(seed) + ".pk"
        )

        if not os.path.isfile(path):  # if datasetfile is none, create
            idlist = get_idlist()
            for id_ in idlist:
                try:
                    observed_values, observed_masks, gt_masks = parse_id(
                        id_, missing_ratio
                    )
                    self.observed_values.append(observed_values)
                    self.observed_masks.append(observed_masks)
                    self.gt_masks.append(gt_masks)
                except Exception as e:
                    print(id_, e)
                    continue
            self.observed_values = np.array(self.observed_values)
            self.observed_masks = np.array(self.observed_masks)
            self.gt_masks = np.array(self.gt_masks)

            # calc mean and std and normalize values
            # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
            tmp_values = self.observed_values.reshape(-1, 35)
            tmp_masks = self.observed_masks.reshape(-1, 35)
            mean = np.zeros(35)
            std = np.zeros(35)
            for k in range(35):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                mean[k] = c_data.mean()
                std[k] = c_data.std()
            self.observed_values = (
                (self.observed_values - mean) / std * self.observed_masks
            )

            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks], f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                    f
                )
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        """Get a sample from the dataset.
        
        Args:
            org_index: Index of the sample
            
        Returns:
            Dictionary with observed_data, observed_mask, gt_mask, and timepoints
        """
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        """Get the number of samples in the dataset."""
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1):
    """Create data loaders for training, validation, and testing.
    
    Uses 5-fold cross-validation split with 70% training, 10% validation,
    and 20% testing.
    
    Args:
        seed: Random seed for reproducibility
        nfold: Fold number for cross-validation (0-4)
        batch_size: Batch size for data loaders
        missing_ratio: Ratio of values to mask as missing
        
    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    # only to obtain total length of dataset
    dataset = Physio_Dataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    dataset = Physio_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Physio_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Physio_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader
