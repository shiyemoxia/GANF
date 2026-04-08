#%%
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

# %%
def load_traffic(root, batch_size):
    """
    Load traffic dataset
    return train_loader, val_loader, test_loader
    """
    df = pd.read_hdf(root)
    df = df.reset_index()
    df = df.rename(columns={"index":"utc"})
    df["utc"] = pd.to_datetime(df["utc"], unit="s")
    df = df.set_index("utc")
    n_sensor = len(df.columns)

    mean = df.values.flatten().mean()
    std = df.values.flatten().std()

    df = (df - mean)/std
    df = df.sort_index()
    # split the dataset
    train_df = df.iloc[:int(0.75*len(df))]
    val_df = df.iloc[int(0.75*len(df)):int(0.875*len(df))]
    test_df = df.iloc[int(0.75*len(df)):]

    train_loader = DataLoader(Traffic(train_df), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Traffic(val_df), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Traffic(test_df), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor  

class Traffic(Dataset):
    def __init__(self, df, window_size=12, stride_size=1):
        super(Traffic, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.time = self.preprocess(df)
    
    def preprocess(self, df):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(5*self.window_size,unit='min')

        return df.values, start_idx[idx_mask], df.index[start_idx[idx_mask]]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1)

def load_water(root, batch_size,label=False):
    
    data = pd.read_csv(root)
    data.columns = [str(col).strip() for col in data.columns]
    data = data.rename(columns={"Normal/Attack":"label"})

    label_series = data["label"]
    if pd.api.types.is_numeric_dtype(label_series):
        data["label"] = label_series.astype(np.int32)
    else:
        data["label"] = (label_series.astype(str).str.strip() != "Normal").astype(np.int32)

    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    data = data.dropna(subset=["Timestamp"])
    data = data.set_index("Timestamp")

    #%%
    feature = data.iloc[:,:51].apply(pd.to_numeric, errors="coerce")
    mean_df = feature.mean(axis=0)
    std_df = feature.std(axis=0)

    norm_feature = (feature-mean_df)/std_df
    norm_feature = norm_feature.ffill().bfill()
    norm_feature = norm_feature.dropna(axis=1)
    n_sensor = len(norm_feature.columns)

    train_df = norm_feature.iloc[:int(0.6*len(data))]
    train_label = data.label.iloc[:int(0.6*len(data))]

    val_df = norm_feature.iloc[int(0.6*len(data)):int(0.8*len(data))]
    val_label = data.label.iloc[int(0.6*len(data)):int(0.8*len(data))]
    
    test_df = norm_feature.iloc[int(0.8*len(data)):]
    test_label = data.label.iloc[int(0.8*len(data)):]
    if label:
        train_loader = DataLoader(WaterLabel(train_df,train_label), batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(Water(train_df,train_label), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Water(val_df,val_label), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Water(test_df,test_label), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor

class Water(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10):
        super(Water, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
    
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        return df.values, start_idx[idx_mask], label[start_idx[idx_mask]]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1)


class WaterLabel(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10):
        super(WaterLabel, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
        self.label = 1.0-2*self.label 
    
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        return df.values, start_idx[idx_mask], label[start_idx[idx_mask]]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1),self.label[index]


SUPPORTED_TS_CFLOW_DATASETS = {"swat", "wadi", "psm", "smd", "msl", "smap"}


def _ensure_ts_cflow_import():
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


class GANFTimeSeriesAdapter(Dataset):
    """Adapt TS-CFLOW window datasets to GANF's [K, L, 1] input layout."""

    def __init__(self, base_dataset, label_reduction="any"):
        super().__init__()
        self.base_dataset = base_dataset
        self.label_reduction = label_reduction
        self.label = pd.Series(self._build_window_labels(), name="label")

    def _reduce_label(self, label_window):
        if self.label_reduction == "any":
            return int(np.any(label_window))
        if self.label_reduction == "first":
            return int(label_window[0])
        if self.label_reduction == "last":
            return int(label_window[-1])
        raise ValueError(
            "Unsupported label_reduction '{}'. Expected one of: any/first/last".format(
                self.label_reduction
            )
        )

    def _build_window_labels(self):
        labels = getattr(self.base_dataset, "labels", None)
        if labels is None:
            return np.zeros(len(self.base_dataset), dtype=np.int32)

        labels = np.asarray(labels, dtype=np.int32)
        window_labels = np.empty(len(self.base_dataset), dtype=np.int32)

        for idx, start_idx in enumerate(self.base_dataset.indices):
            end_idx = start_idx + self.base_dataset.window_size
            window_labels[idx] = self._reduce_label(labels[start_idx:end_idx])

        return window_labels

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = self.base_dataset[index]
        window = sample["data"].float()
        return window.unsqueeze(-1).transpose(0, 1)


class GANFWindowSubset(Dataset):
    """Subset wrapper that keeps window labels accessible for GANF metrics."""

    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label = pd.Series(
            dataset.label.to_numpy(dtype=np.int32, copy=False)[self.indices],
            name="label",
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]


def _split_eval_dataset(dataset, val_ratio):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1), got {}".format(val_ratio))

    n_windows = len(dataset)
    if n_windows < 2:
        raise ValueError("Need at least 2 windows to build validation/test splits")

    split_idx = int(n_windows * val_ratio)
    split_idx = min(max(split_idx, 1), n_windows - 1)
    indices = np.arange(n_windows)
    return (
        GANFWindowSubset(dataset, indices[:split_idx]),
        GANFWindowSubset(dataset, indices[split_idx:]),
    )


def load_timeseries(
    data_dir,
    batch_size,
    dataset_type="swat",
    machine=None,
    window_size=60,
    stride_size=10,
    val_ratio=0.5,
    label_reduction="any",
):
    """Load TS-CFLOW datasets and adapt them to GANF's expected input format."""

    dataset_type = dataset_type.lower()
    if dataset_type not in SUPPORTED_TS_CFLOW_DATASETS:
        raise ValueError(
            "Unsupported dataset_type '{}'. Expected one of: {}".format(
                dataset_type, ", ".join(sorted(SUPPORTED_TS_CFLOW_DATASETS))
            )
        )

    _ensure_ts_cflow_import()
    from ts_cflow.datasets import load_dataset as load_ts_cflow_dataset

    machine = None if machine in (None, "", "None", "none") else machine

    train_base = load_ts_cflow_dataset(
        dataset_type=dataset_type,
        data_path=str(data_dir),
        split="train",
        window_size=window_size,
        stride=stride_size,
        machine=machine,
        normalize=True,
    )
    test_base = load_ts_cflow_dataset(
        dataset_type=dataset_type,
        data_path=str(data_dir),
        split="test",
        window_size=window_size,
        stride=stride_size,
        machine=machine,
        normalize=True,
        norm_stats=train_base.norm_stats,
    )

    train_dataset = GANFTimeSeriesAdapter(train_base, label_reduction=label_reduction)
    eval_dataset = GANFTimeSeriesAdapter(test_base, label_reduction=label_reduction)
    val_dataset, test_dataset = _split_eval_dataset(eval_dataset, val_ratio=val_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    n_sensor = train_base.data.shape[1]

    return train_loader, val_loader, test_loader, n_sensor
