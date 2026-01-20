from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torchcde

DATASET_CONFIG = {
    "kdd": {
        "path": "data/kdd/kdd_norm",
        "n_fft": 16,
    },
    "guangzhou": {
        "path": "data/guangzhou/guangzhou_norm",
        "n_fft": 16,
    },
    "physio2012": {
        "path": "data/physio2012/physio_norm",
        "n_fft": 16,
    },
    "pems07": {
        "path": "data/pems07/pems07_norm",
        "n_fft": 16,
    },
    "pems08": {
        "path": "data/pems08/pems08_norm",
        "n_fft": 16,
    },
}

def compute_stft(data, n_fft):
    """
    Compute STFT for all channels in data.
    Args:
        data: tensor of shape (seq_len, num_channels)
        n_fft: FFT window size
    Returns:
        stft_result: tensor of shape (freq_bins, seq_frames, num_channels)
    """
    if n_fft is None:
        return data.unsqueeze(0)
    
    temp_list = []
    for j in range(data.shape[1]):
        x = data[:, j]
        xf = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=n_fft // 2,
            window=torch.hann_window(n_fft),
            center=False,
            onesided=True,
            return_complex=True
        )
        xf = torch.abs(xf)
        temp_list.append(xf)
    
    return torch.stack(temp_list, dim=2)

def get_mask_file(dataset_name, mode, mechanism, missing_rate, seed):
    dataset_lower = dataset_name.lower()
    
    if dataset_lower in ["pems08", "kdd"]:
        if mode == "train":
            return f"data/mask/{dataset_lower}/{dataset_lower}_train_mcar_{missing_rate}_3407.csv"
        else:
            return f"data/mask/{dataset_lower}/{dataset_lower}_{mode}_{mechanism}_{missing_rate}_{seed}.csv"
    else:
        return f"data/mask/{dataset_lower.replace('_', '')}/{dataset_lower}_{mode}_{mechanism}_{missing_rate}_{seed}.csv"

class BaseDataset(Dataset):
    """Universal dataset class for time series imputation."""
    
    def __init__(self, configs, dataset_name, mode="train"):
        super().__init__()
        self.configs = configs
        self.mode = mode
        self.dataset_name = dataset_name.lower()
        
        if self.dataset_name not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.dataset_cfg = DATASET_CONFIG[self.dataset_name]
        self._load_data()
        self._load_mask()
        self._compute_frequency_domain()
    
    def _load_data(self):
        """Load time series data."""
        mode_suffix = "_train" if self.mode == "train" else f"_{self.mode}"
        data_file = f"{self.dataset_cfg['path']}{mode_suffix}.csv"
        
        data_raw = np.loadtxt(data_file, delimiter=",")
        self.data = data_raw.reshape(-1, self.configs.seq_len, self.configs.enc_in)
        print(f"Loaded {self.dataset_name} {self.mode}: {self.data.shape}")
    
    def _load_mask(self):
        """Load or generate mask."""
        if self.configs.missing_rate == 0:
            self.mask = np.ones_like(self.data)
            self.mask[np.where(self.data == -200)] = 0
        else:
            mask_file = get_mask_file(
                self.dataset_name,
                self.mode,
                self.configs.mechanism,
                self.configs.missing_rate,
                self.configs.seed
            )
            mask_raw = np.loadtxt(mask_file, delimiter=",")
            self.mask = mask_raw.reshape(-1, self.configs.seq_len, self.configs.enc_in)
        
        self.mask_gt = np.ones_like(self.data)
        self.mask_gt[np.where(self.data == -200)] = 0
    
    def _compute_frequency_domain(self):
        """Compute frequency domain representation."""
        self.dataf = np.array(self.data, dtype=np.float32)
        self.dataf[np.where(self.mask == 0)] = np.nan
        self.dataf = torch.from_numpy(self.dataf).float()
        self.dataf = torchcde.linear_interpolation_coeffs(self.dataf)
        
        if self.dataset_cfg["n_fft"] is None:
            n_fft = self.configs.n_fft if hasattr(self.configs, 'n_fft') else 24
        else:
            n_fft = self.dataset_cfg["n_fft"]
        
        stft_list = []
        for idx in range(self.dataf.shape[0]):
            stft_result = compute_stft(self.dataf[idx], n_fft)
            stft_list.append(stft_result)
        
        self.dataf = torch.stack(stft_list, dim=0)
        print(f"STFT shape: {self.dataf.shape}")
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        data_res = torch.from_numpy(self.data[index]).float()
        dataf_res = self.dataf[index]
        mask_res = torch.from_numpy(self.mask[index]).float()
        observed_tp = torch.from_numpy(np.arange(self.configs.seq_len)).float()
        mask_gt = torch.from_numpy(self.mask_gt[index]).float()
        
        return data_res, dataf_res, mask_res, observed_tp, mask_gt

def get_dataloaders(configs):
    """
    Get train/valid/test dataloaders for specified dataset.
    """
    dataset_name = configs.dataset.lower()
    
    train_dataset = BaseDataset(configs, dataset_name, mode="train")
    valid_dataset = BaseDataset(configs, dataset_name, mode="valid")
    test_dataset = BaseDataset(configs, dataset_name, mode="test")
    
    train_loader = DataLoader(train_dataset, batch_size=configs.batch, num_workers=0, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch, num_workers=0, shuffle=False)
    
    return train_loader, valid_loader, test_loader

def get_dataset(configs):
    """Wrapper for backward compatibility."""
    return get_dataloaders(configs)
