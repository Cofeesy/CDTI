import torch
import numpy as np
import train_utils as trainer
import argparse
import datasets as A_dataset

parser = argparse.ArgumentParser(description='CDTI')

parser.add_argument('--device', type=str, default="cuda:0", help='device to use (cuda:0, cpu, etc.)')
parser.add_argument('--batch', type=int, default=16, help='input batch size')

parser.add_argument('--dataset', type=str, default="pems08", help='dataset name (guangzhou, pems07, pems08, kdd, physionet)')
parser.add_argument('--missing_rate', type=float, default=0.1, help='missing percent for experiment')
parser.add_argument('--seed', type=int, default=3407, help='random seed')
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--enc_in', type=int, default=None, help='encoder input size (auto-set if None)')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')

parser.add_argument('--diffusion_step_num', type=int, default=50, help='total number of diffusion step')
parser.add_argument('--timeemb', type=int, default=128, help='side information timeemb dimension')
parser.add_argument('--featureemb', type=int, default=16, help='side information featureemb dimension')
parser.add_argument('--nheads', type=int, default=8, help='number of head for attention')
parser.add_argument('--channel', type=int, default=128, help='channel dimension of diffusion')
parser.add_argument('--proj_t', type=int, default=128, help='proj_t for feature self-attention')
parser.add_argument('--residual_layers', type=int, default=4, help='number of residual layers in diffusion model')
parser.add_argument('--schedule', type=str, default='quad', help='beta increase schedule')
parser.add_argument('--beta_start', type=float, default=0.0001)
parser.add_argument('--beta_end', type=float, default=0.2)

parser.add_argument('--n_fft', type=int, default=16, help='n_fft for stft')

parser.add_argument('--epoch_diff', type=int, default=200, help='training epoch for diffusion training')
parser.add_argument('--learning_rate_diff', type=float, default=1e-3, help='learning rate of diffusion training')
parser.add_argument('--valid_epoch_interval', type=int, default=50, help='per epoch for val')

parser.add_argument('--mechanism', type=str, default="mcar", help='mechanism of missing data (mcar, mar, mnar)')

if __name__ == '__main__':
    configs = parser.parse_args()

    # Auto-configure enc_in based on dataset
    dataset_config = {
        'guangzhou': 214,
        'pems07': 883,
        'pems08': 170,
        'kdd': 99,
        'physionet': 37,
    }
    
    if configs.enc_in is None:
        configs.enc_in = dataset_config.get(configs.dataset, 170)
    
    # Validate parameters
    assert 0 < configs.missing_rate < 1.0, "missing_rate must be between 0 and 1"
    assert configs.mechanism in ['mcar', 'mar', 'mnar'], "mechanism must be mcar, mar, or mnar"
    assert configs.dataset in dataset_config.keys(), f"dataset must be one of {list(dataset_config.keys())}"
    
    print("\nTraining Configuration:")
    print(f"  Dataset: {configs.dataset}")
    print(f"  Missing rate: {configs.missing_rate}")
    print(f"  Missing mechanism: {configs.mechanism}")
    print(f"  Feature dimension: {configs.enc_in}")
    print(f"  Device: {configs.device}")
    print(f"  Epochs: {configs.epoch_diff}\n")

    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    torch.cuda.manual_seed(configs.seed)
    torch.cuda.manual_seed_all(configs.seed)
    
    train_loader, val_loader, test_loader = A_dataset.get_dataset(configs)

    model = trainer.diffusion_train(configs, train_loader, val_loader)
    print("\nTesting model...")
    trainer.diffusion_test(configs, model, test_loader)

