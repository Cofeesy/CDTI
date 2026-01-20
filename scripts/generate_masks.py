import numpy as np
import argparse
from mask_utils import generate_masks, calc_missing_rate

# Dataset configuration
DATASET_CONFIG = {
    'guangzhou': {
        'data_path': 'data/guangzhou/guangzhou_norm_{mode}.csv',
        'mask_path': 'data/mask/guangzhou/guangzhou_{mode}_{mechanism}_{missing_p}_{seed}.csv',
    },
    'pems07': {
        'data_path': 'data/pems07/pems07_norm_{mode}.csv',
        'mask_path': 'data/mask/pems07/pems07_{mode}_{mechanism}_{missing_p}_{seed}.csv',
    },
    'pems08': {
        'data_path': 'data/pems08/pems08_norm_{mode}.csv',
        'mask_path': 'data/mask/pems08/pems08_{mode}_{mechanism}_{missing_p}_{seed}.csv',
    },
    'kdd': {
        'data_path': 'data/kdd/kdd_norm_{mode}.csv',
        'mask_path': 'data/mask/kdd/kdd_{mode}_{mechanism}_{missing_p}_{seed}.csv',
    },
    'physionet': {
        'data_path': 'data/physio2012/physio_{mode}_norm.csv',
        'mask_path': 'data/mask/physio2012/physio_{mode}_{mechanism}_{missing_p}_{seed}.csv',
    },
}


def generate_dataset_masks(dataset, missing_p, seeds, mechanisms, modes):
    """
    Generate masks for a specific dataset.
    
    Args:
        dataset: Dataset name
        missing_p: Missing percentage
        seeds: List of random seeds
        mechanisms: List of missing mechanisms (mcar, mar, mnar)
        modes: List of data modes (train, valid, test)
    """
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Dataset {dataset} not supported. Choose from {list(DATASET_CONFIG.keys())}")
    
    config = DATASET_CONFIG[dataset]
    
    for mode in modes:
        print(f"\nGenerating masks for {dataset} {mode} set...")
        data_file = config['data_path'].format(mode=mode)
        
        try:
            data = np.loadtxt(data_file, delimiter=",")
            print(f"  Loaded data shape: {data.shape}")
        except FileNotFoundError:
            print(f"  Error: File not found - {data_file}")
            continue
        
        for seed in seeds:
            masks = generate_masks(data, missing_p, seed, mechanisms)
            
            for mechanism in mechanisms:
                mask_file = config['mask_path'].format(
                    mode=mode,
                    mechanism=mechanism,
                    missing_p=missing_p,
                    seed=seed
                )
                np.savetxt(mask_file, masks[mechanism], fmt="%d", delimiter=",")
                
                missing_rate = calc_missing_rate(masks[mechanism])
                print(f"    Seed {seed} ({mechanism}): {missing_rate:.4f} - Saved to {mask_file}")
    
    print(f"\n✓ {dataset} mask generation completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate missing data masks for datasets')
    
    parser.add_argument('--dataset', type=str, default='guangzhou', 
                        help='Dataset name (guangzhou, pems07, pems08, kdd, physionet)')
    parser.add_argument('--missing_rate', type=float, default=0.1,
                        help='Missing percentage (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed for mask generation')
    parser.add_argument('--mechanisms', type=str, nargs='+', default=['mcar', 'mar', 'mnar'],
                        help='Missing mechanisms to generate')
    parser.add_argument('--modes', type=str, nargs='+', default=['train', 'valid', 'test'],
                        help='Data modes to generate masks for')
    
    args = parser.parse_args()
    
    # Validate parameters
    assert 0 < args.missing_rate < 1.0, "missing_rate must be between 0 and 1"
    valid_mechanisms = ['mcar', 'mar', 'mnar']
    for m in args.mechanisms:
        assert m in valid_mechanisms, f"mechanism {m} not supported, choose from {valid_mechanisms}"
    
    print(f"\nMask Generation Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Missing rate: {args.missing_rate}")
    print(f"  Mechanisms: {args.mechanisms}")
    print(f"  Seed: {args.seed}")
    print(f"  Modes: {args.modes}\n")
    
    generate_dataset_masks(args.dataset, args.missing_rate, [args.seed], args.mechanisms, args.modes)
    
    print("\n" + "="*60)
    print("All mask generation completed!")
    print("="*60)
