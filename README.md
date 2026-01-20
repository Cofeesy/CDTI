# CDTI - Time-Frequency Conditioned Diffusion for Multivariate Time Series Imputation


## Installation

```bash
pip install -r requirements.txt
```


## 1. Data Preparation

### Download Datasets

### Download Datasets

- **Guangzhou**: https://zenodo.org/record/1205229
  
- **PEMS07 & PEMS08**: https://zenodo.org/record/3939792

- **KDD**: http://www.kdd.org/kdd2018/

- **Physio2012**: https://physionet.org/content/challenge-2012/1.0.0/

### Preprocess Data

Before training, get the splited dataset:

```bash
# Preprocess PEMS08 dataset
python preprocess/pems08.py
```

```bash
# Preprocess KDD dataset
python preprocess/kdd.py
```

```bash
# Preprocess GuangZhou dataset
python preprocess/guangzhou.py
```

```bash
# Preprocess physionet2012 dataset
python preprocess/physionet12.py
```

```bash
# Preprocess PEMS07 dataset
python preprocess/pems07.py
```

## 2. Mask Generation

After preprocessing, generate masks for different missing data mechanisms using the unified script:

```bash
# Generate masks for a single dataset
python scripts/generate_masks.py --dataset guangzhou

python scripts/generate_masks.py --dataset pems07

python scripts/generate_masks.py --dataset pems08

# Custom parameters
python scripts/generate_masks.py --dataset pems08 --missing_rate 0.2 --mechanisms mcar mar
```

The mask generation supports three mechanisms:
- **MCAR** (Missing Completely At Random): Data missing without any pattern
- **MAR** (Missing At Random): Data missing based on observed values
- **MNAR** (Missing Not At Random): Data missing based on unobserved values

Masks are saved to `data/mask/{dataset}/` directory with filename format:
`{dataset}_{mode}_{mechanism}_{missing_rate}_{seed}.csv`


## 3. Demo Running

Run the training script with desired parameters:
```
python train.py --dataset pems08 enc_in=170 --device cuda:0
```

```
python train.py --dataset kdd enc_in=99 --device cuda:0
```

```
python train.py --dataset phy enc_in=37 --device cuda:0
```

```
python train.py --dataset guangzhou enc_in=214 --device cuda:0
```

```
python train.py --dataset pems07 enc_in=883 --device cuda:0
```